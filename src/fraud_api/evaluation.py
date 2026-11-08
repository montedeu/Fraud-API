"""Evaluation metrics over labels, Fraud Scores and Age Groups.

Every function is pure. ``labels`` are 1 for a Fraudulent Application and 0 for a
Legitimate Application; ``scores`` are Fraud Scores, higher meaning riskier. An
Application is flagged (sent to Review) when its Fraud Score is at or above the
Threshold.
"""

import math
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score

TARGET_FPR = 0.05
"""Share of Legitimate Applications the Threshold may send to Review."""

AGE_GROUP_BOUNDARY = 50
"""Applicants with ``customer_age`` at or above this form one Age Group, the rest the other."""

BOOTSTRAP_RESAMPLES = 1000
"""How many resamples of the Test Months the paired bootstrap draws."""
CONFIDENCE = 0.95
"""Coverage of the bootstrap's percentile intervals."""
BOOTSTRAP_METRICS = ("recall_at_5pct_fpr", "pr_auc", "fpr_ratio")
"""The metrics the paired bootstrap puts intervals on."""


def threshold_for_fpr(
    labels: ArrayLike, scores: ArrayLike, target_fpr: float = TARGET_FPR
) -> float:
    """The lowest Threshold that flags at most ``target_fpr`` of Legitimate Applications.

    Legitimate Applications tied on a Fraud Score are flagged together, so when a tie
    would push the FPR past the target, the Threshold moves just above the tied score
    and the actual FPR falls below the target.
    """
    if not 0 < target_fpr < 1:
        raise ValueError(f"target_fpr must be between 0 and 1, got {target_fpr}")
    y, s = _validate(labels, scores)
    legitimate = np.sort(s[y == 0])[::-1]
    if legitimate.size == 0:
        raise ValueError("no Legitimate Applications to set a Threshold on")
    # Exact decimal arithmetic: 0.29 * 100 is 28.999999999999996 in floating point, but
    # the target as written, 29/100, allows 29 flags.
    allowed = math.floor(Fraction(str(float(target_fpr))) * legitimate.size)
    # Flagging stays within budget exactly when the Threshold is above the score of
    # the first Legitimate Application past the budget.
    return float(np.nextafter(legitimate[allowed], np.inf))


def fpr_at(labels: ArrayLike, scores: ArrayLike, threshold: float) -> float:
    """The share of Legitimate Applications flagged at ``threshold``."""
    y, s = _validate(labels, scores)
    return _flag_rate(s[y == 0], threshold, "Legitimate")


def recall_at(labels: ArrayLike, scores: ArrayLike, threshold: float) -> float:
    """The share of Fraudulent Applications flagged at ``threshold``."""
    y, s = _validate(labels, scores)
    return _flag_rate(s[y == 1], threshold, "Fraudulent")


def recall_at_fpr(labels: ArrayLike, scores: ArrayLike, target_fpr: float = TARGET_FPR) -> float:
    """Recall at the Threshold chosen on the same Applications for ``target_fpr``.

    With the default target this is Recall@5%FPR. The Threshold here is fitted to these
    Applications; to score the Test Months with the Threshold chosen on the Validation
    Month (ADR-0001), use ``recall_at`` and ``fpr_at`` with that Threshold instead.
    """
    return recall_at(labels, scores, threshold_for_fpr(labels, scores, target_fpr))


def pr_auc(labels: ArrayLike, scores: ArrayLike) -> float:
    """Area under the precision-recall curve, as average precision."""
    y, s = _validate(labels, scores)
    if not y.any():
        raise ValueError("no Fraudulent Applications to compute PR-AUC on")
    return float(average_precision_score(y, s))


def fpr_ratio(
    labels: ArrayLike, scores: ArrayLike, customer_age: ArrayLike, threshold: float
) -> float:
    """The lower Age Group FPR divided by the higher, at ``threshold``.

    1 means Legitimate Applications from both Age Groups are flagged equally often,
    including when neither group has any flagged.
    """
    y, s = _validate(labels, scores)
    age = np.asarray(customer_age, dtype=float)
    if age.shape != y.shape:
        raise ValueError("customer_age must have one value per Application")
    if not np.isfinite(age).all():
        raise ValueError("customer_age must be finite")
    legitimate = y == 0
    older = age >= AGE_GROUP_BOUNDARY
    rates = (
        _flag_rate(s[legitimate & older], threshold, f"Legitimate aged {AGE_GROUP_BOUNDARY}+"),
        _flag_rate(s[legitimate & ~older], threshold, f"Legitimate under {AGE_GROUP_BOUNDARY}"),
    )
    lower, higher = sorted(rates)
    return 1.0 if higher == 0 else lower / higher


def paired_bootstrap(
    labels: ArrayLike,
    customer_age: ArrayLike,
    scores: Mapping[str, ArrayLike],
    thresholds: Mapping[str, float],
    *,
    reference: str,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Percentile intervals for Recall@5%FPR, PR-AUC and the FPR Ratio of every model, and
    for the difference between ``reference`` and each other model.

    Applications are resampled with replacement, the same resamples for every model, so
    the differences are paired. Each model keeps its own fixed Threshold for the FPR
    Ratio; Recall@5%FPR picks its own cut on each resample, by its definition. The
    intervals cover sampling noise in these Applications only, not drift between Months
    or a refit of the model. Returns
    ``{model: {"interval": {metric: [low, high]}, "difference": {...}}}``; the
    ``reference`` model has no ``difference``.
    """
    y = np.asarray(labels).astype(int)
    age = np.asarray(customer_age, dtype=float)
    columns = {name: np.asarray(s, dtype=float) for name, s in scores.items()}
    if reference not in columns:
        raise ValueError(f"reference {reference!r} is not one of the models")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be at least 1, got {n_resamples}")
    rng = np.random.default_rng(seed)
    draws: dict[str, list[dict[str, float]]] = {name: [] for name in columns}
    for _ in range(n_resamples):
        sample = rng.integers(0, y.size, y.size)
        ys, ages = y[sample], age[sample]
        for name, s in columns.items():
            draws[name].append(
                {
                    "recall_at_5pct_fpr": recall_at_fpr(ys, s[sample]),
                    "pr_auc": pr_auc(ys, s[sample]),
                    "fpr_ratio": fpr_ratio(ys, s[sample], ages, thresholds[name]),
                }
            )
    tail = (1 - CONFIDENCE) / 2 * 100

    def interval(values: list[float]) -> list[float]:
        low, high = np.percentile(values, [tail, 100 - tail])
        return [float(low), float(high)]

    metrics = BOOTSTRAP_METRICS
    result: dict[str, dict[str, Any]] = {}
    for name in columns:
        result[name] = {"interval": {m: interval([d[m] for d in draws[name]]) for m in metrics}}
        if name != reference:
            result[name]["difference"] = {
                m: interval(
                    [r[m] - o[m] for r, o in zip(draws[reference], draws[name], strict=True)]
                )
                for m in metrics
            }
    return result


def _validate(labels: ArrayLike, scores: ArrayLike) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or y.shape != s.shape:
        raise ValueError("labels and scores must be 1-D and the same length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 (Legitimate) or 1 (Fraudulent)")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    return y.astype(int), s


def _flag_rate(group_scores: NDArray[np.float64], threshold: float, group: str) -> float:
    if group_scores.size == 0:
        raise ValueError(f"no {group} Applications")
    return float(np.mean(group_scores >= threshold))
