"""Evaluation metrics over labels, Fraud Scores and Age Groups.

Every function is pure. ``labels`` are 1 for a Fraudulent Application and 0 for a
Legitimate Application; ``scores`` are Fraud Scores, higher meaning riskier. An
Application is flagged (sent to Review) when its Fraud Score is at or above the
Threshold.
"""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score

TARGET_FPR = 0.05
"""Share of Legitimate Applications the Threshold may send to Review."""

AGE_GROUP_BOUNDARY = 50
"""Applicants with ``customer_age`` at or above this form one Age Group, the rest the other."""


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
    # Round before flooring so that, e.g., 0.29 * 100 = 28.999999999999996 allows 29.
    allowed = math.floor(round(target_fpr * legitimate.size, 9))
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

    With the default target this is Recall@5%FPR.
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
    legitimate = y == 0
    older = age >= AGE_GROUP_BOUNDARY
    rates = (
        _flag_rate(s[legitimate & older], threshold, f"Legitimate aged {AGE_GROUP_BOUNDARY}+"),
        _flag_rate(s[legitimate & ~older], threshold, f"Legitimate under {AGE_GROUP_BOUNDARY}"),
    )
    lower, higher = sorted(rates)
    return 1.0 if higher == 0 else lower / higher


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
