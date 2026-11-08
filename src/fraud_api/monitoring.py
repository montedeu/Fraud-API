"""Drift monitoring from the Prediction Log.

The training pipeline records how every feature was distributed in the Validation Month
(the Month the Threshold was chosen on) and what share of its Applications the
Threshold sent to Review. ``fraud-api monitor`` compares each day of the Prediction Log
with that reference: Applications scored, the share sent to Review, the mean Fraud
Score, and the features whose distribution moved, by the population stability index.
Every function here except :func:`read_prediction_log` is pure.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import psycopg

PSI_WARNING = 0.1
"""The usual warning level, as in the EDA notebook's drift section."""
MIN_APPLICATIONS_FOR_PSI = 500
"""Below this many Applications a day PSI is not reported: sampled from the Validation
Month itself, 100 Applications push several features past the warning level by chance
alone, 500 practically none."""
MISSING = "missing"
DISCRETE_MAX_VALUES = 12
"""Fields with at most this many values are compared value by value, the rest by decile."""
_FLOOR = 1e-4
"""Share used for an empty bin, so PSI stays finite."""

Reference = dict[str, Any]
PREDICTION_LOG_COLUMNS = ["created_at", "model_version", "application", "fraud_score", "decision"]
"""The Prediction Log columns the report reads."""


def reference_distribution(
    applications: pd.DataFrame, categories: Mapping[str, Sequence[str]]
) -> dict[str, Reference]:
    """For each field: its value shares (categories and fields with few values) or its
    decile edges and the share in each decile, with missing as a bin of its own."""
    reference: dict[str, Reference] = {}
    for field in applications.columns:
        values = applications[field]
        if field in categories or values.nunique() <= DISCRETE_MAX_VALUES:
            reference[field] = {"kind": "values", "shares": _value_shares(values)}
        else:
            present = values.dropna().astype(float)
            edges = np.unique(np.quantile(present, np.linspace(0, 1, 11)))[1:-1].tolist()
            reference[field] = {"kind": "deciles", "edges": edges}
            reference[field]["shares"] = bin_shares(values, reference[field])
    return reference


def bin_shares(values: pd.Series, reference: Reference) -> dict[str, float]:
    """The share of ``values`` in each of ``reference``'s bins."""
    if reference["kind"] == "values":
        return _value_shares(values)
    numeric = pd.to_numeric(values, errors="coerce")
    present = numeric.notna()
    bins = np.searchsorted(reference["edges"], numeric[present].to_numpy(), side="right")
    keys = [f"decile {b + 1}" for b in bins] + [MISSING] * int((~present).sum())
    return _share_of_each(keys)


def psi(reference: Mapping[str, float], observed: Mapping[str, float]) -> float:
    """Population stability index of ``observed`` against ``reference``; 0 when equal."""
    total = 0.0
    for key in set(reference) | set(observed):
        r = max(reference.get(key, 0.0), _FLOOR)
        o = max(observed.get(key, 0.0), _FLOOR)
        total += (o - r) * math.log(o / r)
    return total


def drift(reference: Mapping[str, Reference], applications: pd.DataFrame) -> dict[str, float]:
    """PSI of every reference field in ``applications``, largest first."""
    result = {
        field: psi(entry["shares"], bin_shares(applications[field], entry))
        for field, entry in reference.items()
    }
    return dict(sorted(result.items(), key=lambda kv: -kv[1]))


def report(
    predictions: pd.DataFrame,
    monitoring: Mapping[str, Any],
    *,
    model_version: str,
    exclude: Iterable[Mapping[str, Any]] = (),
) -> str:
    """A Markdown report, one row per UTC day, of the Prediction Log rows in
    ``predictions`` (:data:`PREDICTION_LOG_COLUMNS`) scored by ``model_version``.
    Applications equal to one in ``exclude`` (the smoke test's reference Applications)
    are left out."""
    excluded = {_frozen(application) for application in exclude}
    rows = predictions[
        (predictions["model_version"] == model_version)
        & ~predictions["application"].map(_frozen).isin(excluded)
    ]
    expected = monitoring["expected_review_share"]
    lines = [
        f"# Prediction Log monitoring: Model Version `{model_version}`",
        "",
        f"Reference: Validation Month {monitoring['reference_month']}. The Threshold, chosen "
        f"there for 5% FPR, sent {expected:.1%} of all its Applications to Review. Features "
        f"are listed when their PSI against that Month exceeds {PSI_WARNING}, on days with "
        f"at least {MIN_APPLICATIONS_FOR_PSI} Applications. Days are UTC.",
        "",
        f"| Day (UTC) | Applications | Sent to Review (Validation Month: {expected:.1%}) "
        "| Mean Fraud Score | Drifted features (PSI) |",
        "|---|---:|---:|---:|---|",
    ]
    if rows.empty:
        lines.append("| no Applications in this period | | | | |")
        return "\n".join(lines) + "\n"
    dates = pd.to_datetime(rows["created_at"], utc=True).dt.date
    for day, group in rows.groupby(dates, sort=True):
        review = (group["decision"] == "review").mean()
        if len(group) < MIN_APPLICATIONS_FOR_PSI:
            drifted = "too few to judge"
        else:
            applications = pd.DataFrame(list(group["application"]))
            moved = {
                field: value
                for field, value in drift(monitoring["features"], applications).items()
                if value > PSI_WARNING
            }
            drifted = ", ".join(f"`{f}` {v:.2f}" for f, v in moved.items()) or "none"
        lines.append(
            f"| {day} | {len(group):,} | {review:.1%} | {group['fraud_score'].mean():.4f} "
            f"| {drifted} |"
        )
    return "\n".join(lines) + "\n"


def read_prediction_log(database_url: str, days: int) -> pd.DataFrame:
    """The Prediction Log's rows from the last ``days`` days."""
    with psycopg.connect(database_url) as connection:
        rows = connection.execute(
            f"SELECT {', '.join(PREDICTION_LOG_COLUMNS)} FROM prediction_log "
            "WHERE created_at >= now() - make_interval(days => %s) ORDER BY created_at",
            (days,),
        ).fetchall()
    return pd.DataFrame(rows, columns=PREDICTION_LOG_COLUMNS)


def _key(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return MISSING
    if isinstance(value, str):
        return value
    return f"{float(value):g}"


def _value_shares(values: pd.Series) -> dict[str, float]:
    return _share_of_each([_key(v) for v in values.astype(object)])


def _share_of_each(keys: list[str]) -> dict[str, float]:
    if not keys:
        return {}
    counts = pd.Series(keys).value_counts(normalize=True)
    return {key: float(share) for key, share in sorted(counts.items())}


def _frozen(application: Mapping[str, Any]) -> tuple:
    return tuple(sorted((field, _key(value)) for field, value in application.items()))
