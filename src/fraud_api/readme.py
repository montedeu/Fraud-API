"""The README's numbers, generated from the committed Model Version and the service code.

Each generated part of ``README.md`` sits between ``<!-- generated:NAME -->`` and
``<!-- /generated:NAME -->``. ``fraud-api readme`` rewrites those parts; a test fails
when they no longer match ``model/metadata.json`` (and the service's limits), so the
README cannot drift from the model it describes. Text outside the markers is written by
hand and holds no results.
"""

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fraud_api import protection
from fraud_api.application import EXAMPLE_APPLICATION

README = Path("README.md")
SERVICE_URL = "https://fraud-api-r6bn3ptcuq-ew.a.run.app"
BAF_REFERENCE_FPR_RATIO = "about 0.3"
"""Reported by the BAF authors for their best LightGBM models on Base (NeurIPS 2022)."""

Metadata = dict[str, Any]

_MODELS: dict[str, Callable[[Metadata], Metadata]] = {
    "LightGBM (served)": lambda metadata: metadata,
    "Logistic regression": lambda metadata: metadata["baselines"]["logistic_regression"],
    "`credit_risk_score` rule": lambda metadata: metadata["baselines"]["credit_risk_score_rule"],
    "LightGBM without `customer_age` (experiment)": lambda metadata: metadata["experiments"][
        "without_age"
    ],
}


def render(readme: str, metadata: Metadata) -> str:
    """``readme`` with every generated part rewritten from ``metadata``."""
    for name, part in _PARTS.items():
        pattern = re.compile(
            rf"(<!-- generated:{name} -->\n).*?(<!-- /generated:{name} -->)", re.DOTALL
        )
        match = pattern.search(readme)
        if not match:
            raise ValueError(f"README has no generated part {name!r}")
        readme = readme[: match.end(1)] + part(metadata) + readme[match.start(2) :]
    return readme


def _months(months: list[int]) -> str:
    """``[0, 1, 2, 3]`` as ``0–3``; a single Month as itself."""
    if len(months) > 1 and months == list(range(months[0], months[-1] + 1)):
        return f"{months[0]}–{months[-1]}"
    return ", ".join(str(month) for month in months)


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _points(a: float, b: float) -> str:
    """The difference between two shares as shown, so the text agrees with the table."""
    return f"{(round(a, 3) - round(b, 3)) * 100:+.1f} points"


def _metrics(metadata: Metadata, model: str, split: str) -> Metadata:
    return _MODELS[model](metadata)["metrics"][split]


def _fraud_rate(metadata: Metadata, split: str) -> float:
    metrics = metadata["metrics"][split]
    return metrics["fraudulent_applications"] / metrics["applications"]


def _table(metadata: Metadata, split: str) -> str:
    rows = [
        "| Model | Recall@5%FPR | Recall at the Threshold | FPR at the Threshold | PR-AUC "
        "| FPR Ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in _MODELS:
        metrics = _metrics(metadata, model, split)
        rows.append(
            f"| {model} | {_pct(metrics['recall_at_5pct_fpr'])} "
            f"| {_pct(metrics['recall_at_threshold'])} | {_pct(metrics['fpr_at_threshold'])} "
            f"| {metrics['pr_auc']:.3f} | {metrics['fpr_ratio']:.2f} |"
        )
    return "\n".join(rows) + "\n"


def _split_size(metadata: Metadata, split: str) -> str:
    metrics = metadata["metrics"][split]
    return (
        f"{metrics['applications']:,} Applications, {metrics['fraudulent_applications']:,} "
        f"of them Fraudulent ({_pct(_fraud_rate(metadata, split))})"
    )


def _about(metadata: Metadata) -> str:
    return (
        f"Model Version `{metadata['model_version']}`, Threshold {metadata['threshold']:.4f}."
        f"\n\n**Try it:** open the live [`/docs`]({SERVICE_URL}/docs), pick `POST /predict` "
        "→ *Try it out* → *Execute*. Or from a terminal:\n"
    )


def _curl(metadata: Metadata) -> str:
    reference = next(
        r for r in metadata["reference_applications"] if r["application"] == EXAMPLE_APPLICATION
    )
    response = {
        "fraud_score": reference["fraud_score"],
        "decision": "review" if reference["fraud_score"] >= metadata["threshold"] else "pass",
        "model_version": metadata["model_version"],
        "request_id": "…",
    }
    return (
        "```bash\n"
        f"curl -s {SERVICE_URL}/predict \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        f"  -d '{json.dumps(EXAMPLE_APPLICATION)}'\n"
        "```\n\n"
        "```json\n" + json.dumps(response, ensure_ascii=False) + "\n```\n"
    )


def _split(metadata: Metadata) -> str:
    training = metadata["training"]
    optuna = metadata["optuna"]
    return (
        "| Months | Role |\n"
        "|---|---|\n"
        f"| {_months(training['training_months'])} | Training Months: the final model is "
        "fitted here. |\n"
        f"| {optuna['tuning_month']} | Tuning Month: Optuna compares configurations trained "
        f"on Months {_months(optuna['trial_fit_months'])} here. |\n"
        f"| {training['validation_month']} | Validation Month: used only to choose the "
        "Threshold. |\n"
        f"| {_months(training['test_months'])} | Test Months: scored once, after every "
        "choice was fixed. |\n"
    )


def _test_results(metadata: Metadata) -> str:
    return (
        f"Test Months {_months(metadata['training']['test_months'])}: "
        f"{_split_size(metadata, 'test')}. Each model uses its own Threshold, chosen on "
        f"the Validation Month.\n\n" + _table(metadata, "test")
    )


def _validation_results(metadata: Metadata) -> str:
    return (
        f"Validation Month {metadata['training']['validation_month']}: "
        f"{_split_size(metadata, 'validation')}. Optimistic: each Threshold was chosen on "
        "this Month, so FPR at the Threshold is at most 5% by construction (as close as "
        "ties in the scores allow).\n\n" + _table(metadata, "validation")
    )


def _tuning(metadata: Metadata) -> str:
    optuna = metadata["optuna"]
    default = optuna["tuning_month_default_score"]
    tuned = optuna["tuning_month_tuned_score"]
    importances = sorted(optuna["param_importances"].items(), key=lambda kv: -kv[1])
    top = ", ".join(f"`{name}` {share:.0%}" for name, share in importances[:3])
    return (
        f"{optuna['n_trials']} TPE trials, each trained on Months "
        f"{_months(optuna['trial_fit_months'])} and scored on the Tuning Month "
        f"{optuna['tuning_month']}. Recall@5%FPR on the Tuning Month: {_pct(default)} with "
        f"LightGBM defaults, {_pct(tuned)} tuned ({_points(tuned, default)}). LightGBM "
        "defaults were scored on the Tuning Month only, so this is the one like-for-like "
        f"comparison, and the tuned figure is the best of {optuna['n_trials']} trials on "
        "that same Month, so part of the gain is selection. The final model uses the best "
        f"trial's {optuna['best_num_boost_round']} trees. By fANOVA importance, the "
        f"parameters that mattered most were {top}.\n"
    )


def _discussion(metadata: Metadata) -> str:
    lgbm = _metrics(metadata, "LightGBM (served)", "test")
    logreg = _metrics(metadata, "Logistic regression", "test")
    rule = _metrics(metadata, "`credit_risk_score` rule", "test")
    rates = [_fraud_rate(metadata, split) for split in ("training", "validation", "test")]
    trend = "rises" if rates[2] > rates[0] else "falls"
    ratio = lgbm["fpr_ratio"]
    how_often = f"about {1 / ratio:.1f} times as often" if ratio > 0 else "far more often"
    return (
        "- **LightGBM against logistic regression.** On the Test Months, at 5% FPR, "
        f"LightGBM flags {_pct(lgbm['recall_at_5pct_fpr'])} of Fraudulent Applications and "
        f"logistic regression {_pct(logreg['recall_at_5pct_fpr'])} "
        f"({_points(lgbm['recall_at_5pct_fpr'], logreg['recall_at_5pct_fpr'])}); PR-AUC "
        f"is {lgbm['pr_auc']:.3f} against {logreg['pr_auc']:.3f}. A well-preprocessed "
        "linear model gets most of the way; the single-feature rule flags "
        f"{_pct(rule['recall_at_5pct_fpr'])}.\n"
        "- **FPR Ratio between Age Groups.** At its Threshold, LightGBM's FPR Ratio on the "
        f"Test Months is {ratio:.2f}: Legitimate Applications from one Age Group are "
        f"flagged {how_often} as from the other. Fraud rises with age (EDA section 6), and "
        f"the model learns it. The BAF authors report {BAF_REFERENCE_FPR_RATIO} for their "
        "best models; this one is published rather than hidden "
        "([ADR 0003](docs/adr/0003-keep-age-as-feature-and-report-fpr-ratio.md)).\n"
        f"- **Drift between Months.** The share of Fraudulent Applications {trend} from "
        f"{_pct(rates[0])} in the Training Months and {_pct(rates[1])} in the Validation "
        f"Month to {_pct(rates[2])} in the Test Months. The Threshold chosen for 5% FPR on "
        f"the Validation Month flags {_pct(lgbm['fpr_at_threshold'])} of Legitimate "
        f"Applications on the Test Months, and {_pct(lgbm['recall_at_threshold'])} of "
        "Fraudulent ones. A real deployment would re-choose the Threshold on recent data.\n"
    )


def _without_age(metadata: Metadata) -> str:
    with_age = _metrics(metadata, "LightGBM (served)", "test")
    without = _metrics(metadata, "LightGBM without `customer_age` (experiment)", "test")
    recall = _points(without["recall_at_5pct_fpr"], with_age["recall_at_5pct_fpr"])
    moved = "raises" if without["fpr_ratio"] > with_age["fpr_ratio"] else "lowers"
    ratio = without["fpr_ratio"]
    remaining = (
        f"one Age Group's Legitimate Applications are still flagged about {1 / ratio:.1f} "
        "times as often as the other's"
        if 0 < ratio < 1
        else "the Age Groups are flagged equally often"
        if ratio >= 1
        else "one Age Group's Legitimate Applications are still flagged far more often"
    )
    return (
        "Measured: the same LightGBM, with the served model's parameters and tree count but "
        "without `customer_age`, is the last row of the tables above. On the Test Months, "
        f"dropping age changes Recall@5%FPR by {recall} "
        f"({_pct(with_age['recall_at_5pct_fpr'])} → {_pct(without['recall_at_5pct_fpr'])}) "
        f"and PR-AUC from {with_age['pr_auc']:.3f} to {without['pr_auc']:.3f}, and it "
        f"{moved} the FPR Ratio from {with_age['fpr_ratio']:.2f} to {ratio:.2f}: "
        f"{remaining}. The served model keeps age; whether to drop it is a decision for "
        "[ADR 0003](docs/adr/0003-keep-age-as-feature-and-report-fpr-ratio.md).\n"
    )


def _limits(metadata: Metadata) -> str:
    return (
        f"- **Rate limit.** `/predict` allows {protection.RATE_LIMIT_PER_MINUTE} requests a "
        "minute per client, with a 429 and `Retry-After` past that; request bodies over "
        f"{protection.MAX_BODY_BYTES:,} bytes get a 413 before they are parsed. The client "
        "is the last `X-Forwarded-For` hop, the address Cloud Run saw; earlier hops are "
        "client-supplied and ignored, so a forged header doesn't escape the limit. The "
        "limit is per address: it slows one client, not a distributed attack, and an IPv6 "
        "client can rotate addresses within its /64. It is kept in memory, which is enough "
        "for one instance.\n"
    )


_PARTS: dict[str, Callable[[Metadata], str]] = {
    "about": _about,
    "curl": _curl,
    "split": _split,
    "test-results": _test_results,
    "validation-results": _validation_results,
    "tuning": _tuning,
    "discussion": _discussion,
    "without-age": _without_age,
    "limits": _limits,
}
