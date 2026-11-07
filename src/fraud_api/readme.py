"""The README's numbers, generated from the committed Model Version metadata.

Each generated part of ``README.md`` sits between ``<!-- generated:NAME -->`` and
``<!-- /generated:NAME -->``. ``fraud-api readme`` rewrites those parts; a test fails
when they no longer match ``model/metadata.json``, so the README cannot drift from the
model it describes. Text outside the markers is written by hand and holds no results.
"""

import json
import re
from collections.abc import Callable
from typing import Any

from fraud_api.application import EXAMPLE_APPLICATION

README = "README.md"
SERVICE_URL = "https://fraud-api-r6bn3ptcuq-ew.a.run.app"

Metadata = dict[str, Any]

_MODELS = {
    "LightGBM (served)": lambda m: m,
    "Logistic regression": lambda m: m["baselines"]["logistic_regression"],
    "`credit_risk_score` rule": lambda m: m["baselines"]["credit_risk_score_rule"],
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
    if months == list(range(months[0], months[-1] + 1)) and len(months) > 1:
        return f"{months[0]}–{months[-1]}"
    return ", ".join(str(month) for month in months)


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _table(metadata: Metadata, split: str) -> str:
    rows = [
        "| Model | Recall@5%FPR | Recall at the Threshold | FPR at the Threshold | PR-AUC "
        "| FPR Ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in _MODELS.items():
        m = result(metadata)["metrics"][split]
        rows.append(
            f"| {name} | {_pct(m['recall_at_5pct_fpr'])} | {_pct(m['recall_at_threshold'])} "
            f"| {_pct(m['fpr_at_threshold'])} | {m['pr_auc']:.3f} | {m['fpr_ratio']:.2f} |"
        )
    return "\n".join(rows) + "\n"


def _split_size(metadata: Metadata, split: str) -> str:
    m = metadata["metrics"][split]
    return (
        f"{m['applications']:,} Applications, {m['fraudulent_applications']:,} of them "
        f"Fraudulent ({_pct(m['fraudulent_applications'] / m['applications'])})"
    )


def _test_results(metadata: Metadata) -> str:
    return (
        f"Test Months {_months(metadata['training']['test_months'])}: "
        f"{_split_size(metadata, 'test')}. Each model uses its own Threshold, chosen on "
        f"the Validation Month.\n\n" + _table(metadata, "test")
    )


def _validation_results(metadata: Metadata) -> str:
    month = metadata["training"]["validation_month"]
    return (
        f"Validation Month {month}: {_split_size(metadata, 'validation')}. Optimistic: "
        "each Threshold was chosen on this Month, so FPR at the Threshold is 5% by "
        "construction.\n\n" + _table(metadata, "validation")
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
        f"{optuna['tuning_month']}. Recall@5%FPR on the Tuning Month went "
        f"from {_pct(default)} with LightGBM defaults to {_pct(tuned)} tuned "
        f"(+{(tuned - default) * 100:.1f} points). The tuned figure is the best of "
        f"{optuna['n_trials']} trials on that Month, so part of the gain is selection; "
        f"the Validation Month and Test Months above are the honest measure. The final "
        f"model uses the best trial's {optuna['best_num_boost_round']} trees. Most of the "
        f"variance between trials came from {top}.\n"
    )


def _discussion(metadata: Metadata) -> str:
    lgbm = metadata["metrics"]["test"]
    logreg = metadata["baselines"]["logistic_regression"]["metrics"]["test"]
    rule = metadata["baselines"]["credit_risk_score_rule"]["metrics"]["test"]
    rate = {
        split: metadata["metrics"][split]["fraudulent_applications"]
        / metadata["metrics"][split]["applications"]
        for split in ("training", "validation", "test")
    }
    return (
        f"- **LightGBM against logistic regression.** On the Test Months LightGBM flags "
        f"{_pct(lgbm['recall_at_5pct_fpr'])} of Fraudulent Applications at 5% FPR, "
        f"logistic regression {_pct(logreg['recall_at_5pct_fpr'])}: "
        f"{(lgbm['recall_at_5pct_fpr'] - logreg['recall_at_5pct_fpr']) * 100:.1f} points. "
        f"PR-AUC is {lgbm['pr_auc']:.3f} against {logreg['pr_auc']:.3f}. The gap is real "
        f"but modest: a well-preprocessed linear model gets most of the way, and the "
        f"single-feature rule gets {_pct(rule['recall_at_5pct_fpr'])}.\n"
        f"- **Age disparity.** At its Threshold, LightGBM's FPR Ratio on the Test Months "
        f"is {lgbm['fpr_ratio']:.2f}: Legitimate Applications from one Age Group are "
        f"flagged about {1 / lgbm['fpr_ratio']:.1f} times as often as from the other "
        f"(fraud rises with age, and the model learns it: EDA section 6). That is in "
        f"line with the BAF authors' best models, around 0.3, and is "
        f"published rather than hidden "
        f"([ADR 0003](docs/adr/0003-keep-age-as-feature-and-report-fpr-ratio.md)).\n"
        f"- **Drift between Months.** The share of Fraudulent Applications rises from "
        f"{_pct(rate['training'])} in the Training Months and {_pct(rate['validation'])} "
        f"in the Validation Month to {_pct(rate['test'])} in the Test Months. The "
        f"Threshold chosen for 5% FPR on the Validation Month flags "
        f"{_pct(lgbm['fpr_at_threshold'])} of Legitimate Applications on the Test Months, "
        f"and catches {_pct(lgbm['recall_at_threshold'])} of Fraudulent ones: the "
        f"operating point moves, and a real deployment would re-choose the Threshold on "
        f"recent data.\n"
    )


def _about(metadata: Metadata) -> str:
    return f"Model Version `{metadata['model_version']}`, Threshold {metadata['threshold']:.4f}.\n"


def _curl(metadata: Metadata) -> str:
    reference = next(
        r for r in metadata["reference_applications"] if r["application"] == EXAMPLE_APPLICATION
    )
    decision = "review" if reference["fraud_score"] >= metadata["threshold"] else "pass"
    response = {
        "fraud_score": reference["fraud_score"],
        "decision": decision,
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


_PARTS: dict[str, Callable[[Metadata], str]] = {
    "about": _about,
    "curl": _curl,
    "test-results": _test_results,
    "validation-results": _validation_results,
    "tuning": _tuning,
    "discussion": _discussion,
}
