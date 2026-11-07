"""The training pipeline: from Base.csv to a committed Model Version (ADR 0001, 0002)."""

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import sklearn

from fraud_api import baselines, evaluation, reports, tuning
from fraud_api.application import CATEGORIES, FIELDS, Dataset, read_dataset
from fraud_api.data import DATA_DIR, FILE_NAME, SHA256, verify
from fraud_api.features import prepare
from fraud_api.model_version import MODEL_DIR, ModelVersion, write

TRAINING_MONTHS = (0, 1, 2, 3, 4)
VALIDATION_MONTH = 5
TEST_MONTHS = (6, 7)
SEED = 42


def train(
    data_path: Path = DATA_DIR / FILE_NAME,
    output_dir: Path = MODEL_DIR,
    *,
    reports_dir: Path = reports.REPORTS_DIR,
    expected_sha256: str = SHA256,
    seed: int = SEED,
    n_trials: int = tuning.N_TRIALS,
) -> ModelVersion:
    """Tune LightGBM on the Tuning Month, train it on the Training Months with the best
    configuration, choose the Threshold on the Validation Month, score the Test Months
    once, and write the Model Version to ``output_dir`` and the tuning report to
    ``reports_dir``.

    The two baselines go through the same protocol, each with its own Threshold, and are
    reported under ``baselines`` in the metadata. Only LightGBM is written as a model.
    """
    if n_trials < tuning.MIN_TRIALS:
        raise ValueError(f"n_trials must be at least {tuning.MIN_TRIALS}, got {n_trials}")
    verify(data_path, expected_sha256)
    splits = split_by_month(read_dataset(data_path))

    training = splits["training"]
    # Only the Training Months reach the search; it holds out Month 4 itself.
    tuned = tuning.tune(training, n_trials=n_trials, seed=seed)
    # No early stopping here: the tree count is the best trial's, so neither the
    # Validation Month nor the Test Months steer the fit.
    booster = lgb.train(
        tuned.params,
        lgb.Dataset(prepare(training.applications, FIELDS, CATEGORIES), label=training.labels),
        num_boost_round=tuned.num_boost_round,
    )
    importances = tuning.param_importances(tuned.study, seed)

    served = _evaluate(lambda applications: _score(booster, applications), splits)
    threshold = served["threshold"]
    baseline_scorers: dict[str, baselines.Scorer] = {
        "logistic_regression": baselines.fit_logistic_regression(
            training.applications, training.labels
        ),
        "credit_risk_score_rule": baselines.credit_risk_score_rule,
    }

    references = reference_applications()
    reference_scores = _score(booster, pd.DataFrame(references))

    metadata = {
        "threshold": threshold,
        "features": FIELDS,
        "categories": CATEGORIES,
        "metrics": served["metrics"],
        "baselines": {
            name: {"settings": baselines.SETTINGS[name], **_evaluate(scorer, splits)}
            for name, scorer in baseline_scorers.items()
        },
        "data_sha256": expected_sha256,
        "optuna": {
            "n_trials": n_trials,
            "trial_fit_months": list(tuning.TRIAL_FIT_MONTHS),
            "tuning_month": tuning.TUNING_MONTH,
            "objective": "Recall@5%FPR on the Tuning Month",
            "best_params": tuned.study.best_params,
            "best_num_boost_round": tuned.num_boost_round,
            "tuning_month_default_score": tuned.default_score,
            "tuning_month_tuned_score": tuned.tuned_score,
            "note": "The tuned score is the best of n_trials on the Tuning Month and so "
            "optimistic; compare default and tuned on the Validation Month and Test Months.",
            "param_importances": importances,
        },
        "training": {
            "training_months": list(TRAINING_MONTHS),
            "validation_month": VALIDATION_MONTH,
            "test_months": list(TEST_MONTHS),
            "lightgbm_params": tuned.params,
            "num_boost_round": tuned.num_boost_round,
            "lightgbm_version": lgb.__version__,
            "sklearn_version": sklearn.__version__,
            "optuna_version": optuna.__version__,
        },
        "reference_applications": [
            {"application": application, "fraud_score": float(score)}
            for application, score in zip(references, reference_scores, strict=True)
        ],
    }
    model = write(output_dir, booster, metadata)
    # Written after the Model Version, so a failed run never leaves reports that
    # describe a model other than the committed one.
    reports_dir.mkdir(parents=True, exist_ok=True)
    tuning.trials_table(tuned.study).to_csv(reports_dir / reports.TRIALS_FILE, index=False)
    reports.plot_param_importances(importances, reports_dir / reports.IMPORTANCE_PLOT)
    return model


def split_by_month(dataset: Dataset) -> dict[str, Dataset]:
    """Training Months, Validation Month and Test Months, as in ADR 0001."""
    return {
        "training": dataset.where(np.isin(dataset.months, TRAINING_MONTHS)),
        "validation": dataset.where(dataset.months == VALIDATION_MONTH),
        "test": dataset.where(np.isin(dataset.months, TEST_MONTHS)),
    }


def reference_applications() -> list[dict[str, Any]]:
    """Eleven synthetic Applications for the training/serving skew test.

    Together they use every category value, a missing value in every field that may be
    missing, and negative ``credit_risk_score`` and ``velocity_6h``. The last one carries
    the strongest fraud signals in the data, so that it is sent to Review.
    """
    base: dict[str, Any] = {
        "income": 0.6,
        "name_email_similarity": 0.5,
        "prev_address_months_count": 20,
        "current_address_months_count": 60,
        "customer_age": 30,
        "days_since_request": 0.02,
        "intended_balcon_amount": 10.5,
        "zip_count_4w": 1500,
        "velocity_6h": 5000.0,
        "velocity_24h": 4800.0,
        "velocity_4w": 5000.0,
        "bank_branch_count_8w": 10,
        "date_of_birth_distinct_emails_4w": 8,
        "credit_risk_score": 120,
        "email_is_free": 1,
        "phone_home_valid": 0,
        "phone_mobile_valid": 1,
        "bank_months_count": 10,
        "has_other_cards": 0,
        "proposed_credit_limit": 500.0,
        "foreign_request": 0,
        "session_length_in_minutes": 5.0,
        "keep_alive_session": 1,
        "device_distinct_emails_8w": 1,
    }
    variations: list[dict[str, Any]] = [
        {},
        {field: None for field in ("prev_address_months_count", "bank_months_count")},
        {"current_address_months_count": None, "intended_balcon_amount": None},
        {"credit_risk_score": -80, "customer_age": 50, "income": 0.9},
        {"velocity_6h": -20.0, "session_length_in_minutes": None, "foreign_request": 1},
        {"device_distinct_emails_8w": None, "customer_age": 70, "proposed_credit_limit": 1500.0},
        {"customer_age": 20, "income": 0.1, "email_is_free": 0, "has_other_cards": 1},
        {"customer_age": 90, "credit_risk_score": 300, "proposed_credit_limit": 2100.0},
        {"keep_alive_session": 0, "phone_home_valid": 1, "phone_mobile_valid": 0},
        {field: None for field in ("prev_address_months_count", "intended_balcon_amount")}
        | {"customer_age": 10, "proposed_credit_limit": 190.0},
        {
            "housing_status": "BA",
            "device_os": "windows",
            "keep_alive_session": 0,
            "prev_address_months_count": None,
            "bank_months_count": None,
            "current_address_months_count": 10,
            "credit_risk_score": 250,
            "proposed_credit_limit": 2000.0,
            "customer_age": 60,
            "name_email_similarity": 0.05,
            "phone_home_valid": 0,
            "income": 0.9,
            "payment_type": "AC",
            "employment_status": "CA",
            "source": "INTERNET",
            "device_distinct_emails_8w": 2,
            "date_of_birth_distinct_emails_4w": 2,
        },
    ]
    applications = []
    for i, variation in enumerate(variations):
        categories = {field: values[i % len(values)] for field, values in CATEGORIES.items()}
        application = base | categories | variation
        applications.append({field: application[field] for field in FIELDS})
    return applications


def _score(booster: lgb.Booster, applications: pd.DataFrame) -> np.ndarray:
    return booster.predict(prepare(applications, FIELDS, CATEGORIES))


def _evaluate(scorer: baselines.Scorer, splits: dict[str, Dataset]) -> dict[str, Any]:
    """Threshold at 5% FPR on the Validation Month, then metrics on every split."""
    scores = {name: scorer(split.applications) for name, split in splits.items()}
    threshold = evaluation.threshold_for_fpr(splits["validation"].labels, scores["validation"])
    return {
        "threshold": threshold,
        "metrics": {
            name: _metrics(split, scores[name], threshold) for name, split in splits.items()
        },
    }


def _metrics(split: Dataset, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    labels = split.labels
    ages = split.applications["customer_age"].to_numpy()
    return {
        "applications": len(split),
        "fraudulent_applications": int(labels.sum()),
        "recall_at_5pct_fpr": evaluation.recall_at_fpr(labels, scores),
        "recall_at_threshold": evaluation.recall_at(labels, scores, threshold),
        "fpr_at_threshold": evaluation.fpr_at(labels, scores, threshold),
        "pr_auc": evaluation.pr_auc(labels, scores),
        "fpr_ratio": evaluation.fpr_ratio(labels, scores, ages, threshold),
    }
