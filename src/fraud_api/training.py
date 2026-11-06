"""The training pipeline: from Base.csv to a committed Model Version (ADR 0001, 0002)."""

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from fraud_api import evaluation
from fraud_api.application import CATEGORIES, FIELDS, Dataset, read_dataset
from fraud_api.baselines import Scorer, credit_risk_score_rule, fit_logistic_regression
from fraud_api.data import DATA_DIR, FILE_NAME, SHA256, verify
from fraud_api.features import prepare
from fraud_api.model_version import MODEL_DIR, ModelVersion, write

TRAINING_MONTHS = (0, 1, 2, 3, 4)
VALIDATION_MONTH = 5
TEST_MONTHS = (6, 7)
SEED = 42
NUM_BOOST_ROUND = 100  # LightGBM's default


def train(
    data_path: Path = DATA_DIR / FILE_NAME,
    output_dir: Path = MODEL_DIR,
    *,
    expected_sha256: str = SHA256,
    seed: int = SEED,
) -> ModelVersion:
    """Train LightGBM on the Training Months, choose the Threshold on the Validation Month,
    score the Test Months once, and write the Model Version to ``output_dir``.

    The two baselines go through the same protocol, each with its own Threshold, and are
    reported next to LightGBM in the metadata. Only LightGBM is written as a model.
    """
    verify(data_path, expected_sha256)
    splits = split_by_month(read_dataset(data_path))

    params = _lightgbm_params(seed)
    training = splits["training"]
    booster = lgb.train(
        params,
        lgb.Dataset(prepare(training.applications, FIELDS, CATEGORIES), label=training.labels),
        num_boost_round=NUM_BOOST_ROUND,
    )

    models: dict[str, Scorer] = {
        "lightgbm": lambda applications: _score(booster, applications),
        "logistic_regression": fit_logistic_regression(training.applications, training.labels),
        "credit_risk_score_rule": credit_risk_score_rule,
    }
    comparison = {name: _evaluate(scorer, splits) for name, scorer in models.items()}
    threshold = comparison["lightgbm"]["threshold"]

    references = reference_applications()
    reference_scores = _score(booster, pd.DataFrame(references))

    metadata = {
        "threshold": threshold,
        "features": FIELDS,
        "categories": CATEGORIES,
        "metrics": comparison["lightgbm"]["metrics"],
        "comparison": comparison,
        "data_sha256": expected_sha256,
        "training": {
            "training_months": list(TRAINING_MONTHS),
            "validation_month": VALIDATION_MONTH,
            "test_months": list(TEST_MONTHS),
            "lightgbm_params": params,
            "num_boost_round": NUM_BOOST_ROUND,
            "lightgbm_version": lgb.__version__,
        },
        "reference_applications": [
            {"application": application, "fraud_score": float(score)}
            for application, score in zip(references, reference_scores, strict=True)
        ],
    }
    return write(output_dir, booster, metadata)


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


def _lightgbm_params(seed: int) -> dict[str, Any]:
    return {
        "objective": "binary",
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
    }


def _score(booster: lgb.Booster, applications: pd.DataFrame) -> np.ndarray:
    return booster.predict(prepare(applications, FIELDS, CATEGORIES))


def _evaluate(scorer: Scorer, splits: dict[str, Dataset]) -> dict[str, Any]:
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
