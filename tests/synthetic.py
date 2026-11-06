"""A synthetic dataset in Base.csv format, so tests never need the real data."""

from pathlib import Path

import numpy as np
import pandas as pd

from fraud_api.application import CATEGORIES, FIELDS, SENTINEL_FIELDS

MONTHS = 8
PROPOSED_CREDIT_LIMITS = [190, 200, 210, 490, 500, 510, 990, 1000, 1500, 1900, 2000, 2100]


def make_dataset(rows_per_month: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Eight Months with the real column set, -1 sentinels, negative
    ``intended_balcon_amount`` and a planted fraud signal."""
    rng = np.random.default_rng(seed)
    n = rows_per_month * MONTHS
    rows = pd.DataFrame(
        {
            "income": rng.integers(1, 10, n) / 10,
            "name_email_similarity": rng.random(n),
            "prev_address_months_count": rng.integers(0, 300, n),
            "current_address_months_count": rng.integers(0, 400, n),
            "customer_age": rng.integers(1, 10, n) * 10,
            "days_since_request": rng.exponential(1.0, n),
            "intended_balcon_amount": rng.normal(5, 20, n),
            "zip_count_4w": rng.integers(1, 6000, n),
            "velocity_6h": rng.normal(5000, 3000, n),
            "velocity_24h": rng.normal(4800, 1500, n),
            "velocity_4w": rng.normal(4900, 900, n),
            "bank_branch_count_8w": rng.integers(0, 2000, n),
            "date_of_birth_distinct_emails_4w": rng.integers(0, 40, n),
            "credit_risk_score": rng.integers(-170, 390, n),
            "email_is_free": rng.integers(0, 2, n),
            "phone_home_valid": rng.integers(0, 2, n),
            "phone_mobile_valid": rng.integers(0, 2, n),
            "bank_months_count": rng.integers(0, 33, n),
            "has_other_cards": rng.integers(0, 2, n),
            "proposed_credit_limit": rng.choice(PROPOSED_CREDIT_LIMITS, n).astype(float),
            "foreign_request": rng.integers(0, 2, n),
            "session_length_in_minutes": rng.exponential(8.0, n),
            "keep_alive_session": rng.integers(0, 2, n),
            "device_distinct_emails_8w": rng.integers(0, 3, n),
            **{field: rng.choice(values, n) for field, values in CATEGORIES.items()},
        }
    )
    for field in SENTINEL_FIELDS:
        rows.loc[rng.random(n) < 0.2, field] = -1

    risk = (
        -4.5
        + 0.008 * rows["credit_risk_score"]
        + 1.2 * (rows["prev_address_months_count"] == -1)
        + 0.8 * (rows["customer_age"] >= 50)
        + 1.0 * (rows["housing_status"] == "BA")
    )
    rows["fraud_bool"] = (rng.random(n) < 1 / (1 + np.exp(-risk))).astype(int)
    rows["device_fraud_count"] = 0
    rows["month"] = np.repeat(np.arange(MONTHS), rows_per_month)
    return rows[["fraud_bool", *FIELDS, "device_fraud_count", "month"]]


def write_dataset(path: Path, rows: pd.DataFrame) -> Path:
    rows.to_csv(path, index=False)
    return path
