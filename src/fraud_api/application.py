"""What an Application is, and how dataset rows become Applications."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES: dict[str, list[str]] = {
    "payment_type": ["AA", "AB", "AC", "AD", "AE"],
    "employment_status": ["CA", "CB", "CC", "CD", "CE", "CF", "CG"],
    "housing_status": ["BA", "BB", "BC", "BD", "BE", "BF", "BG"],
    "source": ["INTERNET", "TELEAPP"],
    "device_os": ["linux", "macintosh", "other", "windows", "x11"],
}
"""Every category field and its accepted values, in a fixed order."""

SENTINEL_FIELDS = (
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
)
"""Fields where the dataset writes -1 for a missing value."""

FIELDS = [
    "income",
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
    "customer_age",
    "days_since_request",
    "intended_balcon_amount",
    "payment_type",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "employment_status",
    "credit_risk_score",
    "email_is_free",
    "housing_status",
    "phone_home_valid",
    "phone_mobile_valid",
    "bank_months_count",
    "has_other_cards",
    "proposed_credit_limit",
    "foreign_request",
    "source",
    "session_length_in_minutes",
    "device_os",
    "keep_alive_session",
    "device_distinct_emails_8w",
]
"""The 29 fields of an Application, in dataset column order."""

LABEL = "fraud_bool"
MONTH = "month"
DROPPED = ("device_fraud_count",)
"""Dataset columns that are neither part of an Application nor needed: constant in Base.csv."""


@dataclass(frozen=True)
class Dataset:
    """Applications with their labels (1 = Fraudulent Application) and Months."""

    applications: pd.DataFrame
    labels: np.ndarray
    months: np.ndarray

    def __len__(self) -> int:
        return len(self.applications)

    def where(self, mask: np.ndarray) -> "Dataset":
        return Dataset(
            self.applications[mask].reset_index(drop=True), self.labels[mask], self.months[mask]
        )


def read_dataset(path: Path) -> Dataset:
    """Read a BAF-format CSV file into Applications."""
    return from_dataset(pd.read_csv(path))


def from_dataset(rows: pd.DataFrame) -> Dataset:
    """Turn BAF dataset rows into Applications, separating the label and the Month."""
    expected = {*FIELDS, LABEL, MONTH, *DROPPED}
    if missing := expected - set(rows.columns):
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    if extra := set(rows.columns) - expected:
        raise ValueError(f"dataset has unexpected columns: {sorted(extra)}")

    applications = rows[FIELDS].copy()
    for field in SENTINEL_FIELDS:
        applications[field] = applications[field].where(applications[field] != -1)
    # Negative amounts are the -1 sentinel smeared by the dataset's privacy noise.
    balcon = applications["intended_balcon_amount"]
    applications["intended_balcon_amount"] = balcon.where(balcon >= 0)
    return Dataset(
        applications.reset_index(drop=True),
        rows[LABEL].to_numpy(dtype=int),
        rows[MONTH].to_numpy(dtype=int),
    )
