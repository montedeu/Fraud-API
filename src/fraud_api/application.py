"""What an Application is, and the two ways in: dataset rows and API payloads."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import AfterValidator, BaseModel, ConfigDict, Field

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

PROPOSED_CREDIT_LIMITS = (190, 200, 210, 490, 500, 510, 990, 1000, 1500, 1900, 2000, 2100)
"""The only values ``proposed_credit_limit`` takes in the dataset."""

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


# ── API side ──────────────────────────────────────────────────────────────

EXAMPLE_APPLICATION: dict[str, Any] = {
    "income": 0.6,
    "name_email_similarity": 0.5,
    "prev_address_months_count": 20,
    "current_address_months_count": 60,
    "customer_age": 30,
    "days_since_request": 0.02,
    "intended_balcon_amount": 10.5,
    "payment_type": "AA",
    "zip_count_4w": 1500,
    "velocity_6h": 5000.0,
    "velocity_24h": 4800.0,
    "velocity_4w": 5000.0,
    "bank_branch_count_8w": 10,
    "date_of_birth_distinct_emails_4w": 8,
    "employment_status": "CA",
    "credit_risk_score": 120,
    "email_is_free": 1,
    "housing_status": "BA",
    "phone_home_valid": 0,
    "phone_mobile_valid": 1,
    "bank_months_count": 10,
    "has_other_cards": 0,
    "proposed_credit_limit": 500.0,
    "foreign_request": 0,
    "source": "INTERNET",
    "session_length_in_minutes": 5.0,
    "device_os": "linux",
    "keep_alive_session": 1,
    "device_distinct_emails_8w": 1,
}
"""A valid Application, shown in ``/docs``."""


def _one_of(allowed: Sequence[float], name: str):
    def check(value: float) -> float:
        for candidate in allowed:
            if abs(value - candidate) < 1e-9:
                return float(candidate)
        raise ValueError(f"{name} must be one of {list(allowed)}")

    return AfterValidator(check)


Number = Annotated[float, Field(allow_inf_nan=False)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
MaybeMissing = NonNegative | None
# Not Literal[0, 1]: that accepts JSON true and false even in strict mode.
Flag = Annotated[int, Field(strict=True, ge=0, le=1)]


class ApplicationIn(BaseModel):
    """One Application as an API client submits it. Every field is required; ``null``
    marks a missing value where one is allowed, and unknown fields are rejected."""

    model_config = ConfigDict(
        strict=True, extra="forbid", json_schema_extra={"examples": [EXAMPLE_APPLICATION]}
    )

    income: Annotated[float, _one_of([i / 10 for i in range(1, 10)], "income")]
    name_email_similarity: NonNegative
    prev_address_months_count: MaybeMissing
    current_address_months_count: MaybeMissing
    customer_age: Literal[10, 20, 30, 40, 50, 60, 70, 80, 90]
    days_since_request: NonNegative
    intended_balcon_amount: MaybeMissing
    payment_type: Literal[*CATEGORIES["payment_type"]]
    zip_count_4w: NonNegative
    velocity_6h: Number
    velocity_24h: NonNegative
    velocity_4w: NonNegative
    bank_branch_count_8w: NonNegative
    date_of_birth_distinct_emails_4w: NonNegative
    employment_status: Literal[*CATEGORIES["employment_status"]]
    credit_risk_score: Number
    email_is_free: Flag
    housing_status: Literal[*CATEGORIES["housing_status"]]
    phone_home_valid: Flag
    phone_mobile_valid: Flag
    bank_months_count: MaybeMissing
    has_other_cards: Flag
    proposed_credit_limit: Annotated[
        float, _one_of(PROPOSED_CREDIT_LIMITS, "proposed_credit_limit")
    ]
    foreign_request: Flag
    source: Literal[*CATEGORIES["source"]]
    session_length_in_minutes: MaybeMissing
    device_os: Literal[*CATEGORIES["device_os"]]
    keep_alive_session: Flag
    device_distinct_emails_8w: MaybeMissing


def to_frame(applications: Sequence[ApplicationIn]) -> pd.DataFrame:
    """Applications from the API, in the same shape as Applications from the dataset."""
    return pd.DataFrame([a.model_dump() for a in applications], columns=FIELDS)
