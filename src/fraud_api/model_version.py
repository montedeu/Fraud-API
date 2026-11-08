"""A Model Version on disk: the native LightGBM model file and its metadata."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from fraud_api.features import prepare

MODEL_FILE = "model.txt"
METADATA_FILE = "metadata.json"
MODEL_DIR = Path("model")
SCORING_FIELDS = ("threshold", "features", "categories")
"""Metadata that changes what the model decides, so it is part of the Model Version id."""
TOP_FACTORS = 3
EXPECTED_VALUE = "expected_value"


@dataclass(frozen=True)
class ModelVersion:
    booster: lgb.Booster
    metadata: dict[str, Any]

    @property
    def id(self) -> str:
        return self.metadata["model_version"]

    @property
    def threshold(self) -> float:
        return self.metadata["threshold"]

    def score(self, applications: pd.DataFrame, *, num_threads: int = 1) -> np.ndarray:
        """Fraud Scores for ``applications``, prepared the same way as in training.

        One thread by default: the service scores one Application per request, and an
        OpenMP team per request would only oversubscribe the CPU.
        """
        features = prepare(applications, self.metadata["features"], self.metadata["categories"])
        return self.booster.predict(features, num_threads=num_threads)

    def contributions(self, applications: pd.DataFrame, *, num_threads: int = 1) -> pd.DataFrame:
        """See :func:`contributions`."""
        return contributions(
            self.booster,
            applications,
            self.metadata["features"],
            self.metadata["categories"],
            num_threads=num_threads,
        )


def contributions(
    booster: lgb.Booster,
    applications: pd.DataFrame,
    features: Sequence[str],
    categories: Mapping[str, Sequence[str]],
    *,
    num_threads: int = 1,
) -> pd.DataFrame:
    """How much each field moved each Application's raw score, LightGBM's TreeSHAP values.

    One column per feature, plus ``expected_value``; each row sums to the raw score, in
    log-odds, whose logistic transform is the Fraud Score. Positive pushes towards Review.
    """
    prepared = prepare(applications, features, categories)
    values = booster.predict(prepared, pred_contrib=True, num_threads=num_threads)
    return pd.DataFrame(
        values, columns=[*prepared.columns, EXPECTED_VALUE], index=applications.index
    )


def top_factors(
    contributions_row: pd.Series, application: Mapping[str, Any], n: int = TOP_FACTORS
) -> list[dict[str, Any]]:
    """The ``n`` fields with the largest absolute contribution, largest first, each with
    the Applicant's value (``None`` if missing) and its signed contribution in log-odds."""
    fields = contributions_row.drop(EXPECTED_VALUE)
    # Stable, so ties keep the feature order and the result never depends on the platform.
    order = np.argsort(-fields.abs().to_numpy(), kind="stable")[:n]
    return [
        {
            "field": fields.index[i],
            "value": application[fields.index[i]],
            "contribution": float(fields.iloc[i]),
        }
        for i in order
    ]


def model_version_id(model_text: str, metadata: dict[str, Any]) -> str:
    """Derived from the model file and its Threshold, features and categories, so the same
    model deciding the same way always gets the same id, and any change gets a new one."""
    scoring = json.dumps({field: metadata[field] for field in SCORING_FIELDS}, sort_keys=True)
    digest = hashlib.sha256(model_text.encode())
    digest.update(b"\0" + scoring.encode())
    return "lgbm-" + digest.hexdigest()[:12]


def write(directory: Path, booster: lgb.Booster, metadata: dict[str, Any]) -> ModelVersion:
    """Write the model file and its metadata, with the Model Version id filled in."""
    model_text = booster.model_to_string()
    metadata = {"model_version": model_version_id(model_text, metadata), **metadata}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MODEL_FILE).write_text(model_text)
    (directory / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n")
    return load(directory)


def load(directory: Path = MODEL_DIR) -> ModelVersion:
    model_text = (directory / MODEL_FILE).read_text()
    metadata = json.loads((directory / METADATA_FILE).read_text())
    if metadata["model_version"] != model_version_id(model_text, metadata):
        raise ValueError(f"{directory} does not match its Model Version id")
    return ModelVersion(lgb.Booster(model_str=model_text), metadata)
