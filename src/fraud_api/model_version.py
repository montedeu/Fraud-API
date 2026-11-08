"""A Model Version on disk: the native LightGBM model file and its metadata."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import lightgbm as lgb
import numpy as np
import pandas as pd

from fraud_api.features import prepare

MODEL_FILE = "model.txt"
METADATA_FILE = "metadata.json"
MODEL_DIR = Path("model")
SCORING_FIELDS = ("threshold", "features", "categories")
"""Metadata that changes what the model decides, so it is part of the Model Version id."""
TOP_FACTOR_COUNT = 3
"""How many top factors each scored Application gets."""
EXPECTED_VALUE = "expected_value"
"""The column of :meth:`ModelVersion.contributions` that holds the model's baseline."""


class TopFactor(TypedDict):
    field: str
    value: Any
    """The Applicant's value, ``None`` if missing."""
    contribution: float
    """How far this field moved the raw score, in log-odds; positive pushes towards Review."""


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
        return self.booster.predict(self._prepare(applications), num_threads=num_threads)

    def contributions(self, applications: pd.DataFrame) -> pd.DataFrame:
        """How much each field moved each Application's raw score: LightGBM's own TreeSHAP
        values (``pred_contrib``).

        One column per feature plus ``expected_value``. Each row sums to the raw score, in
        log-odds, and the Fraud Score is its logistic transform.
        """
        prepared = self._prepare(applications)
        values = self.booster.predict(prepared, pred_contrib=True, num_threads=1)
        return pd.DataFrame(
            values, columns=[*prepared.columns, EXPECTED_VALUE], index=applications.index
        )

    def top_factors(self, applications: Sequence[Mapping[str, Any]]) -> list[list[TopFactor]]:
        """For each Application, the fields with the largest absolute contribution, largest
        first, each with the value as given in ``applications``."""
        frame = pd.DataFrame(list(applications), columns=self.metadata["features"])
        contributions = self.contributions(frame).drop(columns=EXPECTED_VALUE)
        result = []
        for application, (_, row) in zip(applications, contributions.iterrows(), strict=True):
            # Stable, so ties keep the feature order whatever the platform.
            order = np.argsort(-row.abs().to_numpy(), kind="stable")[:TOP_FACTOR_COUNT]
            result.append(
                [
                    TopFactor(
                        field=row.index[i],
                        value=application[row.index[i]],
                        contribution=float(row.iloc[i]),
                    )
                    for i in order
                ]
            )
        return result

    def _prepare(self, applications: pd.DataFrame) -> pd.DataFrame:
        return prepare(applications, self.metadata["features"], self.metadata["categories"])


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
    booster = lgb.Booster(model_str=model_text)
    # Contributions are labelled by the metadata's feature order, so it must be the model's.
    if booster.feature_name() != metadata["features"]:
        raise ValueError(f"{directory}: the model's features differ from its metadata")
    return ModelVersion(booster, metadata)
