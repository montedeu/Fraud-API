"""A Model Version on disk: the native LightGBM model file and its metadata."""

import hashlib
import json
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

    def score(self, applications: pd.DataFrame) -> np.ndarray:
        """Fraud Scores for ``applications``, prepared the same way as in training."""
        features = prepare(applications, self.metadata["categories"])
        return self.booster.predict(features[self.metadata["features"]])


def model_version_id(model_text: str) -> str:
    """Derived from the model file's content, so the same model always gets the same id."""
    return "lgbm-" + hashlib.sha256(model_text.encode()).hexdigest()[:12]


def write(directory: Path, booster: lgb.Booster, metadata: dict[str, Any]) -> ModelVersion:
    """Write the model file and its metadata, with the Model Version id filled in."""
    model_text = booster.model_to_string()
    metadata = {"model_version": model_version_id(model_text), **metadata}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MODEL_FILE).write_text(model_text)
    (directory / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n")
    return load(directory)


def load(directory: Path = MODEL_DIR) -> ModelVersion:
    model_text = (directory / MODEL_FILE).read_text()
    metadata = json.loads((directory / METADATA_FILE).read_text())
    if metadata["model_version"] != model_version_id(model_text):
        raise ValueError(f"{directory / MODEL_FILE} does not match its metadata's Model Version")
    return ModelVersion(lgb.Booster(model_str=model_text), metadata)
