import json
import shutil

import numpy as np
import pandas as pd
import pytest
from synthetic import make_dataset, write_dataset

from fraud_api import evaluation, model_version
from fraud_api.application import CATEGORIES, FIELDS, SENTINEL_FIELDS, read_dataset
from fraud_api.data import ChecksumMismatchError, sha256
from fraud_api.training import train

REQUIRED_METADATA = {
    "model_version",
    "threshold",
    "features",
    "categories",
    "metrics",
    "data_sha256",
    "reference_applications",
}
REQUIRED_METRICS = {"recall_at_5pct_fpr", "fpr_at_threshold", "pr_auc", "fpr_ratio"}


@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("data") / "Base.csv", make_dataset())


@pytest.fixture(scope="module")
def trained(dataset_path, tmp_path_factory):
    output = tmp_path_factory.mktemp("model")
    train(dataset_path, output, expected_sha256=sha256(dataset_path))
    return output


def test_model_version_files_exist_and_load(trained):
    assert (trained / model_version.MODEL_FILE).is_file()
    assert (trained / model_version.METADATA_FILE).is_file()

    loaded = model_version.load(trained)

    assert loaded.id.startswith("lgbm-")


def test_metadata_has_every_required_field(trained):
    metadata = json.loads((trained / model_version.METADATA_FILE).read_text())

    assert metadata.keys() >= REQUIRED_METADATA
    assert metadata["features"] == FIELDS
    assert metadata["categories"] == CATEGORIES
    for split in ("training", "validation", "test"):
        assert metadata["metrics"][split].keys() >= REQUIRED_METRICS


def test_reference_applications_cover_missing_values_categories_and_negative_scores(trained):
    references = [
        r["application"] for r in model_version.load(trained).metadata["reference_applications"]
    ]

    assert 8 <= len(references) <= 12
    for field in [*SENTINEL_FIELDS, "intended_balcon_amount"]:
        assert any(r[field] is None for r in references), field
    for field, values in CATEGORIES.items():
        assert {r[field] for r in references} == set(values), field
    assert any(r["credit_risk_score"] < 0 for r in references)


def test_reference_fraud_scores_match_the_loaded_model(trained):
    loaded = model_version.load(trained)
    references = loaded.metadata["reference_applications"]

    scores = loaded.score(pd.DataFrame([r["application"] for r in references]))

    assert scores == pytest.approx([r["fraud_score"] for r in references], rel=1e-12)


def test_threshold_gives_5_percent_fpr_on_the_validation_month(trained, dataset_path):
    loaded = model_version.load(trained)
    validation = _month(dataset_path, 5)

    fpr = evaluation.fpr_at(
        validation.labels, loaded.score(validation.applications), loaded.threshold
    )

    assert fpr == pytest.approx(0.05, abs=0.005)
    assert fpr <= 0.05


def test_label_fields_and_month_are_not_model_features(trained):
    features = model_version.load(trained).booster.feature_name()

    assert {"month", "fraud_bool", "device_fraud_count"}.isdisjoint(features)
    assert features == FIELDS


def test_perturbing_test_month_labels_changes_neither_threshold_nor_model(trained, tmp_path):
    rows = make_dataset()
    test_months = rows["month"].isin([6, 7])
    rows.loc[test_months, "fraud_bool"] = 1 - rows.loc[test_months, "fraud_bool"]
    perturbed_path = write_dataset(tmp_path / "Base.csv", rows)

    perturbed = train(perturbed_path, tmp_path / "model", expected_sha256=sha256(perturbed_path))
    original = model_version.load(trained)

    assert perturbed.threshold == original.threshold
    assert perturbed.id == original.id


def test_same_seed_reproduces_the_model_version_id(trained, dataset_path, tmp_path):
    again = train(dataset_path, tmp_path / "model", expected_sha256=sha256(dataset_path))

    assert again.id == model_version.load(trained).id


def test_refuses_data_with_the_wrong_checksum(dataset_path, tmp_path):
    with pytest.raises(ChecksumMismatchError):
        train(dataset_path, tmp_path / "model", expected_sha256="0" * 64)

    assert not (tmp_path / "model").exists()


def test_reading_the_dataset_converts_sentinels_and_negative_balcon_to_missing(dataset_path):
    dataset = read_dataset(dataset_path)

    for field in SENTINEL_FIELDS:
        assert dataset.applications[field].isna().any(), field
        assert not (dataset.applications[field] == -1).any(), field
    balcon = dataset.applications["intended_balcon_amount"]
    assert balcon.isna().any()
    assert (balcon.dropna() >= 0).all()
    assert list(dataset.applications.columns) == FIELDS


def _month(path, month):
    dataset = read_dataset(path)
    return dataset.where(np.asarray(dataset.months == month))


def test_editing_the_threshold_breaks_the_model_version_id(trained, tmp_path):
    copy = tmp_path / "model"
    shutil.copytree(trained, copy)
    metadata_path = copy / model_version.METADATA_FILE
    metadata = json.loads(metadata_path.read_text())
    metadata["threshold"] = metadata["threshold"] / 2
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="Model Version id"):
        model_version.load(copy)
