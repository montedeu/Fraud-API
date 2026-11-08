import json
import shutil

import numpy as np
import pandas as pd
import pytest
from synthetic import make_dataset, write_dataset

from fraud_api import evaluation, model_version, reports
from fraud_api.application import CATEGORIES, FIELDS, SENTINEL_FIELDS, read_dataset
from fraud_api.data import ChecksumMismatchError, sha256
from fraud_api.training import train as train_pipeline

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


TRIALS = 3
RESAMPLES = 50


def train(data_path, output_dir, **kwargs):
    """The pipeline with a handful of trials and the reports next to the model."""
    kwargs.setdefault("expected_sha256", sha256(data_path))
    return train_pipeline(
        data_path,
        output_dir,
        reports_dir=output_dir / "reports",
        n_trials=TRIALS,
        n_resamples=RESAMPLES,
        **kwargs,
    )


@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("data") / "Base.csv", make_dataset())


@pytest.fixture(scope="module")
def trained(dataset_path, tmp_path_factory):
    output = tmp_path_factory.mktemp("model")
    train(dataset_path, output)
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


def test_contributions_add_up_to_the_fraud_score(trained, dataset_path):
    # Every field's contribution plus the expected value is the raw score (log-odds),
    # and the Fraud Score is its logistic transform.
    loaded = model_version.load(trained)
    applications = _month(dataset_path, 6).applications.head(50)

    raw = loaded.contributions(applications).sum(axis=1).to_numpy()

    assert 1 / (1 + np.exp(-raw)) == pytest.approx(loaded.score(applications), rel=1e-9)


def test_each_reference_application_records_its_top_factors(trained):
    for reference in model_version.load(trained).metadata["reference_applications"]:
        factors = reference["top_factors"]

        assert len(factors) == model_version.TOP_FACTOR_COUNT
        assert all(reference["application"][f["field"]] == f["value"] for f in factors)


def test_threshold_gives_5_percent_fpr_on_the_validation_month(trained, dataset_path):
    loaded = model_version.load(trained)
    validation = _month(dataset_path, 5)

    fpr = evaluation.fpr_at(
        validation.labels, loaded.score(validation.applications), loaded.threshold
    )

    assert fpr == pytest.approx(0.05, abs=0.005)
    assert fpr <= 0.05


def test_metadata_reports_both_baselines_on_every_split(trained):
    baselines = model_version.load(trained).metadata["baselines"]

    assert set(baselines) == {"logistic_regression", "credit_risk_score_rule"}
    for name, baseline in baselines.items():
        assert baseline["settings"], name
        assert baseline["metrics"].keys() == {"training", "validation", "test"}, name
        for split in baseline["metrics"].values():
            assert split.keys() >= REQUIRED_METRICS, name


def test_lightgbm_without_age_is_reported_on_every_split_and_never_sees_age(trained):
    without_age = model_version.load(trained).metadata["experiments"]["without_age"]

    assert "customer_age" not in without_age["settings"]["features"]
    assert set(without_age["settings"]["features"]) == set(FIELDS) - {"customer_age"}
    assert without_age["metrics"].keys() == {"training", "validation", "test"}
    for split in without_age["metrics"].values():
        assert split.keys() >= REQUIRED_METRICS


def test_metadata_has_the_validation_month_as_the_monitoring_reference(trained):
    reference = model_version.load(trained).metadata["monitoring"]

    assert reference["reference_month"] == 5
    assert 0 < reference["expected_review_share"] < 1
    assert list(reference["features"]) == FIELDS
    for field, entry in reference["features"].items():
        assert sum(entry["shares"].values()) == pytest.approx(1), field


def test_metadata_has_paired_bootstrap_intervals_for_every_model(trained):
    bootstrap = model_version.load(trained).metadata["bootstrap"]

    models = bootstrap["models"]
    assert set(models) == {
        "lightgbm",
        "logistic_regression",
        "credit_risk_score_rule",
        "without_age",
    }
    for name, result in models.items():
        assert set(result["interval"]) == {"recall_at_5pct_fpr", "pr_auc", "fpr_ratio"}, name
        for low, high in result["interval"].values():
            assert low <= high, name
        assert ("difference" in result) == (name != bootstrap["reference"]), name


@pytest.mark.parametrize(
    "path",
    [
        ("metrics",),
        ("baselines", "logistic_regression", "metrics"),
        ("baselines", "credit_risk_score_rule", "metrics"),
        ("experiments", "without_age", "metrics"),
    ],
    ids=["lightgbm", "logistic_regression", "credit_risk_score_rule", "without_age"],
)
def test_each_model_threshold_gives_5_percent_fpr_on_the_validation_month(trained, path):
    metrics = model_version.load(trained).metadata
    for key in path:
        metrics = metrics[key]

    assert metrics["validation"]["fpr_at_threshold"] == pytest.approx(0.05, abs=0.005)
    assert metrics["validation"]["fpr_at_threshold"] <= 0.05


def test_label_fields_and_month_are_not_model_features(trained):
    features = model_version.load(trained).booster.feature_name()

    assert {"month", "fraud_bool", "device_fraud_count"}.isdisjoint(features)
    assert features == FIELDS


def test_perturbing_test_month_labels_changes_neither_threshold_nor_model(trained, tmp_path):
    rows = make_dataset()
    test_months = rows["month"].isin([6, 7])
    rows.loc[test_months, "fraud_bool"] = 1 - rows.loc[test_months, "fraud_bool"]
    perturbed_path = write_dataset(tmp_path / "Base.csv", rows)

    perturbed = train(perturbed_path, tmp_path / "model")
    original = model_version.load(trained)

    assert perturbed.threshold == original.threshold
    assert perturbed.id == original.id
    for path in [
        ("experiments", "without_age"),
        ("baselines", "logistic_regression"),
        ("baselines", "credit_risk_score_rule"),
    ]:
        perturbed_model, original_model = perturbed.metadata, original.metadata
        for key in path:
            perturbed_model, original_model = perturbed_model[key], original_model[key]
        assert perturbed_model["threshold"] == original_model["threshold"], path


def test_same_seed_reproduces_the_model_version_id(trained, dataset_path, tmp_path):
    again = train(dataset_path, tmp_path / "model")

    assert again.id == model_version.load(trained).id


def test_metadata_has_the_optuna_summary(trained):
    summary = model_version.load(trained).metadata["optuna"]

    assert summary["n_trials"] == TRIALS
    assert summary["tuning_month"] == 4
    assert summary["trial_fit_months"] == [0, 1, 2, 3]
    assert set(summary["best_params"]) == {
        "num_leaves",
        "learning_rate",
        "min_child_samples",
        "feature_fraction",
        "bagging_fraction",
        "lambda_l1",
        "lambda_l2",
        "scale_pos_weight",
    }
    assert summary["best_num_boost_round"] >= 1
    assert 0 <= summary["tuning_month_default_score"] <= 1
    assert 0 <= summary["tuning_month_tuned_score"] <= 1
    assert set(summary["param_importances"]) == set(summary["best_params"])


def test_the_final_model_has_the_best_trial_s_tree_count(trained):
    loaded = model_version.load(trained)

    assert loaded.booster.num_trees() == loaded.metadata["optuna"]["best_num_boost_round"]


def test_writes_the_trials_table_and_the_importance_plot(trained):
    table = pd.read_csv(trained / "reports" / reports.TRIALS_FILE)

    assert len(table) == TRIALS
    assert "tuning_month_recall_at_5pct_fpr" in table.columns
    assert (trained / "reports" / reports.IMPORTANCE_PLOT).stat().st_size > 0


def test_perturbing_validation_and_test_month_labels_changes_no_tuning_choice(trained, tmp_path):
    rows = make_dataset()
    held_out = rows["month"].isin([5, 6, 7])
    rows.loc[held_out, "fraud_bool"] = 1 - rows.loc[held_out, "fraud_bool"]
    perturbed_path = write_dataset(tmp_path / "Base.csv", rows)

    perturbed = train(perturbed_path, tmp_path / "model").metadata["optuna"]
    original = model_version.load(trained).metadata["optuna"]

    assert perturbed["best_params"] == original["best_params"]
    assert perturbed["best_num_boost_round"] == original["best_num_boost_round"]


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


def test_fewer_than_two_trials_are_refused_before_any_work(dataset_path, tmp_path):
    with pytest.raises(ValueError, match="n_trials"):
        train_pipeline(
            dataset_path, tmp_path / "model", expected_sha256=sha256(dataset_path), n_trials=1
        )

    assert not (tmp_path / "model").exists()


def test_cli_defaults_match_the_pipeline():
    from fraud_api import cli, training, tuning

    assert (cli.SEED, cli.N_TRIALS, cli.REPORTS_DIR) == (
        training.SEED,
        tuning.N_TRIALS,
        reports.REPORTS_DIR,
    )
