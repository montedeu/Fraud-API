"""Seam 1: the HTTP API, against the committed Model Version."""

import json
import math
import shutil
import uuid

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_api import model_version
from fraud_api.application import EXAMPLE_APPLICATION, FIELDS, SENTINEL_FIELDS, ApplicationIn
from fraud_api.service import create_app

COMMITTED = model_version.load()


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(rate_limit_per_minute=10_000)) as client:
        yield client


def _with(**changes):
    return EXAMPLE_APPLICATION | changes


def test_scores_a_valid_application(client):
    response = client.post("/predict", json=EXAMPLE_APPLICATION)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"fraud_score", "decision", "model_version", "request_id", "top_factors"}
    assert 0 <= body["fraud_score"] <= 1
    assert body["decision"] in {"pass", "review"}
    assert body["model_version"] == COMMITTED.id
    uuid.UUID(body["request_id"])


def test_each_request_gets_its_own_request_id(client):
    first = client.post("/predict", json=EXAMPLE_APPLICATION).json()["request_id"]
    second = client.post("/predict", json=EXAMPLE_APPLICATION).json()["request_id"]

    assert first != second


@pytest.mark.parametrize(
    "reference", COMMITTED.metadata["reference_applications"], ids=lambda r: str(r["fraud_score"])
)
def test_decision_is_review_exactly_when_the_fraud_score_reaches_the_threshold(client, reference):
    body = client.post("/predict", json=reference["application"]).json()

    expected = "review" if body["fraud_score"] >= COMMITTED.threshold else "pass"
    assert body["decision"] == expected


def test_both_decisions_occur():
    # Guards the test above: the reference Applications must exercise both branches.
    scores = [r["fraud_score"] for r in COMMITTED.metadata["reference_applications"]]

    assert min(scores) < COMMITTED.threshold <= max(scores)


def test_training_and_serving_give_the_same_fraud_scores(client):
    for reference in COMMITTED.metadata["reference_applications"]:
        body = client.post("/predict", json=reference["application"]).json()

        assert body["fraud_score"] == pytest.approx(reference["fraud_score"], rel=1e-9, abs=1e-12)


@pytest.mark.parametrize(
    "reference", COMMITTED.metadata["reference_applications"], ids=lambda r: str(r["fraud_score"])
)
def test_top_factors_are_the_three_largest_contributions_with_the_applicant_s_values(
    client, reference
):
    application = reference["application"]
    factors = client.post("/predict", json=application).json()["top_factors"]

    assert len(factors) == 3
    assert len({f["field"] for f in factors}) == 3
    for factor in factors:
        assert factor["value"] == application[factor["field"]]
    sizes = [abs(f["contribution"]) for f in factors]
    assert sizes == sorted(sizes, reverse=True)


def test_training_and_serving_give_the_same_top_factors(client):
    for reference in COMMITTED.metadata["reference_applications"]:
        factors = client.post("/predict", json=reference["application"]).json()["top_factors"]

        expected = reference["top_factors"]
        assert [f["field"] for f in factors] == [f["field"] for f in expected]
        for served, trained in zip(factors, expected, strict=True):
            assert served["contribution"] == pytest.approx(trained["contribution"], rel=1e-9)


def test_contributions_add_up_to_the_fraud_score(client):
    # Every field's contribution plus the expected value is the raw score (log-odds),
    # and the Fraud Score is its logistic transform.
    for reference in COMMITTED.metadata["reference_applications"]:
        fraud_score = client.post("/predict", json=reference["application"]).json()["fraud_score"]
        contributions = COMMITTED.contributions(
            pd.DataFrame([reference["application"]], columns=FIELDS)
        )

        raw = contributions.sum(axis=1).iloc[0]
        assert 1 / (1 + math.exp(-raw)) == pytest.approx(fraud_score, rel=1e-9)


def test_top_factors_name_null_values_too(client):
    nulls = {field: None for field in [*SENTINEL_FIELDS, "intended_balcon_amount"]}

    factors = client.post("/predict", json=_with(**nulls)).json()["top_factors"]

    assert all(
        f["value"] == nulls.get(f["field"], EXAMPLE_APPLICATION[f["field"]]) for f in factors
    )


def test_accepts_null_for_every_field_that_may_be_missing(client):
    nulls = {field: None for field in [*SENTINEL_FIELDS, "intended_balcon_amount"]}

    response = client.post("/predict", json=_with(**nulls))

    assert response.status_code == 200


def test_accepts_negative_credit_risk_score_and_velocity_6h(client):
    response = client.post("/predict", json=_with(credit_risk_score=-120, velocity_6h=-40.5))

    assert response.status_code == 200


@pytest.mark.parametrize(
    "changes",
    [
        {"payment_type": "AZ"},
        {"device_os": "android"},
        {"customer_age": 35},
        {"customer_age": 100},
        {"income": 0.65},
        {"income": 1.0},
        {"proposed_credit_limit": 750},
        {"email_is_free": 2},
        {"email_is_free": True},
        {"keep_alive_session": False},
        {"bank_months_count": -1},
        {"intended_balcon_amount": -3.2},
        {"zip_count_4w": -1},
        {"credit_risk_score": None},
        {"customer_age": "30"},
        {"unexpected_field": 1},
    ],
    ids=str,
)
def test_rejects_invalid_applications(client, changes):
    response = client.post("/predict", json=_with(**changes))

    assert response.status_code == 422


def test_rejects_a_missing_field(client):
    application = dict(EXAMPLE_APPLICATION)
    del application["housing_status"]

    response = client.post("/predict", json=application)

    assert response.status_code == 422


@pytest.mark.parametrize("value", [float("nan"), float("inf")], ids=str)
def test_rejects_non_finite_numbers(client, value):
    # Python's JSON parser accepts NaN and Infinity, so the contract must reject them.
    response = client.post(
        "/predict",
        content=json.dumps(_with(velocity_6h=value)),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422


def test_rejections_name_the_field_and_the_rejected_value(client):
    response = client.post("/predict", json=_with(income=0.65))

    [error] = response.json()["detail"]
    assert error["loc"] == ["body", "income"]
    assert error["input"] == 0.65
    assert "income must be one of" in error["ctx"]["error"]


def test_the_api_contract_has_exactly_the_application_fields():
    assert list(ApplicationIn.model_fields) == FIELDS


def test_refuses_to_start_with_a_model_version_built_on_another_contract(tmp_path):
    copy = tmp_path / "model"
    shutil.copytree(model_version.MODEL_DIR, copy)
    metadata_path = copy / model_version.METADATA_FILE
    metadata = json.loads(metadata_path.read_text())
    metadata["categories"]["device_os"] = ["linux", "windows"]
    metadata["model_version"] = model_version.model_version_id(
        (copy / model_version.MODEL_FILE).read_text(), metadata
    )
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="API contract"), TestClient(create_app(copy)):
        pass


def test_health_returns_the_model_version(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": COMMITTED.id}


def test_docs_show_an_example_application(client):
    schema = client.get("/openapi.json").json()

    assert schema["components"]["schemas"]["ApplicationIn"]["examples"] == [EXAMPLE_APPLICATION]
