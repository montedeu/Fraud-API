"""Drift monitoring from the Prediction Log: pure functions, then the command against a real
Postgres."""

import json
import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import psycopg
import pytest
from psycopg.types.json import Jsonb

from fraud_api import cli, model_version, monitoring, prediction_log
from fraud_api.application import EXAMPLE_APPLICATION

COMMITTED = model_version.load().metadata
# Not one of the reference Applications, which the report leaves out as smoke tests (the
# /docs example is one of them).
LIVE_APPLICATION = EXAMPLE_APPLICATION | {"income": 0.1, "customer_age": 40}


def _applications(n=2000, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "amount": np.where(rng.random(n) < 0.2, np.nan, rng.normal(100 + shift, 10, n)),
            "device": rng.choice(["a", "b"], n, p=[0.7, 0.3]),
        }
    )


REFERENCE = monitoring.reference_distribution(_applications(), {"device": ["a", "b"]})


class TestReferenceAndPsi:
    def test_the_reference_month_itself_shows_no_drift(self):
        drift = monitoring.drift(REFERENCE, _applications(seed=1))

        assert all(value < monitoring.PSI_WARNING for value in drift.values())

    def test_a_shifted_field_is_flagged_and_an_unshifted_one_is_not(self):
        drift = monitoring.drift(REFERENCE, _applications(shift=8.0, seed=1))

        assert drift["amount"] > monitoring.PSI_WARNING
        assert drift["device"] < monitoring.PSI_WARNING

    def test_identical_shares_have_zero_psi(self):
        shares = {"a": 0.7, "b": 0.3}

        assert monitoring.psi(shares, shares) == 0

    def test_a_hand_computed_psi(self):
        # (0.4 - 0.7) ln(0.4 / 0.7) + (0.6 - 0.3) ln(0.6 / 0.3)
        expected = -0.3 * np.log(0.4 / 0.7) + 0.3 * np.log(2)

        assert monitoring.psi({"a": 0.7, "b": 0.3}, {"a": 0.4, "b": 0.6}) == pytest.approx(expected)

    def test_null_and_numbers_compare_the_same_whatever_their_type(self):
        reference = {"kind": "values", "shares": {"1": 0.5, "missing": 0.5}}

        observed = monitoring.bin_shares(pd.Series([1, 1.0, None, float("nan")]), reference)

        assert observed == {"1": 0.5, "missing": 0.5}


def _log(rows):
    return pd.DataFrame(rows, columns=monitoring.PREDICTION_LOG_COLUMNS)


def test_the_report_counts_each_day_and_leaves_out_other_models_and_smoke_tests():
    day = datetime(2026, 11, 9, 12, tzinfo=UTC)
    smoke = COMMITTED["reference_applications"][0]["application"]
    rows = (
        [(day, COMMITTED["model_version"], LIVE_APPLICATION, 0.2, "review")] * 2
        + [(day, COMMITTED["model_version"], LIVE_APPLICATION, 0.05, "pass")] * 6
        + [(day, "lgbm-older", LIVE_APPLICATION, 0.9, "review")] * 5
        + [(day, COMMITTED["model_version"], smoke, 0.9, "review")] * 3
    )

    report = monitoring.report(
        _log(rows),
        COMMITTED["monitoring"],
        model_version=COMMITTED["model_version"],
        exclude=[r["application"] for r in COMMITTED["reference_applications"]],
    )

    assert "| 2026-11-09 | 8 | 25.0% | 0.0875 | too few to judge |" in report


def test_the_report_names_the_features_that_drifted():
    day = datetime(2026, 11, 9, 12, tzinfo=UTC)
    # Every Application the same: whatever varies in the Validation Month has drifted.
    rows = [(day, COMMITTED["model_version"], LIVE_APPLICATION, 0.01, "pass")] * 500

    report = monitoring.report(
        _log(rows), COMMITTED["monitoring"], model_version=COMMITTED["model_version"]
    )

    [line] = [line for line in report.splitlines() if line.startswith("| 2026-11-09")]
    assert "`housing_status`" in line
    assert "none" not in line


def test_an_empty_period_says_so():
    report = monitoring.report(
        _log([]), COMMITTED["monitoring"], model_version=COMMITTED["model_version"]
    )

    assert "no Applications in this period" in report


def test_fraud_api_monitor_reads_the_prediction_log(database_url, monkeypatch, capsys):
    now = datetime.now(UTC)
    smoke = COMMITTED["reference_applications"][0]["application"]
    rows = (
        [(now, LIVE_APPLICATION, 0.2, "review")] * 100
        + [(now, LIVE_APPLICATION, 0.05, "pass")] * 400
        + [(now, smoke, 0.01, "pass")] * 3  # the smoke test's: left out
        + [(now - timedelta(days=30), LIVE_APPLICATION, 0.05, "pass")]  # outside 7 days
    )
    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute(prediction_log.CREATE_TABLE)
        connection.cursor().executemany(
            "INSERT INTO prediction_log VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [
                (
                    uuid.uuid4(),
                    created_at,
                    COMMITTED["model_version"],
                    Jsonb(json.loads(json.dumps(application))),
                    fraud_score,
                    decision,
                    1.0,
                )
                for created_at, application, fraud_score, decision in rows
            ],
        )
    monkeypatch.setenv("DATABASE_URL", database_url)

    cli.main(["monitor"])

    [line] = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith(f"| {now.date()}")
    ]
    assert line.startswith(f"| {now.date()} | 500 | 20.0% | 0.0800 |")
    # 500 copies of one Application: whatever varies in the Validation Month has drifted.
    assert "`housing_status`" in line


def test_fraud_api_monitor_needs_a_database_url():
    with pytest.raises(SystemExit, match="DATABASE_URL"):
        cli.main(["monitor"])


def test_fraud_api_monitor_needs_at_least_one_day(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")

    with pytest.raises(SystemExit, match="--days"):
        cli.main(["monitor", "--days", "0"])
