"""The README's numbers come from the committed Model Version, never by hand."""

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fraud_api import readme
from fraud_api.application import EXAMPLE_APPLICATION
from fraud_api.model_version import METADATA_FILE, MODEL_DIR
from fraud_api.service import create_app

ROOT = Path(__file__).resolve().parent.parent
METADATA = json.loads((ROOT / MODEL_DIR / METADATA_FILE).read_text())


def test_the_readme_matches_the_committed_model_version():
    current = (ROOT / readme.README).read_text()

    assert readme.render(current, METADATA) == current, "run: uv run fraud-api readme"


def test_a_changed_metric_changes_the_readme():
    current = (ROOT / readme.README).read_text()
    changed = copy.deepcopy(METADATA)
    changed["metrics"]["test"]["recall_at_5pct_fpr"] = 0.123

    assert "12.3%" in readme.render(current, changed)


def test_a_readme_without_a_generated_part_is_refused():
    current = (ROOT / readme.README).read_text()
    without_results = current.replace("<!-- /generated:test-results -->", "")

    with pytest.raises(ValueError, match="test-results"):
        readme.render(without_results, METADATA)


@pytest.mark.parametrize(
    ("served", "without_age", "wording"),
    [
        (0.32, 0.42, "raises the FPR Ratio from 0.32 to 0.42"),
        (0.42, 0.32, "lowers the FPR Ratio from 0.42 to 0.32"),
        (0.321, 0.324, "leaves the FPR Ratio at 0.32"),
    ],
)
def test_the_without_age_wording_follows_the_numbers(served, without_age, wording):
    current = (ROOT / readme.README).read_text()
    changed = copy.deepcopy(METADATA)
    changed["metrics"]["test"]["fpr_ratio"] = served
    changed["experiments"]["without_age"]["metrics"]["test"]["fpr_ratio"] = without_age

    assert wording in readme.render(current, changed)


def test_the_readme_example_is_what_the_api_answers():
    current = (ROOT / readme.README).read_text()
    shown = json.loads(current.split("```json\n", 1)[1].split("\n```", 1)[0])
    with TestClient(create_app()) as client:
        answer = client.post("/predict", json=EXAMPLE_APPLICATION).json()

    shown.pop("request_id"), answer.pop("request_id")
    # As JSON text, so 20 and 20.0 count as different.
    assert json.dumps(shown) == json.dumps(answer)
