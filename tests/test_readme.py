"""The README's numbers come from the committed Model Version, never by hand."""

import json
from pathlib import Path

import pytest

from fraud_api import readme

ROOT = Path(__file__).resolve().parent.parent
METADATA = json.loads((ROOT / "model" / "metadata.json").read_text())


def test_the_readme_matches_the_committed_model_version():
    current = (ROOT / readme.README).read_text()

    assert readme.render(current, METADATA) == current, "run: uv run fraud-api readme"


def test_a_changed_metric_changes_the_readme():
    current = (ROOT / readme.README).read_text()
    changed = json.loads(json.dumps(METADATA))
    changed["metrics"]["test"]["recall_at_5pct_fpr"] = 0.123

    assert "12.3%" in readme.render(current, changed)


def test_a_readme_without_a_generated_part_is_refused():
    current = (ROOT / readme.README).read_text()
    without_results = current.replace("<!-- /generated:test-results -->", "")

    with pytest.raises(ValueError, match="test-results"):
        readme.render(without_results, METADATA)
