"""Smoke-test a running Fraud API: python scripts/smoke_test.py https://<service-url>

Checks that the service at the URL serves the committed Model Version: ``/health``
names it, ``/predict`` scores one Pass and one Review reference Application from its
metadata to the Fraud Scores, Decisions and top factors recorded at training time, and ``/docs``
loads. Exits non-zero on the first failure. Standard library only, so it runs on a bare
CI runner without installing the project.
"""

import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# model_version.MODEL_DIR / METADATA_FILE; not imported, so the project needn't be installed.
METADATA = Path(__file__).resolve().parent.parent / "model" / "metadata.json"
TIMEOUT_SECONDS = 30
ATTEMPTS = 3
"""A 5xx or a dropped connection is retried: a fresh instance may still be warming up."""


def _request(url: str, body: dict | None = None) -> tuple[int, bytes]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method="GET" if body is None else "POST")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:  # a 4xx/5xx is a failed check, not a crash
            if error.code < 500 or attempt == ATTEMPTS:
                return error.code, error.read()
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == ATTEMPTS:
                sys.exit(f"FAIL {url}: {error}")
        time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def _check(condition: bool, message: str) -> None:
    if not condition:
        sys.exit(f"FAIL {message}")
    print(f"ok   {message}")


def _json(status: int, body: bytes, path: str) -> dict:
    _check(
        status == 200,
        f"{path} answers 200" + ("" if status == 200 else f": {status} {body[:200]!r}"),
    )
    return json.loads(body)


def main(base_url: str) -> None:
    base_url = base_url.rstrip("/")
    metadata = json.loads(METADATA.read_text())
    expected_version = metadata["model_version"]
    threshold = metadata["threshold"]

    health = _json(*_request(f"{base_url}/health"), "/health")
    _check(health == {"status": "ok", "model_version": expected_version}, f"/health: {health}")

    # One reference Application per Decision, so both sides of the Threshold are seen
    # without filling the Prediction Log with smoke-test rows.
    references = {}
    for reference in metadata["reference_applications"]:
        decision = "review" if reference["fraud_score"] >= threshold else "pass"
        references.setdefault(decision, reference)
    _check(set(references) == {"pass", "review"}, "the references cover Pass and Review")

    for decision, reference in references.items():
        scored = _json(*_request(f"{base_url}/predict", reference["application"]), "/predict")
        fraud_score = scored.get("fraud_score")
        _check(
            scored.get("model_version") == expected_version
            and scored.get("decision") == decision
            and bool(scored.get("request_id"))
            and isinstance(fraud_score, float)
            and math.isclose(fraud_score, reference["fraud_score"], rel_tol=1e-9, abs_tol=1e-12)
            and [f["field"] for f in scored.get("top_factors", [])]
            == [f["field"] for f in reference["top_factors"]],
            f"/predict scores a {decision} reference as at training time: {scored}",
        )

    status, body = _request(f"{base_url}/docs")
    _check(status == 200 and b"swagger" in body.lower(), "/docs loads")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
