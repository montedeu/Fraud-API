"""Smoke-test a running Fraud API: python scripts/smoke_test.py https://<service-url>

Checks that the service at the URL serves the committed Model Version: ``/health``
names it, ``/predict`` scores a reference Application from its metadata to the Fraud
Score recorded at training time, with the Decision its Threshold implies, and ``/docs``
loads. Exits non-zero on the first failure. Standard library only, so it runs on a bare
CI runner without installing the project.
"""

import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

METADATA = Path(__file__).resolve().parent.parent / "model" / "metadata.json"
TIMEOUT_SECONDS = 30


def _request(url: str, body: dict | None = None) -> tuple[int, bytes]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method="GET" if body is None else "POST")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:  # a 4xx/5xx is a failed check, not a crash
        return error.code, error.read()


def _check(condition: bool, message: str) -> None:
    if not condition:
        sys.exit(f"FAIL {message}")
    print(f"ok   {message}")


def main(base_url: str) -> None:
    base_url = base_url.rstrip("/")
    metadata = json.loads(METADATA.read_text())
    expected_version = metadata["model_version"]
    reference = metadata["reference_applications"][0]

    status, body = _request(f"{base_url}/health")
    health = json.loads(body)
    _check(status == 200 and health.get("status") == "ok", f"/health is ok: {health}")
    _check(
        health.get("model_version") == expected_version,
        f"/health serves Model Version {expected_version}",
    )

    status, body = _request(f"{base_url}/predict", reference["application"])
    scored = json.loads(body)
    _check(status == 200, f"/predict answers 200: {scored}")
    _check(
        scored.get("model_version") == expected_version,
        f"/predict names Model Version {expected_version}",
    )
    fraud_score = scored.get("fraud_score")
    _check(
        isinstance(fraud_score, float)
        and math.isclose(fraud_score, reference["fraud_score"], rel_tol=1e-9, abs_tol=1e-12),
        f"/predict Fraud Score {fraud_score} matches the training-time {reference['fraud_score']}",
    )
    expected_decision = "review" if fraud_score >= metadata["threshold"] else "pass"
    _check(scored.get("decision") == expected_decision, f"/predict Decision is {expected_decision}")
    _check(bool(scored.get("request_id")), "/predict returns a request id")

    status, body = _request(f"{base_url}/docs")
    _check(status == 200 and b"swagger" in body.lower(), "/docs loads")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
