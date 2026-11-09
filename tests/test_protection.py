"""Seam 1 again: the API protects itself without authentication."""

import pytest
from fastapi.testclient import TestClient

from fraud_api.application import EXAMPLE_APPLICATION
from fraud_api.protection import RateLimiter
from fraud_api.service import create_app

LIMIT = 3


@pytest.fixture
def client():
    with TestClient(create_app(rate_limit_per_minute=LIMIT, max_body_bytes=2048)) as client:
        yield client


def _predict(client, forwarded_for=None):
    headers = {"X-Forwarded-For": forwarded_for} if forwarded_for else {}
    return client.post("/predict", json=EXAMPLE_APPLICATION, headers=headers)


def test_the_request_past_the_limit_gets_429(client):
    statuses = [_predict(client, "203.0.113.7").status_code for _ in range(LIMIT + 1)]

    assert statuses == [200] * LIMIT + [429]


def test_429_says_when_to_retry(client):
    for _ in range(LIMIT):
        _predict(client, "203.0.113.7")

    response = _predict(client, "203.0.113.7")

    assert 1 <= int(response.headers["retry-after"]) <= 60


def test_another_forwarded_for_client_is_not_affected(client):
    for _ in range(LIMIT + 1):
        _predict(client, "203.0.113.7")

    assert _predict(client, "198.51.100.2").status_code == 200


def test_a_forged_forwarded_for_does_not_escape_the_limit(client):
    # Cloud Run appends the address it saw; the hops before it are the client's own.
    statuses = [
        _predict(client, f"198.51.100.{i}, 203.0.113.7").status_code for i in range(LIMIT + 1)
    ]

    assert statuses == [200] * LIMIT + [429]


def test_clients_behind_the_same_forged_hop_are_kept_apart(client):
    for _ in range(LIMIT + 1):
        _predict(client, "198.51.100.1, 203.0.113.7")

    assert _predict(client, "198.51.100.1, 203.0.113.8").status_code == 200


def test_without_forwarded_for_the_connection_address_is_the_client(client):
    statuses = [_predict(client).status_code for _ in range(LIMIT + 1)]

    assert statuses[-1] == 429


def test_health_is_never_limited(client):
    for _ in range(LIMIT + 1):
        _predict(client, "203.0.113.7")

    statuses = {client.get("/health").status_code for _ in range(LIMIT * 3)}

    assert statuses == {200}


def test_the_limit_rolls_over_after_the_window():
    now = [1000.0]
    limiter = RateLimiter(limit=2, window=60.0, clock=lambda: now[0])

    assert limiter.hit("a") is None
    assert limiter.hit("a") is None
    assert limiter.hit("a") == pytest.approx(60.0)
    now[0] += 30
    assert limiter.hit("a") == pytest.approx(30.0)
    now[0] += 30
    assert limiter.hit("a") is None


def test_a_client_sending_faster_than_the_limit_still_gets_the_limit_in_every_window():
    # Rejected requests must not count: a planted bug that recorded them passed the
    # tests above, then rejected a steady stream forever after its first window.
    now = [0.0]
    limiter = RateLimiter(limit=60, window=60.0, clock=lambda: now[0])
    windows, per_second = 5, 2
    allowed_per_window = [0] * windows
    promised = None

    for tick in range(int(windows * limiter.window * per_second)):
        now[0] = tick / per_second
        retry_after = limiter.hit("a")
        if retry_after is None:
            # Retry-After must name the moment the client gets through, not just be positive.
            if promised is not None:
                assert now[0] == pytest.approx(promised)
                promised = None
            allowed_per_window[int(now[0] // limiter.window)] += 1
        else:
            assert retry_after > 0
            promised = now[0] + retry_after

    assert allowed_per_window == [limiter.limit] * windows


def test_an_oversized_body_is_rejected_with_413(client):
    padded = EXAMPLE_APPLICATION | {"padding": "x" * 4000}

    response = client.post("/predict", json=padded)

    assert response.status_code == 413


def test_an_oversized_body_without_content_length_is_rejected_with_413(client):
    def chunks():
        yield b'{"income": 0.6, "padding": "'
        for _ in range(10):
            yield b"x" * 500
        yield b'"}'

    response = client.post(
        "/predict", content=chunks(), headers={"content-type": "application/json"}
    )

    assert response.status_code == 413


def test_a_body_under_the_cap_is_parsed_as_usual(client):
    response = client.post("/predict", json=EXAMPLE_APPLICATION | {"unexpected_field": 1})

    assert response.status_code == 422


def test_a_limit_below_one_is_refused():
    with pytest.raises(ValueError):
        RateLimiter(limit=0)


def test_tracks_at_most_max_clients_forgetting_the_least_recent():
    limiter = RateLimiter(limit=1, window=60.0, clock=lambda: 0.0, max_clients=2)
    limiter.hit("a")
    limiter.hit("b")
    limiter.hit("a")  # a is now the most recent; b the least
    limiter.hit("c")  # over the cap: b is forgotten

    assert limiter.hit("a") is not None  # a is still tracked, and over its limit
    assert limiter.hit("b") is None  # b starts afresh


def test_limits_can_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("FRAUD_API_RATE_LIMIT_PER_MINUTE", "1")
    monkeypatch.setenv("FRAUD_API_MAX_BODY_BYTES", "100")

    with TestClient(create_app()) as client:
        assert client.post("/predict", json=EXAMPLE_APPLICATION).status_code == 413
        assert client.post("/predict", content=b"{}").status_code == 422
        assert client.post("/predict", content=b"{}").status_code == 429
