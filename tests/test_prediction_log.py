"""Seam 1 with a real Postgres: every scored Application lands in the Prediction Log."""

import os
import socket
import threading
import time
import uuid
from urllib.parse import urlsplit, urlunsplit

import httpx2 as httpx
import psycopg
import pytest
import uvicorn
from fastapi.testclient import TestClient

from fraud_api.application import EXAMPLE_APPLICATION
from fraud_api.service import create_app

COLUMNS = {
    "request_id",
    "created_at",
    "model_version",
    "application",
    "fraud_score",
    "decision",
    "latency_ms",
}


@pytest.fixture(scope="session")
def postgres_url(tmp_path_factory):
    """``$TEST_DATABASE_URL`` (CI's service container), else an embedded Postgres."""
    if url := os.environ.get("TEST_DATABASE_URL"):
        yield url
        return
    pgserver = pytest.importorskip("pgserver")
    server = pgserver.get_server(tmp_path_factory.mktemp("pgdata"), cleanup_mode="stop")
    yield server.get_uri()
    server.cleanup()


@pytest.fixture
def database_url(postgres_url):
    with psycopg.connect(postgres_url, autocommit=True) as connection:
        connection.execute("DROP TABLE IF EXISTS prediction_log")
    return postgres_url


def _rows(url):
    with psycopg.connect(url) as connection, connection.cursor() as cursor:
        cursor.execute("SELECT * FROM prediction_log")
        names = [column.name for column in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _columns(url):
    with psycopg.connect(url) as connection:
        rows = connection.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'prediction_log'"
        ).fetchall()
    return {name for (name,) in rows}


def test_one_prediction_leaves_exactly_one_row_with_what_was_answered(database_url):
    with TestClient(create_app(database_url=database_url)) as client:
        body = client.post("/predict", json=EXAMPLE_APPLICATION).json()

    [row] = _rows(database_url)
    assert row["request_id"] == uuid.UUID(body["request_id"])
    assert row["model_version"] == body["model_version"]
    assert row["fraud_score"] == body["fraud_score"]
    assert row["decision"] == body["decision"]
    assert row["application"] == EXAMPLE_APPLICATION
    assert row["latency_ms"] > 0
    assert row["created_at"].tzinfo is not None


def test_the_table_holds_no_ip_address_or_headers(database_url):
    with TestClient(create_app(database_url=database_url)) as client:
        client.post(
            "/predict", json=EXAMPLE_APPLICATION, headers={"X-Forwarded-For": "203.0.113.7"}
        )

    assert _columns(database_url) == COLUMNS
    assert "203.0.113.7" not in str(_rows(database_url))


def test_the_table_is_created_at_startup(database_url):
    with TestClient(create_app(database_url=database_url)):
        assert _columns(database_url) == COLUMNS


def test_the_table_is_created_on_first_write_when_the_database_was_down_at_startup(
    postgres_url,
):
    # A database that doesn't exist yet stands in for one that is unreachable at startup.
    name = f"late_{uuid.uuid4().hex[:8]}"
    parts = urlsplit(postgres_url)
    late_url = urlunsplit(parts._replace(path=f"/{name}"))

    with TestClient(create_app(database_url=late_url)) as client:
        with psycopg.connect(postgres_url, autocommit=True) as connection:
            connection.execute(f"CREATE DATABASE {name}")
        response = client.post("/predict", json=EXAMPLE_APPLICATION)

    assert response.status_code == 200
    assert len(_rows(late_url)) == 1


def test_scoring_works_when_the_database_is_unreachable():
    unreachable = "postgresql://user:password@127.0.0.1:1/fraud?connect_timeout=1"

    with TestClient(create_app(database_url=unreachable)) as client:
        response = client.post("/predict", json=EXAMPLE_APPLICATION)

    assert response.status_code == 200
    assert response.json()["decision"] in {"pass", "review"}


def test_with_no_database_url_nothing_is_attempted(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    def refuse(*args, **kwargs):
        raise AssertionError("no database URL is configured, so nothing should connect")

    monkeypatch.setattr(psycopg, "connect", refuse)

    with TestClient(create_app()) as client:
        response = client.post("/predict", json=EXAMPLE_APPLICATION)

    assert response.status_code == 200


def test_the_response_does_not_wait_for_the_write():
    """A database that accepts the connection and then never answers: the response still
    arrives at once, because the write runs after it was sent."""
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen()
        port = silent.getsockname()[1]
        url = f"postgresql://user:password@127.0.0.1:{port}/fraud?connect_timeout=3"

        with _running(create_app(database_url=url)) as base_url:
            started = time.perf_counter()
            response = httpx.post(f"{base_url}/predict", json=EXAMPLE_APPLICATION, timeout=10)
            elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert elapsed < 1.0


class _running:
    """The app under a real uvicorn server, in a thread."""

    def __init__(self, app):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        deadline = time.monotonic() + 15
        while not self.server.started:
            if time.monotonic() > deadline:
                raise TimeoutError("server did not start")
            time.sleep(0.05)
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=10)
