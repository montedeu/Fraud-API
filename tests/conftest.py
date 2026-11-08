import os

import psycopg
import pytest


@pytest.fixture(autouse=True)
def no_real_database(monkeypatch):
    """Tests never write to a real Prediction Log, even with DATABASE_URL exported."""
    monkeypatch.delenv("DATABASE_URL", raising=False)


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
