import pytest


@pytest.fixture(autouse=True)
def no_real_database(monkeypatch):
    """Tests never write to a real Prediction Log, even with DATABASE_URL exported."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
