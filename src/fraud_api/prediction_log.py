"""The Prediction Log: a durable record of every Application scored and what was answered.

Writing is best-effort. A failure is logged and never reaches the API client: scoring
must keep working while the database is asleep or down.
"""

import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)

DATABASE_URL_ENV = "DATABASE_URL"
CONNECT_TIMEOUT_SECONDS = 5

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS prediction_log (
    request_id    uuid PRIMARY KEY,
    created_at    timestamptz NOT NULL,
    model_version text NOT NULL,
    application   jsonb NOT NULL,
    fraud_score   double precision NOT NULL,
    decision      text NOT NULL CHECK (decision IN ('pass', 'review')),
    latency_ms    double precision NOT NULL
)
"""

INSERT = """
INSERT INTO prediction_log
    (request_id, created_at, model_version, application, fraud_score, decision, latency_ms)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""


@dataclass(frozen=True)
class Prediction:
    """What the Prediction Log keeps about one scored Application. Deliberately no IP
    address and no headers."""

    request_id: uuid.UUID
    created_at: datetime
    model_version: str
    application: dict[str, Any]
    fraud_score: float
    decision: str
    latency_ms: float


class PredictionLog:
    """Writes Predictions to Postgres at ``database_url``, or does nothing when it is
    ``None`` or empty."""

    def __init__(self, database_url: str | None) -> None:
        self._url = database_url or None
        self._table_ready = False
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._url is not None

    def ensure_table(self) -> bool:
        """Create the table if it is missing. False, and logged, if the database is
        unreachable; the next write tries again."""
        if not self.enabled:
            return False
        with self._lock:
            if self._table_ready:
                return True
            try:
                with self._connect() as connection:
                    connection.execute(CREATE_TABLE)
            except Exception as error:  # best-effort: never let logging break scoring
                logger.warning("Prediction Log table not ready: %s", error)
                return False
            self._table_ready = True
            return True

    def write(self, prediction: Prediction) -> None:
        """Record ``prediction``, best-effort."""
        if not self.enabled or not self.ensure_table():
            return
        try:
            with self._connect() as connection:
                connection.execute(
                    INSERT,
                    (
                        prediction.request_id,
                        prediction.created_at,
                        prediction.model_version,
                        Jsonb(prediction.application),
                        prediction.fraud_score,
                        prediction.decision,
                        prediction.latency_ms,
                    ),
                )
        except Exception as error:  # best-effort: never let logging break scoring
            logger.warning("Prediction %s not logged: %s", prediction.request_id, error)

    def _connect(self) -> psycopg.Connection:
        assert self._url is not None
        return psycopg.connect(self._url, connect_timeout=CONNECT_TIMEOUT_SECONDS)
