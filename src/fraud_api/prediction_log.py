"""The Prediction Log: a durable record of every Application scored and what was answered.

Writing is best-effort and happens on one writer thread of its own, fed by a bounded
queue. Scoring never waits for the database: when it is asleep or down, Predictions queue
up, and past the queue's size the newest are dropped and logged. No request thread is
ever held by a slow or dead database.
"""

import contextlib
import logging
import queue
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg
from psycopg.conninfo import conninfo_to_dict
from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)

DATABASE_URL_ENV = "DATABASE_URL"
CONNECT_TIMEOUT_SECONDS = 5
QUEUE_SIZE = 1000

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
# Added after the first release. Checked first, so the ALTER (which takes an exclusive
# lock even when the column exists) runs once per table, ever; old rows keep NULL.
HAS_TOP_FACTORS = """
SELECT 1 FROM information_schema.columns
WHERE table_schema = current_schema() AND table_name = 'prediction_log'
  AND column_name = 'top_factors'
"""
ADD_TOP_FACTORS = "ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS top_factors jsonb"

INSERT = """
INSERT INTO prediction_log
    (request_id, created_at, model_version, application, fraud_score, decision, latency_ms,
     top_factors)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
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
    """From the request reaching the service to the Decision being made."""
    top_factors: list[dict[str, Any]]


_STOP = object()


class PredictionLog:
    """Writes Predictions to Postgres at ``database_url``, or does nothing when it is
    ``None`` or empty."""

    def __init__(self, database_url: str | None, *, queue_size: int = QUEUE_SIZE) -> None:
        self._url = database_url or None
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        # Only the writer thread touches these after start().
        self._connection: psycopg.Connection | None = None
        self._table_ready = False

    @property
    def enabled(self) -> bool:
        return self._url is not None

    def start(self) -> None:
        """Create the table, then start the writer thread. An unreachable database delays
        this by at most the connect timeout; the table is then created on the first write."""
        if not self.enabled or self._thread is not None:
            return
        try:
            self._ensure_table()
        except Exception as error:  # best-effort: never let logging break scoring
            logger.warning("Prediction Log table not created at startup: %s", error)
            self._disconnect()
        self._thread = threading.Thread(target=self._run, name="prediction-log", daemon=True)
        self._thread.start()

    def submit(self, prediction: Prediction) -> None:
        """Queue ``prediction`` for writing. Never blocks."""
        if self._thread is None:
            return
        try:
            self._queue.put_nowait(prediction)
        except queue.Full:
            logger.warning("Prediction Log queue full; %s not logged", prediction.request_id)

    def close(self, timeout: float = 10.0) -> None:
        """Write what is queued, for up to ``timeout`` seconds, then stop."""
        if self._thread is None:
            return
        try:
            self._queue.put(_STOP, timeout=timeout)
        except queue.Full:
            logger.warning("Prediction Log queue full at shutdown; queued Predictions lost")
            return
        self._thread.join(timeout)
        self._thread = None

    def _run(self) -> None:
        while (item := self._queue.get()) is not _STOP:
            self._write(item)
        self._disconnect()

    def _write(self, prediction: Prediction) -> None:
        values = (
            prediction.request_id,
            prediction.created_at,
            prediction.model_version,
            Jsonb(prediction.application),
            prediction.fraud_score,
            prediction.decision,
            prediction.latency_ms,
            Jsonb(prediction.top_factors),
        )
        try:
            self._ensure_table()
            try:
                self._connected().execute(INSERT, values)
            except psycopg.errors.UndefinedTable:
                # Dropped or reset since it was created: create it again, once.
                self._table_ready = False
                self._ensure_table()
                self._connected().execute(INSERT, values)
        except Exception as error:  # best-effort: never let logging break scoring
            logger.warning("Prediction %s not logged: %s", prediction.request_id, error)
            self._disconnect()

    def _ensure_table(self) -> None:
        if not self._table_ready:
            connection = self._connected()
            connection.execute(CREATE_TABLE)
            if connection.execute(HAS_TOP_FACTORS).fetchone() is None:
                connection.execute(ADD_TOP_FACTORS)
            self._table_ready = True

    def _connected(self) -> psycopg.Connection:
        if self._connection is None or self._connection.closed:
            assert self._url is not None
            # A connect_timeout in the URL wins over the default.
            options = (
                {}
                if "connect_timeout" in conninfo_to_dict(self._url)
                else {"connect_timeout": CONNECT_TIMEOUT_SECONDS}
            )
            self._connection = psycopg.connect(self._url, autocommit=True, **options)
        return self._connection

    def _disconnect(self) -> None:
        if self._connection is not None:
            with contextlib.suppress(Exception):  # already broken; nothing left to release
                self._connection.close()
            self._connection = None
