"""The scoring service: POST an Application, get a Fraud Score, a Decision and the top
factors behind it."""

import math
import os
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Receive, Scope, Send

from fraud_api import model_version, protection
from fraud_api.application import CATEGORIES, EXAMPLE_APPLICATION, FIELDS, ApplicationIn, to_frame
from fraud_api.prediction_log import DATABASE_URL_ENV, Prediction, PredictionLog

MODEL_DIR_ENV = "FRAUD_API_MODEL_DIR"
RATE_LIMIT_ENV = "FRAUD_API_RATE_LIMIT_PER_MINUTE"
MAX_BODY_BYTES_ENV = "FRAUD_API_MAX_BODY_BYTES"


class Factor(BaseModel):
    field: str = Field(description="An Application field.")
    value: str | int | float | None = Field(description="The Applicant's value; null if missing.")
    contribution: float = Field(
        description="How far this field moved the model's raw score, in log-odds (not a "
        "probability). Positive pushes towards Review, negative towards Pass."
    )


class Scored(BaseModel):
    fraud_score: float
    decision: Literal["pass", "review"]
    model_version: str
    request_id: uuid.UUID
    top_factors: list[Factor] = Field(
        description=f"The {model_version.TOP_FACTOR_COUNT} fields that moved this Fraud Score "
        "the most, largest first."
    )


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str


def create_app(
    model_dir: Path | None = None,
    *,
    rate_limit_per_minute: int | None = None,
    max_body_bytes: int | None = None,
    database_url: str | None = None,
) -> FastAPI:
    """The FastAPI app. The Model Version is loaded once, at startup, from ``model_dir``,
    else from ``$FRAUD_API_MODEL_DIR``, else from ``model/``.

    ``/predict`` is limited to ``rate_limit_per_minute`` requests per client; ``/health``
    is never limited. Request bodies over ``max_body_bytes`` get a 413. Each limit comes
    from its argument, else from its environment variable, else from the default.

    Every scored Application goes to the Prediction Log at ``database_url``, else at
    ``$DATABASE_URL``, after the response is sent. With neither, logging is off.
    """
    rate_limit = protection.RateLimiter(
        _setting(rate_limit_per_minute, RATE_LIMIT_ENV, protection.RATE_LIMIT_PER_MINUTE),
        window=60.0,
    )
    max_body_bytes = _setting(max_body_bytes, MAX_BODY_BYTES_ENV, protection.MAX_BODY_BYTES)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        directory = model_dir or Path(os.environ.get(MODEL_DIR_ENV, model_version.MODEL_DIR))
        model = model_version.load(directory)
        # The API validates against the code's contract; a Model Version built on another
        # contract would fail on Applications the API accepted.
        if model.metadata["features"] != FIELDS or model.metadata["categories"] != CATEGORIES:
            raise RuntimeError(f"Model Version {model.id} does not match the API contract")
        app.state.model = model
        prediction_log = PredictionLog(database_url or os.environ.get(DATABASE_URL_ENV))
        prediction_log.start()
        app.state.prediction_log = prediction_log
        yield
        prediction_log.close()

    app = FastAPI(
        title="Fraud API",
        summary="Scores bank account opening Applications for fraud risk.",
        lifespan=lifespan,
    )
    app.add_middleware(protection.BodySizeLimit, max_bytes=max_body_bytes)
    app.add_middleware(_ReceivedAt)  # outermost, so latency covers the whole request

    @app.exception_handler(RequestValidationError)
    async def invalid_application(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422, content={"detail": jsonable_encoder(_encodable(exc.errors()))}
        )

    @app.post("/predict", dependencies=[Depends(rate_limit)])
    def predict(
        application: ApplicationIn, request: Request, background: BackgroundTasks
    ) -> Scored:
        model: model_version.ModelVersion = request.app.state.model
        fraud_score = float(model.score(to_frame([application]))[0])
        submitted = application.model_dump()
        [factors] = model.top_factors([submitted])
        scored = Scored(
            fraud_score=fraud_score,
            decision="review" if fraud_score >= model.threshold else "pass",
            model_version=model.id,
            request_id=uuid.uuid4(),
            top_factors=factors,
        )
        prediction_log: PredictionLog = request.app.state.prediction_log
        if prediction_log.enabled:
            # Queued after the response has been sent; a writer thread does the rest.
            background.add_task(
                prediction_log.submit,
                Prediction(
                    request_id=scored.request_id,
                    created_at=datetime.now(UTC),
                    model_version=scored.model_version,
                    application=submitted,
                    fraud_score=scored.fraud_score,
                    decision=scored.decision,
                    top_factors=factors,
                    latency_ms=(time.perf_counter() - request.state.received_at) * 1000,
                ),
            )
        return scored

    default_openapi = app.openapi

    def openapi_with_an_example_response() -> dict[str, Any]:
        """The schema, with the /docs example Application's real answer as the example
        response (from the Model Version's reference Applications)."""
        if app.openapi_schema is None:
            schema = default_openapi()
            model: model_version.ModelVersion = app.state.model
            reference = next(
                (
                    r
                    for r in model.metadata["reference_applications"]
                    if r["application"] == EXAMPLE_APPLICATION
                ),
                None,
            )
            if reference is not None and "top_factors" in reference:
                schema["components"]["schemas"]["Scored"]["examples"] = [
                    {
                        "fraud_score": reference["fraud_score"],
                        "decision": "review"
                        if reference["fraud_score"] >= model.threshold
                        else "pass",
                        "model_version": model.id,
                        "request_id": "3f2b8c1e-0d4a-4e8b-9f6a-2c7d5e1b9a04",
                        "top_factors": reference["top_factors"],
                    }
                ]
        return app.openapi_schema or default_openapi()

    app.openapi = openapi_with_an_example_response  # type: ignore[method-assign]

    @app.get("/health")
    def health(request: Request) -> Health:
        return Health(status="ok", model_version=request.app.state.model.id)

    return app


class _ReceivedAt:
    """Notes when each request reached the service, for the Prediction Log's latency."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            scope.setdefault("state", {})["received_at"] = time.perf_counter()
        await self.app(scope, receive, send)


def _setting(value: int | None, env: str, default: int) -> int:
    if value is not None:
        return value
    return int(os.environ.get(env) or default)


def _encodable(errors: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validation errors that JSON can carry. The default handler echoes each rejected
    value, and a NaN or Infinity (which the JSON parser lets through) can't be encoded
    back, which turned a 422 into a 500. Exceptions in ``ctx`` become their message."""
    encodable = []
    for error in errors:
        error = dict(error)
        if isinstance(error.get("input"), float) and not math.isfinite(error["input"]):
            error["input"] = str(error["input"])
        if "ctx" in error:
            error["ctx"] = {
                k: str(v) if isinstance(v, Exception) else v for k, v in error["ctx"].items()
            }
        encodable.append(error)
    return encodable


app = create_app()
"""For ``uvicorn fraud_api.service:app``."""
