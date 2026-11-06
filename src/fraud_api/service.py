"""The scoring service: POST an Application, get a Fraud Score and a Decision."""

import math
import os
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from fraud_api import model_version, protection
from fraud_api.application import CATEGORIES, FIELDS, ApplicationIn, to_frame

MODEL_DIR_ENV = "FRAUD_API_MODEL_DIR"
RATE_LIMIT_ENV = "FRAUD_API_RATE_LIMIT_PER_MINUTE"
MAX_BODY_BYTES_ENV = "FRAUD_API_MAX_BODY_BYTES"


class Scored(BaseModel):
    fraud_score: float
    decision: Literal["pass", "review"]
    model_version: str
    request_id: uuid.UUID


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str


def create_app(
    model_dir: Path | None = None,
    *,
    rate_limit_per_minute: int | None = None,
    max_body_bytes: int | None = None,
) -> FastAPI:
    """The FastAPI app. The Model Version is loaded once, at startup, from ``model_dir``,
    else from ``$FRAUD_API_MODEL_DIR``, else from ``model/``.

    ``/predict`` is limited to ``rate_limit_per_minute`` requests per client; ``/health``
    is never limited. Request bodies over ``max_body_bytes`` get a 413. Each limit comes
    from its argument, else from its environment variable, else from the default.
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
        yield

    app = FastAPI(
        title="Fraud API",
        summary="Scores bank account opening Applications for fraud risk.",
        lifespan=lifespan,
    )
    app.add_middleware(protection.BodySizeLimit, max_bytes=max_body_bytes)

    @app.exception_handler(RequestValidationError)
    async def invalid_application(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422, content={"detail": jsonable_encoder(_encodable(exc.errors()))}
        )

    @app.post("/predict", dependencies=[Depends(rate_limit)])
    def predict(application: ApplicationIn, request: Request) -> Scored:
        model: model_version.ModelVersion = request.app.state.model
        fraud_score = float(model.score(to_frame([application]))[0])
        return Scored(
            fraud_score=fraud_score,
            decision="review" if fraud_score >= model.threshold else "pass",
            model_version=model.id,
            request_id=uuid.uuid4(),
        )

    @app.get("/health")
    def health(request: Request) -> Health:
        return Health(status="ok", model_version=request.app.state.model.id)

    return app


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
