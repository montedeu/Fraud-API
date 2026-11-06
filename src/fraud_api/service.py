"""The scoring service: POST an Application, get a Fraud Score and a Decision."""

import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from fraud_api import model_version
from fraud_api.application import ApplicationIn, to_frame

MODEL_DIR_ENV = "FRAUD_API_MODEL_DIR"


class Scored(BaseModel):
    fraud_score: float
    decision: Literal["pass", "review"]
    model_version: str
    request_id: uuid.UUID


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str


def create_app(model_dir: Path | None = None) -> FastAPI:
    """The FastAPI app. The Model Version is loaded once, at startup, from ``model_dir``,
    else from ``$FRAUD_API_MODEL_DIR``, else from ``model/``."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        directory = model_dir or Path(os.environ.get(MODEL_DIR_ENV, model_version.MODEL_DIR))
        app.state.model = model_version.load(directory)
        yield

    app = FastAPI(
        title="Fraud API",
        summary="Scores bank account opening Applications for fraud risk.",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def invalid_application(request: Request, exc: RequestValidationError) -> JSONResponse:
        # The default handler echoes each rejected value back, and a NaN or Infinity the
        # JSON parser let through can't be encoded again, which turned a 422 into a 500.
        errors = [{k: v for k, v in e.items() if k != "input"} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})

    @app.post("/predict")
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


app = create_app()
"""For ``uvicorn fraud_api.service:app``."""
