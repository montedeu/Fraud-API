"""Protection for the public, unauthenticated API: a per-client rate limit and a cap on
request body size."""

import json
import math
import threading
import time
from collections import deque
from collections.abc import Callable

from fastapi import HTTPException, Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

RATE_LIMIT_PER_MINUTE = 60
MAX_BODY_BYTES = 4096
"""An Application is under 1 KB as JSON; anything past a few kilobytes is not one."""


def client_key(request: Request) -> str:
    """The first ``X-Forwarded-For`` hop when present (Cloud Run sets it), otherwise the
    connection's address."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded and (first := forwarded.split(",")[0].strip()):
        return first
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """At most ``limit`` requests per client in any rolling ``window`` seconds.

    Kept in memory, which is enough for a service capped at one instance.
    """

    def __init__(
        self,
        limit: int = RATE_LIMIT_PER_MINUTE,
        window: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit = limit
        self.window = window
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def __call__(self, request: Request) -> None:
        """A FastAPI dependency: raises 429 once the client is over the limit."""
        retry_after = self.hit(client_key(request))
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit is {self.limit} requests per {self.window:g} seconds",
                headers={"Retry-After": str(math.ceil(retry_after))},
            )

    def hit(self, key: str) -> float | None:
        """Record a request from ``key``. Returns ``None`` if it is allowed, else the
        seconds until the client may try again."""
        now = self._clock()
        with self._lock:
            if len(self._hits) > 10_000:
                self._forget_idle(now)
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return hits[0] + self.window - now
            hits.append(now)
            return None

    def _forget_idle(self, now: float) -> None:
        idle = [
            key for key, hits in self._hits.items() if not hits or hits[-1] <= now - self.window
        ]
        for key in idle:
            del self._hits[key]


class BodySizeLimit:
    """ASGI middleware: a 413 for any body over ``max_bytes``, before it is parsed.

    A declared ``Content-Length`` over the limit is refused without reading the body; a
    body sent without one is counted as it arrives.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > self.max_bytes):
            await self._too_large(send)
            return

        received = 0

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # FastAPI turns other errors raised while reading a body into a 400.
                    raise HTTPException(status_code=413, detail=self._detail())
            return message

        await self.app(scope, counting_receive, send)

    def _detail(self) -> str:
        return f"Request body over {self.max_bytes} bytes"

    async def _too_large(self, send: Send) -> None:
        body = json.dumps({"detail": self._detail()}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
