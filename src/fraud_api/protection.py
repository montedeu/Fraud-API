"""Protection for the public, unauthenticated API: a per-client rate limit and a cap on
request body size."""

import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
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

    Kept in memory, which is enough for a service capped at one instance. At most
    ``max_clients`` are tracked; past that, the client seen longest ago is forgotten.
    """

    def __init__(
        self,
        limit: int = RATE_LIMIT_PER_MINUTE,
        window: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        max_clients: int = 10_000,
    ) -> None:
        if limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        self.limit = limit
        self.window = window
        self.max_clients = max_clients
        self._clock = clock
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
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
        with self._lock:
            # Read inside the lock, so each client's timestamps stay in order.
            now = self._clock()
            hits = self._hits.get(key)
            if hits is None:
                hits = self._hits[key] = deque()
                if len(self._hits) > self.max_clients:
                    self._hits.popitem(last=False)
            else:
                self._hits.move_to_end(key)
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return hits[0] + self.window - now
            hits.append(now)
            return None


class BodySizeLimit:
    """ASGI middleware: a 413 for any body over ``max_bytes``, before the app sees it.

    A declared ``Content-Length`` over the limit is refused without reading the body. A
    body sent without one is read here up to the limit, then handed on unchanged.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None:
            if not declared.isdigit() or int(declared) > self.max_bytes:
                await self._too_large(scope, receive, send)
                return
            await self.app(scope, receive, send)
            return

        body = b""
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            body += message.get("body", b"")
            if len(body) > self.max_bytes:
                await self._too_large(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay, send)

    async def _too_large(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            {"detail": f"Request body over {self.max_bytes} bytes"}, status_code=413
        )
        await response(scope, receive, send)
