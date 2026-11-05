"""In-process sliding-window rate limiting for the FastAPI service.

Limits are keyed by the client address the Next.js server forwards in the
``X-Client-IP`` header. FastAPI only accepts requests carrying the internal
service token, so that header comes from our own server, not the browser.

State lives in memory, so each service process counts separately. That suits
the single-process deployment in docker-compose; a multi-replica deployment
should move the counters to a shared store such as Redis.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Annotated

from fastapi import Header, HTTPException


class SlidingWindowLimiter:
    def __init__(self) -> None:
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, bucket: str, key: str, limit: int, window_seconds: float) -> float | None:
        """Record a request; return seconds to wait if it exceeds the limit, else None."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits[(bucket, key)]
            while hits and hits[0] <= now - window_seconds:
                hits.popleft()
            if len(hits) >= limit:
                return max(0.0, hits[0] + window_seconds - now)
            hits.append(now)
            return None

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = SlidingWindowLimiter()


def rate_limit(bucket: str, limit: int, window_seconds: float) -> Callable[..., None]:
    """FastAPI dependency allowing ``limit`` requests per client per window.

    Set RATE_LIMITS_ENABLED=false to switch every limit off (for example in load tests).
    """

    def dependency(x_client_ip: Annotated[str | None, Header()] = None) -> None:
        if os.getenv("RATE_LIMITS_ENABLED", "true").lower() == "false":
            return
        retry_after = limiter.hit(bucket, (x_client_ip or "unknown").strip()[:64], limit, window_seconds)
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please wait and try again.",
                headers={"Retry-After": str(max(1, int(retry_after) + 1))},
            )

    return dependency
