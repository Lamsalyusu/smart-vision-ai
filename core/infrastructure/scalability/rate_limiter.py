from __future__ import annotations

import collections
import time


class SlidingWindowRateLimiter:
    """Distributed Sliding-Window Rate Limiter controlling request burst and throughput."""

    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window = window_seconds
        self._buckets: collections.defaultdict[str, collections.deque[float]] = collections.defaultdict(collections.deque)

    def is_allowed(self, client_key: str, now: float | None = None) -> tuple[bool, int, float]:
        """Check whether request is within rate limits.

        Returns:
            (allowed: bool, remaining_requests: int, reset_after_seconds: float)
        """
        current_time = now if now is not None else time.monotonic()
        bucket = self._buckets[client_key]

        # Evict timestamps older than the sliding window
        while bucket and current_time - bucket[0] >= self.window:
            bucket.popleft()

        if len(bucket) < self.max_requests:
            bucket.append(current_time)
            remaining = self.max_requests - len(bucket)
            reset_after = max(0.0, self.window - (current_time - bucket[0]))
            return True, remaining, reset_after

        # Rate limit exceeded
        oldest_ts = bucket[0]
        reset_after = max(0.01, self.window - (current_time - oldest_ts))
        return False, 0, reset_after

    def reset(self, client_key: str | None = None) -> None:
        if client_key:
            self._buckets.pop(client_key, None)
        else:
            self._buckets.clear()


rate_limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60.0)
