from collections.abc import Callable
from enum import Enum
import math
import random
import threading
import time
from typing import TypeVar


Result = TypeVar("Result")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    def __init__(self, retry_after_seconds: float) -> None:
        self.retry_after_seconds = max(0.0, retry_after_seconds)
        super().__init__(f"circuit is open; retry after {self.retry_after_seconds:.3f}s")


class CircuitBreaker:
    """Thread-safe circuit breaker with a single half-open recovery probe."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 1.0,
        max_recovery_timeout_seconds: float = 30.0,
        jitter_ratio: float = 0.2,
        *,
        clock: Callable[[], float] = time.monotonic,
        jitter_source: Callable[[float, float], float] = random.uniform,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure threshold must be at least one")
        if (
            not math.isfinite(recovery_timeout_seconds)
            or recovery_timeout_seconds <= 0
            or not math.isfinite(max_recovery_timeout_seconds)
            or max_recovery_timeout_seconds < recovery_timeout_seconds
        ):
            raise ValueError("recovery timeouts must be positive and maximum >= initial")
        if not math.isfinite(jitter_ratio) or not 0 <= jitter_ratio <= 1:
            raise ValueError("jitter ratio must be between zero and one")

        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.max_recovery_timeout_seconds = max_recovery_timeout_seconds
        self.jitter_ratio = jitter_ratio
        self._clock = clock
        self._jitter_source = jitter_source
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._open_until = 0.0
        self._probe_in_flight = False
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state is CircuitState.OPEN and self._clock() >= self._open_until:
                self._state = CircuitState.HALF_OPEN
            return self._state

    def call(
        self,
        operation: Callable[[], Result],
        fallback: Callable[[CircuitOpenError], Result] | None = None,
    ) -> Result:
        try:
            self._acquire_permission()
        except CircuitOpenError as error:
            if fallback is not None:
                return fallback(error)
            raise

        try:
            result = operation()
        except Exception:
            self._record_failure()
            raise
        self._record_success()
        return result

    def _acquire_permission(self) -> None:
        now = self._clock()
        with self._lock:
            if self._state is CircuitState.OPEN:
                if now < self._open_until:
                    raise CircuitOpenError(self._open_until - now)
                self._state = CircuitState.HALF_OPEN

            if self._state is CircuitState.HALF_OPEN:
                if self._probe_in_flight:
                    raise CircuitOpenError(self._open_until - now)
                self._probe_in_flight = True

    def _record_success(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._open_until = 0.0
            self._probe_in_flight = False

    def _record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._probe_in_flight = False
            if (
                self._state is CircuitState.HALF_OPEN
                or self._failure_count >= self.failure_threshold
            ):
                self._open_until = self._clock() + self._next_recovery_delay()
                self._state = CircuitState.OPEN

    def _next_recovery_delay(self) -> float:
        exponent = max(0, self._failure_count - self.failure_threshold)
        maximum_exponent = math.ceil(
            math.log2(self.max_recovery_timeout_seconds / self.recovery_timeout_seconds)
        )
        base_delay = min(
            self.max_recovery_timeout_seconds,
            self.recovery_timeout_seconds * (2 ** min(exponent, maximum_exponent)),
        )
        jitter_limit = base_delay * self.jitter_ratio
        jitter = self._jitter_source(-jitter_limit, jitter_limit)
        return min(self.max_recovery_timeout_seconds, max(0.0, base_delay + jitter))