import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

from core.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
)

logger = logging.getLogger(__name__)

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class BackendTier(str, Enum):
    PRIMARY_GPU = "primary_gpu"
    SECONDARY_ONNX = "secondary_onnx"
    FALLBACK_CPU = "fallback_cpu"


@dataclass(frozen=True, slots=True)
class InferenceExecutionResult:
    output: object
    tier_used: BackendTier
    fallback_occurred: bool


class ResilientInferencePipeline:
    """Multi-tier inference execution supervisor with automated circuit breaker cascade.

    Cascades execution through:
      PRIMARY_GPU -> SECONDARY_ONNX -> FALLBACK_CPU
    """

    def __init__(
        self,
        primary_fn: Callable[[InputType], OutputType],
        secondary_fn: Callable[[InputType], OutputType],
        fallback_fn: Callable[[InputType], OutputType],
        *,
        primary_breaker: CircuitBreaker | None = None,
        secondary_breaker: CircuitBreaker | None = None,
    ) -> None:
        self._primary_fn = primary_fn
        self._secondary_fn = secondary_fn
        self._fallback_fn = fallback_fn

        self._primary_breaker = primary_breaker or CircuitBreaker(
            failure_threshold=3, recovery_timeout_seconds=2.0
        )
        self._secondary_breaker = secondary_breaker or CircuitBreaker(
            failure_threshold=5, recovery_timeout_seconds=5.0
        )

        self._executions_count = 0
        self._fallback_count = 0

    @property
    def total_executions(self) -> int:
        return self._executions_count

    @property
    def fallback_count(self) -> int:
        return self._fallback_count

    def execute(self, payload: InputType) -> InferenceExecutionResult:
        """Execute inference payload through the resilient cascading pipeline."""
        self._executions_count += 1

        # 1. Try Primary GPU Tier
        try:
            result = self._primary_breaker.call(lambda: self._primary_fn(payload))
            return InferenceExecutionResult(
                output=result,
                tier_used=BackendTier.PRIMARY_GPU,
                fallback_occurred=False,
            )
        except (CircuitOpenError, Exception) as exc:  # noqa: BLE001
            logger.warning(
                "Primary GPU inference failed or circuit open (%s). Falling back to ONNX.",
                exc,
            )

        # 2. Try Secondary ONNX CPU Tier
        self._fallback_count += 1
        try:
            result = self._secondary_breaker.call(lambda: self._secondary_fn(payload))
            return InferenceExecutionResult(
                output=result,
                tier_used=BackendTier.SECONDARY_ONNX,
                fallback_occurred=True,
            )
        except (CircuitOpenError, Exception) as exc:  # noqa: BLE001
            logger.warning(
                "Secondary ONNX inference failed or circuit open (%s). Cascading to Fallback CPU.",
                exc,
            )

        # 3. Final Fallback CPU / Heuristic Tier (guaranteed to succeed or degrade cleanly)
        result = self._fallback_fn(payload)
        return InferenceExecutionResult(
            output=result,
            tier_used=BackendTier.FALLBACK_CPU,
            fallback_occurred=True,
        )
