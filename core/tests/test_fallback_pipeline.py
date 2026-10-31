import unittest

from core.infrastructure.resilience.circuit_breaker import CircuitBreaker
from core.infrastructure.resilience.fallback_engine import (
    BackendTier,
    ResilientInferencePipeline,
)


class ResilientFallbackPipelineTests(unittest.TestCase):
    def test_primary_succeeds_without_fallback(self) -> None:
        pipeline = ResilientInferencePipeline(
            primary_fn=lambda x: f"gpu_{x}",
            secondary_fn=lambda x: f"onnx_{x}",
            fallback_fn=lambda x: f"cpu_{x}",
        )
        res = pipeline.execute("frame_0")

        self.assertEqual(res.output, "gpu_frame_0")
        self.assertEqual(res.tier_used, BackendTier.PRIMARY_GPU)
        self.assertFalse(res.fallback_occurred)
        self.assertEqual(pipeline.total_executions, 1)
        self.assertEqual(pipeline.fallback_count, 0)

    def test_primary_failure_cascades_to_secondary(self) -> None:
        def failing_gpu(_: str) -> str:
            raise RuntimeError("CUDA Out of Memory")

        pipeline = ResilientInferencePipeline(
            primary_fn=failing_gpu,
            secondary_fn=lambda x: f"onnx_{x}",
            fallback_fn=lambda x: f"cpu_{x}",
        )
        res = pipeline.execute("frame_1")

        self.assertEqual(res.output, "onnx_frame_1")
        self.assertEqual(res.tier_used, BackendTier.SECONDARY_ONNX)
        self.assertTrue(res.fallback_occurred)
        self.assertEqual(pipeline.fallback_count, 1)

    def test_secondary_failure_cascades_to_fallback_cpu(self) -> None:
        def failing_gpu(_: str) -> str:
            raise RuntimeError("Driver crash")

        def failing_onnx(_: str) -> str:
            raise RuntimeError("Thread exhaustion")

        pipeline = ResilientInferencePipeline(
            primary_fn=failing_gpu,
            secondary_fn=failing_onnx,
            fallback_fn=lambda x: f"cpu_linear_{x}",
            primary_breaker=CircuitBreaker(failure_threshold=1),
            secondary_breaker=CircuitBreaker(failure_threshold=1),
        )
        res = pipeline.execute("frame_2")

        self.assertEqual(res.output, "cpu_linear_frame_2")
        self.assertEqual(res.tier_used, BackendTier.FALLBACK_CPU)
        self.assertTrue(res.fallback_occurred)


if __name__ == "__main__":
    unittest.main()
