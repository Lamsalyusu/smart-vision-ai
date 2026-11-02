import os
import tempfile
import unittest
from typing import Any

import numpy as np

from core.application.inference import (
    GpuDevice,
    GpuScheduler,
    TensorRtArtifact,
    TritonHttpClient,
    TritonInferenceError,
    TritonModelSpec,
    triton_datatype,
)


class TritonClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.requests: list[tuple[str, dict, float]] = []

    def transport(self, url: str, payload, timeout: float):
        self.requests.append((url, dict(payload), timeout))
        return {
            "outputs": [
                {
                    "name": "boxes",
                    "datatype": "FP32",
                    "shape": [1, 2, 4],
                    "data": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                }
            ]
        }

    def client(self, **overrides: Any) -> TritonHttpClient:
        values: dict[str, Any] = {"name": "detector", "version": "3", "max_batch_size": 4}
        values.update(overrides)
        return TritonHttpClient(
            "http://triton:8000",
            {"detector": TritonModelSpec(**values)},
            transport=self.transport,
        )

    def test_posts_typed_tensors_and_parses_outputs(self) -> None:
        outputs = self.client().infer(
            "detector", {"images": np.zeros((1, 3, 2, 2), dtype=np.float32)}
        )

        url, payload, timeout = self.requests[0]
        self.assertEqual(url, "http://triton:8000/v2/models/detector/versions/3/infer")
        self.assertAlmostEqual(timeout, 2.0)
        self.assertEqual(payload["inputs"][0]["datatype"], "FP32")
        self.assertEqual(payload["inputs"][0]["shape"], [1, 3, 2, 2])
        self.assertEqual(len(payload["inputs"][0]["data"]), 12)
        self.assertEqual(outputs["boxes"].shape, (1, 2, 4))
        self.assertEqual(float(outputs["boxes"][0, 1, 3]), 7.0)

    def test_rejects_batches_above_the_model_limit(self) -> None:
        with self.assertRaisesRegex(TritonInferenceError, "exceeds configured maximum"):
            self.client().infer("detector", {"images": np.zeros((5, 3), dtype=np.float32)})

    def test_rejects_unknown_models_and_empty_inputs(self) -> None:
        with self.assertRaisesRegex(TritonInferenceError, "not configured"):
            self.client().infer("other", {"images": np.zeros((1, 3))})
        with self.assertRaisesRegex(TritonInferenceError, "at least one input"):
            self.client().infer("detector", {})

    def test_surfaces_server_and_shape_errors(self) -> None:
        client = TritonHttpClient(
            "http://triton:8000",
            {"detector": TritonModelSpec("detector")},
            transport=lambda *_args: {"error": "model unavailable"},
        )
        with self.assertRaisesRegex(TritonInferenceError, "model unavailable"):
            client.infer("detector", {"images": np.zeros((1, 3))})

        client = TritonHttpClient(
            "http://triton:8000",
            {"detector": TritonModelSpec("detector")},
            transport=lambda *_args: {
                "outputs": [
                    {"name": "boxes", "datatype": "FP32", "shape": [2, 2], "data": [1.0]}
                ]
            },
        )
        with self.assertRaisesRegex(TritonInferenceError, "element count"):
            client.infer("detector", {"images": np.zeros((1, 3))})

    def test_datatype_mapping(self) -> None:
        self.assertEqual(triton_datatype(np.zeros(1, dtype=np.float32)), "FP32")
        self.assertEqual(triton_datatype(np.zeros(1, dtype=np.int64)), "INT64")
        self.assertEqual(triton_datatype(np.zeros(1, dtype=np.uint8)), "UINT8")
        with self.assertRaises(TritonInferenceError):
            triton_datatype(np.zeros(1, dtype=np.complex64))


class TensorRtArtifactTests(unittest.TestCase):
    def test_structural_validation(self) -> None:
        with self.assertRaises(ValueError):
            TensorRtArtifact("")
        with self.assertRaises(ValueError):
            TensorRtArtifact("model.plan", precision="bf16")
        with self.assertRaises(ValueError):
            TensorRtArtifact("model.plan", device_id=-1)
        with self.assertRaises(ValueError):
            TensorRtArtifact("model.plan", max_batch_size=0)

    def test_verifies_engine_file_exists_and_is_not_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "detector.plan")
            with open(path, "wb") as handle:
                handle.write(b"engine")
            TensorRtArtifact(path).verify_engine_file()

            with self.assertRaises(FileNotFoundError):
                TensorRtArtifact(os.path.join(directory, "missing.plan")).verify_engine_file()

            empty = os.path.join(directory, "empty.plan")
            open(empty, "wb").close()
            with self.assertRaises(ValueError):
                TensorRtArtifact(empty).verify_engine_file()


class GpuSchedulerTests(unittest.TestCase):
    def test_routes_to_least_loaded_device_and_honors_preference(self) -> None:
        scheduler = GpuScheduler()
        scheduler.register(GpuDevice(0, "A100", 40960, max_inflight=2))
        scheduler.register(GpuDevice(1, "A100", 40960, max_inflight=2))

        first = scheduler.acquire()
        assert first is not None
        self.assertEqual(first.device.device_id, 0)
        second = scheduler.acquire()
        assert second is not None
        self.assertEqual(second.device.device_id, 1)
        third = scheduler.acquire()
        assert third is not None
        self.assertEqual(third.device.device_id, 0)
        fourth = scheduler.acquire()
        assert fourth is not None
        self.assertEqual(fourth.device.device_id, 1)
        self.assertIsNone(scheduler.acquire())

        self.assertTrue(scheduler.release(second))
        self.assertFalse(scheduler.release(second))
        preferred = scheduler.acquire(preferred_device_id=1)
        assert preferred is not None
        self.assertEqual(preferred.device.device_id, 1)

    def test_skips_unhealthy_devices_and_rejects_duplicate_registration(self) -> None:
        scheduler = GpuScheduler()
        scheduler.register(GpuDevice(0, "L4", 24000))
        scheduler.register(GpuDevice(1, "L4", 24000))
        scheduler.mark_unhealthy(0)

        lease = scheduler.acquire()
        assert lease is not None
        self.assertEqual(lease.device.device_id, 1)

        with self.assertRaises(ValueError):
            scheduler.register(GpuDevice(1, "L4", 24000))
        with self.assertRaises(ValueError):
            scheduler.mark_unhealthy(9)


if __name__ == "__main__":
    unittest.main()
