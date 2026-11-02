import unittest

import cv2
import numpy as np

from core.application.simulation import (
    CameraStreamPlan,
    ChaosPolicy,
    StreamingLoadHarness,
    SyntheticCameraFarm,
)


class SyntheticCameraFarmTests(unittest.TestCase):
    def test_frames_are_deterministic_valid_jpegs_with_ground_truth(self) -> None:
        farm = SyntheticCameraFarm(["cam-a", "cam-b"], width=96, height=64)
        first = farm.frame("cam-a", 3, 0.1)
        second = farm.frame("cam-a", 3, 0.1)

        self.assertEqual(first.jpeg_bytes, second.jpeg_bytes)
        decoded = cv2.imdecode(np.frombuffer(first.jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        self.assertEqual(decoded.shape, (64, 96, 3))
        self.assertEqual(len(first.detections), 1)
        x1, y1, x2, y2 = first.detections[0].bbox
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, 96)
        self.assertLessEqual(y2, 64)
        self.assertEqual(first.detections[0].class_name, "person")

    def test_generates_a_timed_sequence(self) -> None:
        farm = SyntheticCameraFarm(["cam-a"], width=48, height=48)
        frames = list(farm.frames("cam-a", 3, start_seconds=1.0, fps=2))
        self.assertEqual([frame.sequence for frame in frames], [0, 1, 2])
        self.assertEqual([frame.captured_at_seconds for frame in frames], [1.0, 1.5, 2.0])

    def test_validates_configuration_and_inputs(self) -> None:
        with self.assertRaises(ValueError):
            SyntheticCameraFarm([])
        with self.assertRaises(ValueError):
            SyntheticCameraFarm(["cam-a", "cam-a"])
        with self.assertRaises(ValueError):
            SyntheticCameraFarm(["cam-a"], width=8)
        farm = SyntheticCameraFarm(["cam-a"])
        with self.assertRaises(ValueError):
            farm.frame("missing", 0)
        with self.assertRaises(ValueError):
            farm.frame("cam-a", -1)


class StreamingLoadHarnessTests(unittest.TestCase):
    def test_runs_more_than_fifty_cameras_and_reports_percentiles(self) -> None:
        farm = SyntheticCameraFarm([f"cam-{index:02d}" for index in range(64)],
                                   width=48, height=48)
        processed: list[str] = []

        def processor(frame) -> None:
            self.assertTrue(frame.jpeg_bytes)
            self.assertEqual(len(frame.detections), 1)
            processed.append(frame.camera_id)

        plans = [
            CameraStreamPlan(camera_id, frames=6, target_fps=60)
            for camera_id in farm.camera_ids
        ]
        report = StreamingLoadHarness(processor, farm=farm).run(plans)

        self.assertEqual(report.cameras, 64)
        self.assertEqual(report.frames_attempted, 384)
        self.assertEqual(report.frames_processed, 384)
        self.assertEqual(len(processed), 384)
        self.assertGreater(report.throughput_fps, 0)
        self.assertGreaterEqual(report.latency_p99_ms, report.latency_p95_ms)
        self.assertGreaterEqual(report.latency_p95_ms, report.latency_p50_ms)
        self.assertGreaterEqual(report.latency_max_ms, report.latency_p99_ms)
        self.assertEqual(
            report.frames_attempted,
            report.frames_processed
            + report.frames_dropped
            + report.disconnects
            + report.processing_errors,
        )

    def test_injects_drops_disconnects_errors_and_spikes(self) -> None:
        farm = SyntheticCameraFarm(["cam-a"], width=48, height=48)

        def processor(frame) -> None:
            if frame.sequence == 0:
                raise RuntimeError("simulated detector failure")

        policy = ChaosPolicy(
            drop_every=2,
            disconnect_every=5,
            error_every=7,
            latency_spike_every=11,
            latency_spike_seconds=0.0,
        )
        report = StreamingLoadHarness(processor, farm=farm).run(
            [CameraStreamPlan("cam-a", frames=30)], chaos=policy
        )

        self.assertEqual(report.frames_attempted, 30)
        self.assertGreater(report.frames_dropped, 0)
        self.assertGreater(report.disconnects, 0)
        self.assertGreater(report.processing_errors, 0)
        self.assertEqual(
            report.frames_attempted,
            report.frames_processed
            + report.frames_dropped
            + report.disconnects
            + report.processing_errors,
        )
        payload = report.to_dict()
        self.assertEqual(payload["cameras"], 1)
        self.assertIn("latency_p99_ms", payload)

    def test_policy_action_precedence_and_validation(self) -> None:
        policy = ChaosPolicy(drop_every=2, disconnect_every=2, error_every=2)
        self.assertEqual(policy.action_for(1), "error")
        self.assertEqual(ChaosPolicy(drop_every=2).action_for(1), "drop")
        self.assertEqual(ChaosPolicy().action_for(5), "process")

        with self.assertRaises(ValueError):
            ChaosPolicy(drop_every=-1)
        with self.assertRaises(ValueError):
            ChaosPolicy(latency_spike_seconds=float("inf"))
        with self.assertRaises(ValueError):
            CameraStreamPlan("cam-a", -1)
        with self.assertRaises(ValueError):
            CameraStreamPlan("cam-a", 1, target_fps=0)

    def test_rejects_duplicate_plans_and_missing_farm_cameras(self) -> None:
        farm = SyntheticCameraFarm(["cam-a"], width=48, height=48)
        harness = StreamingLoadHarness(lambda _frame: None, farm=farm)
        with self.assertRaises(ValueError):
            harness.run([])
        with self.assertRaises(ValueError):
            harness.run([CameraStreamPlan("cam-a", 1), CameraStreamPlan("cam-a", 1)])
        with self.assertRaises(ValueError):
            harness.run([CameraStreamPlan("cam-z", 1)])


if __name__ == "__main__":
    unittest.main()
