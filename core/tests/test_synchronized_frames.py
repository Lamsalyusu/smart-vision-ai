import unittest
from collections.abc import Mapping, Sequence

from core.application.ingestion.synchronized_frames import (
    CameraFrameBatch,
    SynchronizedFrameBridge,
)
from core.domain.spatial.multicam import GlobalTrackMatch, MultiCameraObservation


class RecordingFusion:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, tuple[MultiCameraObservation, ...]]]] = []

    def fuse_synchronized_frame(
        self,
        frame_sequence: int,
        observations_by_camera: Mapping[str, Sequence[MultiCameraObservation]],
    ) -> tuple[GlobalTrackMatch, ...]:
        grouped = {
            camera_id: tuple(observations)
            for camera_id, observations in observations_by_camera.items()
        }
        self.calls.append((frame_sequence, grouped))
        return ()


def batch(camera_id: str, captured_at: float, track_id: int = 1) -> CameraFrameBatch:
    return CameraFrameBatch(
        camera_id,
        captured_at,
        (MultiCameraObservation(camera_id, track_id, (10.0, 20.0)),),
    )


class SynchronizedFrameBridgeTests(unittest.TestCase):
    def test_emits_one_batch_when_all_cameras_align(self) -> None:
        fusion = RecordingFusion()
        bridge = SynchronizedFrameBridge(
            fusion, ["cam-a", "cam-b", "cam-c"], tolerance_seconds=0.05
        )

        self.assertEqual(bridge.submit(batch("cam-a", 10.0)), ())
        self.assertEqual(bridge.submit(batch("cam-b", 10.02)), ())
        bridge.submit(batch("cam-c", 10.03))

        self.assertEqual(len(fusion.calls), 1)
        sequence, grouped = fusion.calls[0]
        self.assertEqual(sequence, 1)
        self.assertEqual(set(grouped), {"cam-a", "cam-b", "cam-c"})
        metrics = bridge.metrics()
        self.assertEqual(metrics.batches_emitted, 1)
        self.assertEqual(metrics.partial_batches, 0)
        self.assertEqual(metrics.frames_dropped_late, 0)

    def test_waits_for_a_late_camera_and_drops_frames_older_than_emitted(self) -> None:
        fusion = RecordingFusion()
        bridge = SynchronizedFrameBridge(fusion, ["cam-a", "cam-b"], tolerance_seconds=0.05)

        bridge.submit(batch("cam-a", 10.0))
        bridge.submit(batch("cam-a", 10.01))
        self.assertEqual(fusion.calls, [])

        bridge.submit(batch("cam-b", 10.02))
        self.assertEqual(len(fusion.calls), 1)
        # The bridge selected the closest pair and discarded the superseded frame.
        self.assertEqual(bridge.metrics().frames_dropped_late, 1)

        self.assertEqual(bridge.submit(batch("cam-a", 9.5)), ())
        self.assertEqual(bridge.metrics().frames_dropped_late, 2)
        self.assertEqual(len(fusion.calls), 1)

    def test_flush_emits_partial_batch_when_a_camera_stalls(self) -> None:
        fusion = RecordingFusion()
        bridge = SynchronizedFrameBridge(
            fusion,
            ["cam-a", "cam-b"],
            tolerance_seconds=0.05,
            stall_timeout_seconds=0.25,
        )

        bridge.submit(batch("cam-a", 10.0))
        self.assertEqual(fusion.calls, [])

        bridge.flush_stalled(now_seconds=10.4)

        self.assertEqual(len(fusion.calls), 1)
        self.assertEqual(set(fusion.calls[0][1]), {"cam-a"})
        self.assertEqual(bridge.metrics().partial_batches, 1)

    def test_stalled_camera_is_flushed_automatically_on_next_submission(self) -> None:
        fusion = RecordingFusion()
        bridge = SynchronizedFrameBridge(
            fusion,
            ["cam-a", "cam-b"],
            tolerance_seconds=0.05,
            stall_timeout_seconds=0.25,
        )
        bridge.submit(batch("cam-b", 10.0))
        bridge.submit(batch("cam-a", 10.3))

        self.assertEqual(len(fusion.calls), 1)
        self.assertEqual(fusion.calls[0][1].keys(), {"cam-a", "cam-b"})

        bridge.submit(batch("cam-a", 10.6), now_seconds=10.6)
        self.assertEqual(len(fusion.calls), 2)
        self.assertEqual(set(fusion.calls[1][1]), {"cam-a"})

    def test_rejects_unknown_cameras_mismatched_observations_and_bad_clocks(self) -> None:
        bridge = SynchronizedFrameBridge(RecordingFusion(), ["cam-a"])

        with self.assertRaisesRegex(ValueError, "not expected"):
            bridge.submit(batch("cam-x", 10.0))

        mismatched = CameraFrameBatch(
            "cam-a", 10.0, (MultiCameraObservation("cam-b", 1, (0.0, 0.0)),)
        )
        with self.assertRaisesRegex(ValueError, "observation camera"):
            bridge.submit(mismatched)

        with self.assertRaisesRegex(ValueError, "timestamp"):
            bridge.submit(batch("cam-a", float("nan")))

    def test_validates_configuration(self) -> None:
        with self.assertRaises(ValueError):
            SynchronizedFrameBridge(RecordingFusion(), [])
        with self.assertRaises(ValueError):
            SynchronizedFrameBridge(RecordingFusion(), ["cam-a", "cam-a"])
        with self.assertRaises(ValueError):
            SynchronizedFrameBridge(RecordingFusion(), ["cam-a"], tolerance_seconds=-1)
        with self.assertRaises(ValueError):
            SynchronizedFrameBridge(RecordingFusion(), ["cam-a"], stall_timeout_seconds=0)


if __name__ == "__main__":
    unittest.main()
