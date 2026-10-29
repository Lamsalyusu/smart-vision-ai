import unittest

from core.application.ingestion import (
    CameraStream,
    CameraStreamRegistry,
    FrameAdmissionScheduler,
    FrameIngressCoordinator,
    IncomingFrame,
    VideoCodec,
)
from core.infrastructure.shared_memory import FramePixelFormat, SharedFrameRingBuffer


class FrameIngressCoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = CameraStreamRegistry(maximum_streams=2)
        self.registry.register(CameraStream("camera-1", VideoCodec.H264, 2, 2, 500_000))
        self.scheduler = FrameAdmissionScheduler(capacity=1, refill_rate_per_second=0)
        self.ring = SharedFrameRingBuffer(capacity=1, max_frame_bytes=32)
        self.ingress = FrameIngressCoordinator(self.registry, self.scheduler, self.ring)

    def tearDown(self) -> None:
        self.ring.close()
        self.ring.unlink()

    @staticmethod
    def frame(sequence: int, keyframe: bool = False) -> IncomingFrame:
        return IncomingFrame(
            "camera-1",
            sequence,
            2,
            2,
            FramePixelFormat.RGB24,
            sequence * 1_000_000,
            bytes([sequence]) * 12,
            keyframe,
        )

    def test_registered_frame_flows_into_shared_ring(self) -> None:
        result = self.ingress.submit(self.frame(1))

        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "enqueued")
        with self.ring.read_frame() as lease:
            self.assertEqual(lease.descriptor.sequence_number, 1)

    def test_unregistered_camera_is_rejected(self) -> None:
        frame = IncomingFrame(
            "unknown", 1, 2, 2, FramePixelFormat.RGB24, 1, bytes(12)
        )

        with self.assertRaisesRegex(KeyError, "not registered"):
            self.ingress.submit(frame)

    def test_rate_limited_frame_signals_tracking_should_advance(self) -> None:
        self.ingress.submit(self.frame(1))

        result = self.ingress.submit(self.frame(2))

        self.assertFalse(result.accepted)
        self.assertTrue(result.advance_tracking)
        self.assertEqual(result.reason, "rate_limited")

    def test_keyframe_bypasses_rate_limit_but_ring_full_still_drops(self) -> None:
        self.ingress.submit(self.frame(1))

        with self.ring.read_frame():
            self.ingress.submit(self.frame(2))
            result = self.ingress.submit(self.frame(3, keyframe=True))

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "ring_full")

    def test_keyframe_bypasses_rate_limit_when_ring_is_available(self) -> None:
        self.ingress.submit(self.frame(1))
        with self.ring.read_frame():
            pass

        result = self.ingress.submit(self.frame(2, keyframe=True))

        self.assertTrue(result.accepted)
        with self.ring.read_frame() as lease:
            self.assertEqual(lease.descriptor.sequence_number, 2)


if __name__ == "__main__":
    unittest.main()