import threading
import unittest

from core.application.ingestion import (
    AdaptiveBitrateController,
    CameraStream,
    CameraStreamRegistry,
    VideoCodec,
)


class AdaptiveBitrateControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = AdaptiveBitrateController(
            1_000_000,
            minimum_bitrate_bps=200_000,
            maximum_bitrate_bps=1_200_000,
            additive_step_bps=100_000,
            target_latency_ms=100,
            maximum_packet_loss_ratio=0.02,
        )

    def test_congestion_multiplicatively_reduces_bitrate(self) -> None:
        decision = self.controller.observe(latency_ms=150, queue_depth=0, packet_loss_ratio=0)

        self.assertTrue(decision.congested)
        self.assertEqual(decision.target_bitrate_bps, 700_000)
        self.assertEqual(decision.reason, "latency")

    def test_healthy_observations_add_bitrate_with_upper_bound(self) -> None:
        decision = self.controller.observe(latency_ms=20, queue_depth=0, packet_loss_ratio=0)

        self.assertFalse(decision.congested)
        self.assertEqual(decision.target_bitrate_bps, 1_100_000)
        self.assertEqual(self.controller.observe(latency_ms=20, queue_depth=0, packet_loss_ratio=0).target_bitrate_bps, 1_200_000)

    def test_bitrate_never_falls_below_minimum(self) -> None:
        for _ in range(20):
            decision = self.controller.observe(latency_ms=500, queue_depth=8, packet_loss_ratio=0.5)

        self.assertEqual(decision.target_bitrate_bps, 200_000)


class CameraStreamRegistryTests(unittest.TestCase):
    @staticmethod
    def make_stream(camera_id: str) -> CameraStream:
        return CameraStream(camera_id, VideoCodec.H264, 1920, 1080, 2_000_000)

    def test_supports_fifty_independent_streams(self) -> None:
        registry = CameraStreamRegistry(maximum_streams=50)
        for index in range(50):
            registry.register(self.make_stream(f"camera-{index}"))

        self.assertEqual(registry.active_count, 50)
        self.assertEqual(registry.get("camera-37").codec, VideoCodec.H264)

    def test_rejects_stream_limit_and_duplicate_id(self) -> None:
        registry = CameraStreamRegistry(maximum_streams=1)
        registry.register(self.make_stream("camera-1"))
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(self.make_stream("camera-1"))
        with self.assertRaises(OverflowError):
            registry.register(self.make_stream("camera-2"))

    def test_updates_bitrate_and_unregisters_stream(self) -> None:
        registry = CameraStreamRegistry(maximum_streams=1)
        registry.register(self.make_stream("camera-1"))

        updated = registry.update_bitrate("camera-1", 800_000)
        removed = registry.unregister("camera-1")

        self.assertEqual(updated.target_bitrate_bps, 800_000)
        self.assertEqual(removed.camera_id, "camera-1")
        self.assertEqual(registry.active_count, 0)

    def test_concurrent_registrations_respect_capacity(self) -> None:
        registry = CameraStreamRegistry(maximum_streams=10)
        successes = []
        lock = threading.Lock()

        def register(index: int) -> None:
            try:
                registry.register(self.make_stream(f"camera-{index}"))
                with lock:
                    successes.append(index)
            except OverflowError:
                pass

        workers = [threading.Thread(target=register, args=(index,)) for index in range(30)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=2)

        self.assertEqual(len(successes), 10)
        self.assertEqual(registry.active_count, 10)


if __name__ == "__main__":
    unittest.main()