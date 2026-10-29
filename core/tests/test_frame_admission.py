import unittest

from core.application.ingestion import FrameAdmissionScheduler


class FrameAdmissionSchedulerTests(unittest.TestCase):
    def test_consumes_tokens_until_non_keyframes_are_rate_limited(self) -> None:
        scheduler = FrameAdmissionScheduler(capacity=2, refill_rate_per_second=0)

        self.assertTrue(scheduler.admit("cam-1", False, 0).admitted)
        self.assertTrue(scheduler.admit("cam-1", False, 0).admitted)
        dropped = scheduler.admit("cam-1", False, 0)

        self.assertFalse(dropped.admitted)
        self.assertTrue(dropped.advance_tracking)
        self.assertEqual(dropped.reason, "rate_limited")

    def test_refills_tokens_by_elapsed_time(self) -> None:
        scheduler = FrameAdmissionScheduler(capacity=2, refill_rate_per_second=1)
        scheduler.admit("cam-1", False, 0)
        scheduler.admit("cam-1", False, 0)
        scheduler.admit("cam-1", False, 0)

        decision = scheduler.admit("cam-1", False, 1)

        self.assertTrue(decision.admitted)
        self.assertEqual(decision.tokens_remaining, 0)

    def test_keyframe_is_admitted_without_consuming_regular_token(self) -> None:
        scheduler = FrameAdmissionScheduler(capacity=1, refill_rate_per_second=0)
        scheduler.admit("cam-1", False, 0)

        keyframe = scheduler.admit("cam-1", True, 0)
        next_regular = scheduler.admit("cam-1", False, 0)

        self.assertTrue(keyframe.admitted)
        self.assertEqual(keyframe.reason, "keyframe")
        self.assertFalse(next_regular.admitted)

    def test_camera_buckets_are_independent(self) -> None:
        scheduler = FrameAdmissionScheduler(capacity=1, refill_rate_per_second=0)
        scheduler.admit("cam-1", False, 0)

        self.assertTrue(scheduler.admit("cam-2", False, 0).admitted)
        self.assertFalse(scheduler.admit("cam-1", False, 0).admitted)

    def test_backwards_clock_does_not_add_tokens(self) -> None:
        scheduler = FrameAdmissionScheduler(capacity=1, refill_rate_per_second=5)
        scheduler.admit("cam-1", False, 2)

        decision = scheduler.admit("cam-1", False, 1)

        self.assertFalse(decision.admitted)

    def test_forget_idle_cameras_removes_stale_buckets(self) -> None:
        scheduler = FrameAdmissionScheduler(capacity=1, refill_rate_per_second=0)
        scheduler.admit("cam-1", False, 0)
        scheduler.admit("cam-2", False, 9)

        removed = scheduler.forget_idle_cameras(now_seconds=10, idle_timeout_seconds=5)

        self.assertEqual(removed, 1)
        self.assertTrue(scheduler.admit("cam-1", False, 10).admitted)

    def test_rejects_invalid_configuration_and_clock(self) -> None:
        with self.assertRaises(ValueError):
            FrameAdmissionScheduler(0, 1)
        scheduler = FrameAdmissionScheduler(1, 0)
        with self.assertRaises(ValueError):
            scheduler.admit("cam-1", False, float("nan"))


if __name__ == "__main__":
    unittest.main()