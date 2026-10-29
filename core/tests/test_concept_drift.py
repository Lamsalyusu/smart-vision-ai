import unittest

import numpy as np

from core.application.learning import ActiveLearningDriftPipeline


class ConceptDriftTests(unittest.TestCase):
    def setUp(self) -> None:
        random = np.random.default_rng(42)
        self.reference = random.normal(0, 0.1, size=(64, 3))

    def test_detects_shifted_embedding_window_with_ks_and_wasserstein(self) -> None:
        pipeline = ActiveLearningDriftPipeline(
            self.reference,
            minimum_samples=16,
            window_size=32,
            wasserstein_threshold=0.2,
        )
        shifted = np.ones((16, 3)) * 2

        report = pipeline.observe([f"frame-{i}" for i in range(16)], shifted, [0.9] * 16)

        self.assertTrue(report.drift_detected)
        self.assertGreater(max(report.ks_statistics), report.ks_critical_value)
        self.assertGreater(max(report.wasserstein_distances), 0.2)
        self.assertEqual(pipeline.queued_count, 16)
        self.assertEqual(pipeline.pop_candidate().reason, "embedding_drift")

    def test_stable_window_does_not_trigger_drift(self) -> None:
        pipeline = ActiveLearningDriftPipeline(
            self.reference,
            minimum_samples=16,
            window_size=32,
            wasserstein_threshold=0.5,
        )
        stable = self.reference[:16]

        report = pipeline.observe([f"stable-{i}" for i in range(16)], stable, [0.9] * 16)

        self.assertFalse(report.drift_detected)
        self.assertEqual(pipeline.queued_count, 0)

    def test_queues_low_confidence_frames_before_drift_window_is_ready(self) -> None:
        pipeline = ActiveLearningDriftPipeline(
            self.reference,
            minimum_samples=16,
            window_size=32,
        )

        report = pipeline.observe(["frame-low"], [[0, 0, 0]], [0.2])

        self.assertFalse(report.drift_detected)
        self.assertEqual(pipeline.pop_candidate().reason, "low_confidence")

    def test_queue_is_bounded_and_prioritizes_harder_frames(self) -> None:
        pipeline = ActiveLearningDriftPipeline(
            self.reference,
            queue_capacity=2,
            confidence_threshold=1.0,
            minimum_samples=16,
            window_size=32,
        )

        pipeline.observe(
            ["easy", "medium", "hard"],
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
            [0.9, 0.5, 0.1],
        )

        self.assertEqual(pipeline.queued_count, 2)
        self.assertEqual(pipeline.pop_candidate().frame_id, "hard")

    def test_rejects_mismatched_batch_lengths(self) -> None:
        pipeline = ActiveLearningDriftPipeline(self.reference)

        with self.assertRaisesRegex(ValueError, "equal lengths"):
            pipeline.observe(["frame"], [], [])


if __name__ == "__main__":
    unittest.main()