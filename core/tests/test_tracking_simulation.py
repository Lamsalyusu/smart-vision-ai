import unittest

from core.application.simulation import (
    SyntheticTrackingScenario,
    evaluate_tracking_sequence,
)


class SyntheticTrackingScenarioTests(unittest.TestCase):
    def test_seed_produces_repeatable_ground_truth_and_detections(self) -> None:
        first = SyntheticTrackingScenario(frame_count=5, object_count=3, seed=7).generate()
        second = SyntheticTrackingScenario(frame_count=5, object_count=3, seed=7).generate()

        self.assertEqual(first, second)
        self.assertEqual(len(first[0].ground_truth), 3)
        self.assertEqual(len(first[0].detections), 3)

    def test_scenario_marks_occlusion_and_injects_false_positives(self) -> None:
        frames = SyntheticTrackingScenario(
            frame_count=3,
            object_count=2,
            seed=2,
            occlusions={1: [1]},
            false_positives_per_frame=1,
        ).generate()

        self.assertEqual(frames[1].occluded_ids, (1,))
        self.assertEqual(len(frames[1].detections), 2)
        self.assertEqual(sum(detection["class"] == "unknown" for detection in frames[1].detections), 1)


class TrackingMetricTests(unittest.TestCase):
    def test_perfect_sequence_scores_one_for_all_metrics(self) -> None:
        truth = [[{"object_id": 1, "class": "person", "bbox": [0, 0, 10, 10]}]]
        prediction = [[{"track_id": 8, "class": "person", "bbox": [0, 0, 10, 10]}]]

        metrics = evaluate_tracking_sequence(truth, prediction)

        self.assertEqual(metrics.mota, 1.0)
        self.assertEqual(metrics.motp, 1.0)
        self.assertEqual(metrics.idf1, 1.0)

    def test_counts_misses_false_positives_and_identity_switches(self) -> None:
        truth = [
            [{"object_id": 1, "class": "person", "bbox": [0, 0, 10, 10]}],
            [{"object_id": 1, "class": "person", "bbox": [1, 0, 11, 10]}],
        ]
        predictions = [
            [{"track_id": 7, "class": "person", "bbox": [0, 0, 10, 10]}],
            [
                {"track_id": 9, "class": "person", "bbox": [1, 0, 11, 10]},
                {"track_id": 10, "class": "vehicle", "bbox": [20, 20, 30, 30]},
            ],
        ]

        metrics = evaluate_tracking_sequence(truth, predictions)

        self.assertEqual(metrics.true_positives, 2)
        self.assertEqual(metrics.false_positives, 1)
        self.assertEqual(metrics.false_negatives, 0)
        self.assertEqual(metrics.identity_switches, 1)
        self.assertAlmostEqual(metrics.mota, 0.0)
        self.assertLess(metrics.idf1, 1.0)

    def test_occlusion_contributes_false_negative(self) -> None:
        truth = [[{"object_id": 1, "class": "person", "bbox": [0, 0, 10, 10]}]]

        metrics = evaluate_tracking_sequence(truth, [[]])

        self.assertEqual(metrics.false_negatives, 1)
        self.assertEqual(metrics.mota, 0.0)
        self.assertEqual(metrics.idf1, 0.0)

    def test_rejects_different_sequence_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame counts must match"):
            evaluate_tracking_sequence([[]], [])


if __name__ == "__main__":
    unittest.main()