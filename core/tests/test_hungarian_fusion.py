import unittest
import numpy as np

from core.domain.tracking.hungarian_fusion import (
    DetectionCandidate,
    HungarianFusionMatcher,
    TrackTarget,
)


class HungarianFusionTests(unittest.TestCase):
    def setUp(self):
        self.matcher = HungarianFusionMatcher(max_distance_threshold=0.7)

    def test_iou_computation(self):
        box1 = (0.0, 0.0, 10.0, 10.0)
        box2 = (5.0, 0.0, 15.0, 10.0)
        iou = HungarianFusionMatcher.compute_iou(box1, box2)
        # Intersection is 5x10 = 50, union is 100 + 100 - 50 = 150 -> 1/3
        self.assertAlmostEqual(iou, 1.0 / 3.0, places=4)

    def test_cosine_distance(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(HungarianFusionMatcher.compute_cosine_distance(v1, v2), 0.0, places=5)

        v3 = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(HungarianFusionMatcher.compute_cosine_distance(v1, v3), 1.0, places=5)

    def test_optimal_bipartite_matching(self):
        # Two tracks
        feat1 = np.array([0.9, 0.1, 0.0])
        feat2 = np.array([0.0, 0.1, 0.9])

        track1 = TrackTarget(
            track_id=1,
            class_name="human_face",
            state_mean=np.array([100.0, 100.0, 2500.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            state_covariance=np.eye(8) * 10.0,
            feature_gallery=[feat1],
        )
        track2 = TrackTarget(
            track_id=2,
            class_name="sunglasses",
            state_mean=np.array([300.0, 300.0, 1600.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            state_covariance=np.eye(8) * 10.0,
            feature_gallery=[feat2],
        )

        # Two detection candidates
        det1 = DetectionCandidate(
            detection_id="d1",
            class_name="human_face",
            bbox=(75.0, 75.0, 125.0, 125.0),
            feature_embedding=feat1,
        )
        det2 = DetectionCandidate(
            detection_id="d2",
            class_name="sunglasses",
            bbox=(280.0, 280.0, 320.0, 320.0),
            feature_embedding=feat2,
        )

        matched, unmatched_t, unmatched_d = self.matcher.match([track1, track2], [det1, det2])
        self.assertEqual(len(matched), 2)
        self.assertEqual(matched[0], (0, 0))  # track1 -> det1
        self.assertEqual(matched[1], (1, 1))  # track2 -> det2
        self.assertEqual(len(unmatched_t), 0)
        self.assertEqual(len(unmatched_d), 0)

    def test_class_mismatch_gating(self):
        track = TrackTarget(
            track_id=1,
            class_name="knife",
            state_mean=np.array([50.0, 50.0, 100.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            state_covariance=np.eye(8),
            feature_gallery=[],
        )
        det = DetectionCandidate(
            detection_id="d1",
            class_name="water_bottle",
            bbox=(45.0, 45.0, 55.0, 55.0),
        )
        matched, unmatched_t, unmatched_d = self.matcher.match([track], [det])
        self.assertEqual(len(matched), 0)
        self.assertEqual(unmatched_t, [0])
        self.assertEqual(unmatched_d, [0])


if __name__ == "__main__":
    unittest.main()
