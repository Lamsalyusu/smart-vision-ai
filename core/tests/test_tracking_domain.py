import unittest

from core.domain.tracking import (
    TrackKinematicLimits,
    TrackLifecycle,
    TrackState3D,
    TrackStatus,
    mahalanobis_squared,
)


class TrackState3DTests(unittest.TestCase):
    def test_state_uses_requested_eight_component_order(self) -> None:
        state = TrackState3D(1, 2, 3, 4, 5, 6, 0.7, 0.8)

        self.assertEqual(state.as_vector(), (1, 2, 3, 4, 5, 6, 0.7, 0.8))

    def test_prediction_advances_position_and_heading(self) -> None:
        state = TrackState3D(1, 2, 3, 4, 5, 6, 0.7, 0.8).predict(0.5)

        self.assertEqual((state.x, state.y, state.z), (3, 4.5, 6))
        self.assertAlmostEqual(state.heading_rad, 1.1)

    def test_prediction_rejects_negative_elapsed_time(self) -> None:
        with self.assertRaises(ValueError):
            TrackState3D(0, 0, 0, 0, 0, 0, 0, 0).predict(-0.1)


class TrackLifecycleTests(unittest.TestCase):
    def test_confirm_coast_and_reidentify_track(self) -> None:
        lifecycle = TrackLifecycle(track_id=7, class_name="person", min_hits_to_confirm=3)
        first_state = TrackState3D(0, 0, 0, 0, 0, 0, 0, 0)
        lifecycle.register_hit(first_state)
        self.assertEqual(lifecycle.status, TrackStatus.TENTATIVE)

        lifecycle.register_hit(first_state)
        self.assertEqual(lifecycle.status, TrackStatus.CONFIRMED)
        self.assertTrue(lifecycle.is_active)
        self.assertTrue(lifecycle.is_visible)

        self.assertEqual(lifecycle.register_miss(), TrackStatus.COASTING)
        lifecycle.register_hit(first_state, is_reid=True)
        self.assertEqual(lifecycle.status, TrackStatus.RE_IDENTIFIED)

    def test_implausible_jump_does_not_update_track(self) -> None:
        lifecycle = TrackLifecycle(
            track_id=9,
            class_name="person",
            last_state=TrackState3D(0, 0, 0, 0, 0, 0, 0, 0),
            kinematic_limits=TrackKinematicLimits(max_speed_mps=1),
        )

        lifecycle.register_hit(TrackState3D(100, 0, 0, 0, 0, 0, 0, 0))

        self.assertEqual(lifecycle.total_hits, 1)
        self.assertEqual(lifecycle.status, TrackStatus.TENTATIVE)
        self.assertIsNotNone(lifecycle.last_state)
        assert lifecycle.last_state is not None
        self.assertEqual(lifecycle.last_state.x, 0)


class MahalanobisAssociationTests(unittest.TestCase):
    def test_identity_covariance_matches_squared_euclidean_distance(self) -> None:
        distance = mahalanobis_squared((3, 4), ((1, 0), (0, 1)))

        self.assertAlmostEqual(distance, 25)

    def test_covariance_scales_association_cost(self) -> None:
        distance = mahalanobis_squared((2, 0), ((2, 0), (0, 2)))

        self.assertAlmostEqual(distance, 2)

    def test_rejects_asymmetric_covariance(self) -> None:
        with self.assertRaises(ValueError):
            mahalanobis_squared((1, 1), ((1, 0.5), (0, 1)))

    def test_rejects_singular_covariance(self) -> None:
        with self.assertRaises(ValueError):
            mahalanobis_squared((1, 1), ((1, 1), (1, 1)))

    def test_rejects_dimension_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            mahalanobis_squared((1, 2), ((1,),))


if __name__ == "__main__":
    unittest.main()