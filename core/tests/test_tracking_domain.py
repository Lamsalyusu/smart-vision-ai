import unittest

from core.domain.tracking import TrackState3D, mahalanobis_squared


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