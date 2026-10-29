import math
import unittest

from core.domain.tracking import TrackState3D, UnscentedKalmanFilter3D


class UnscentedKalmanFilter3DTests(unittest.TestCase):
    def test_predicts_position_and_heading_from_velocity(self) -> None:
        filter_state = UnscentedKalmanFilter3D(
            TrackState3D(1, 2, 3, 4, 5, 6, 0.25, 0.5)
        )

        predicted = filter_state.predict(0.5)

        self.assertAlmostEqual(predicted.x, 3)
        self.assertAlmostEqual(predicted.y, 4.5)
        self.assertAlmostEqual(predicted.z, 6)
        self.assertAlmostEqual(predicted.heading_rad, 0.5)

    def test_measurement_update_moves_state_toward_observation(self) -> None:
        filter_state = UnscentedKalmanFilter3D(TrackState3D(0, 0, 0, 0, 0, 0, 0, 0))
        filter_state.predict(0.1)

        updated = filter_state.update((1, -2, 0.5, 0.2))

        self.assertGreater(updated.x, 0)
        self.assertLess(updated.x, 1)
        self.assertLess(updated.y, 0)
        self.assertGreater(updated.y, -2)

    def test_update_reduces_covariance_for_observed_position(self) -> None:
        filter_state = UnscentedKalmanFilter3D(TrackState3D(0, 0, 0, 0, 0, 0, 0, 0))
        filter_state.predict(0.1)
        predicted_variance = filter_state.covariance[0][0]

        filter_state.update((0.2, 0, 0, 0))

        self.assertLess(filter_state.covariance[0][0], predicted_variance)

    def test_heading_update_uses_short_path_across_pi_boundary(self) -> None:
        filter_state = UnscentedKalmanFilter3D(
            TrackState3D(0, 0, 0, 0, 0, 0, math.pi - 0.1, 0)
        )

        updated = filter_state.update((0, 0, 0, -math.pi + 0.1))
        angular_error = abs((updated.heading_rad - math.pi + math.pi) % (2 * math.pi) - math.pi)

        self.assertLess(angular_error, 0.1)

    def test_rejects_negative_prediction_interval(self) -> None:
        filter_state = UnscentedKalmanFilter3D(TrackState3D(0, 0, 0, 0, 0, 0, 0, 0))

        with self.assertRaises(ValueError):
            filter_state.predict(-0.1)

    def test_rejects_malformed_measurements(self) -> None:
        filter_state = UnscentedKalmanFilter3D(TrackState3D(0, 0, 0, 0, 0, 0, 0, 0))

        for measurement in ((0, 0, 0), (0, 0, 0, float("nan"))):
            with self.subTest(measurement=measurement), self.assertRaises(ValueError):
                filter_state.update(measurement)

    def test_rejects_singular_initial_covariance(self) -> None:
        singular_covariance = [[0.0] * 8 for _ in range(8)]

        with self.assertRaisesRegex(ValueError, "positive definite"):
            UnscentedKalmanFilter3D(
                TrackState3D(0, 0, 0, 0, 0, 0, 0, 0),
                covariance=singular_covariance,
            )


if __name__ == "__main__":
    unittest.main()