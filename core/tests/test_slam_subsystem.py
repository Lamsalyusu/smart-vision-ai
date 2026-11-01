import unittest

import numpy as np

from core.domain.slam import (
    SE3,
    SO3,
    ExtendedKalmanFilterSLAM,
    OccupancyGrid3D,
)


class TestSlamSubsystem(unittest.TestCase):
    def test_so3_exp_log(self):
        omega = np.array([0.1, -0.2, 0.3])
        r = SO3.exp(omega)
        self.assertEqual(r.shape, (3, 3))
        recovered_omega = SO3.log(r)
        np.testing.assert_allclose(recovered_omega, omega, atol=1e-5)

    def test_se3_inverse_and_composition(self):
        t1 = SE3(SO3.exp(np.array([0.1, 0.0, 0.0])), np.array([1.0, 2.0, 3.0]))
        t1_inv = t1.inverse()
        ident = t1.compose(t1_inv)
        np.testing.assert_allclose(ident.r, np.eye(3), atol=1e-5)
        np.testing.assert_allclose(ident.t, np.zeros(3), atol=1e-5)

    def test_occupancy_grid_raycasting(self):
        grid = OccupancyGrid3D(bounds_min=(-2.0, -2.0, -1.0), bounds_max=(2.0, 2.0, 1.0), voxel_size=0.2)
        origin = np.array([0.0, 0.0, 0.0])
        endpoint = np.array([1.0, 0.0, 0.0])
        grid.integrate_ray(origin, endpoint)
        prob = grid.get_probability_grid()
        self.assertGreater(prob.max(), 0.5)

    def test_ekf_slam_prediction_and_update(self):
        ekf = ExtendedKalmanFilterSLAM()
        ekf.predict(v=1.0, omega=0.1, dt=0.1)
        self.assertGreater(ekf.state[0], 0.0)

        # Update landmark
        ekf.update_measurement(landmark_id=1, r_meas=2.0, phi_meas=0.0)
        self.assertIn(1, ekf.landmark_map)


if __name__ == "__main__":
    unittest.main()
