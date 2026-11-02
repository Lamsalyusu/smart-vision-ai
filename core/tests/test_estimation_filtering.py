import unittest

import numpy as np

from core.domain.estimation.filtering import (
    ExtendedKalmanFilter,
    KalmanFilter,
    MotionModel,
    ParticleFilter,
    UnscentedKalmanFilter,
)


class KalmanFilterTests(unittest.TestCase):
    def test_predict_and_update(self):
        kf = KalmanFilter(
            state_dim=2,
            measurement_dim=1,
            F=np.array([[1.0, 0.0], [0.0, 1.0]]),
            H=np.array([[1.0, 0.0]]),
            Q=np.eye(2) * 0.01,
            R=np.array([[0.1]]),
        )
        state = kf.predict()
        self.assertEqual(state.x.shape, (2,))
        self.assertEqual(state.P.shape, (2, 2))

        state = kf.update(np.array([1.0]))
        self.assertEqual(state.x.shape, (2,))

    def test_get_state(self):
        kf = KalmanFilter(
            state_dim=2,
            measurement_dim=1,
            F=np.eye(2),
            H=np.array([[1.0, 0.0]]),
            Q=np.eye(2) * 0.01,
            R=np.array([[0.1]]),
        )
        estimate = kf.get_state()
        self.assertIsNotNone(estimate)
        self.assertEqual(estimate.x.shape, (2,))


class ExtendedKalmanFilterTests(unittest.TestCase):
    def test_predict_and_update(self):
        def motion_model(x, dt):
            new_x = x.copy()
            if len(x) >= 2:
                new_x[0] += x[1] * dt
            return new_x

        def motion_jacobian(x, dt):
            F = np.eye(len(x))
            if len(x) >= 2:
                F[0, 1] = dt
            return F

        def h_func(x):
            return np.array([x[0]])

        def H_func(x):
            return np.array([[1.0, 0.0]])

        ekf = ExtendedKalmanFilter(
            state_dim=2,
            process_noise=np.eye(2) * 0.01,
            motion_model=motion_model,
            motion_jacobian=motion_jacobian,
            measurement_models={"sensor1": (h_func, H_func)},
        )
        state = ekf.predict(1.0)
        self.assertEqual(state.x.shape, (2,))

        state = ekf.update(np.array([1.0]), "sensor1")
        self.assertEqual(state.x.shape, (2,))

    def test_update_unknown_sensor_raises(self):
        def motion_model(x, dt):
            return x.copy()

        def motion_jacobian(x, dt):
            return np.eye(len(x))

        ekf = ExtendedKalmanFilter(
            state_dim=2,
            process_noise=np.eye(2) * 0.01,
            motion_model=motion_model,
            motion_jacobian=motion_jacobian,
            measurement_models={},
        )
        with self.assertRaises(ValueError):
            ekf.update(np.array([1.0]), "unknown")


class UnscentedKalmanFilterTests(unittest.TestCase):
    def test_predict_and_update(self):
        def motion_model(x, dt):
            return x.copy()

        def h_func(x):
            return np.array([x[0]])

        ukf = UnscentedKalmanFilter(
            state_dim=2,
            process_noise=np.eye(2) * 0.01,
        )
        state = ukf.predict(1.0, motion_model)
        self.assertEqual(state.x.shape, (2,))

        state = ukf.update(np.array([1.0]), np.array([[0.1]]), h_func)
        self.assertEqual(state.x.shape, (2,))


class ParticleFilterTests(unittest.TestCase):
    def test_predict_and_update(self):
        def motion_model(x, dt, u=None):
            return x.copy()

        pf = ParticleFilter(
            state_dim=2,
            num_particles=100,
            process_noise=np.eye(2) * 0.01,
            seed=42,
        )
        state = pf.predict(1.0, motion_model)
        self.assertEqual(state.x.shape, (2,))

        def h_func(x):
            return np.array([x[0]])

        result = pf.update(np.array([1.0]), np.array([[0.1]]), h_func)
        self.assertEqual(result.x.shape, (2,))

    def test_deterministic_with_seed(self):
        def motion_model(x, dt, u=None):
            return x.copy()

        pf1 = ParticleFilter(state_dim=2, num_particles=50, seed=42)
        pf2 = ParticleFilter(state_dim=2, num_particles=50, seed=42)
        self.assertTrue(np.allclose(pf1.particles, pf2.particles))


class MotionModelEnumTests(unittest.TestCase):
    def test_values(self):
        self.assertEqual(MotionModel.CONSTANT_VELOCITY.value, "constant_velocity")
        self.assertEqual(MotionModel.ACKERMANN.value, "ackermann")


if __name__ == "__main__":
    unittest.main()
