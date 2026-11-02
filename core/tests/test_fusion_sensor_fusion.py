import unittest

import numpy as np

from core.domain.fusion.sensor_fusion import (
    CameraModel,
    ExtendedKalmanFilter,
    FactorGraphFusion,
    FusedState,
    GPSModel,
    IMUModel,
    KalmanFilter,
    LidarModel,
    LooselyCoupledFusion,
    MultiSensorFusion,
    OutlierRejector,
    RadarModel,
    SensorCalibration,
    SensorConfig,
    SensorData,
    SensorHealthMonitor,
    SensorModel,
    SensorModality,
    SensorSync,
    TightlyCoupledFusion,
    UnscentedKalmanFilter,
)


class SensorModalityTests(unittest.TestCase):
    def test_values(self):
        self.assertEqual(SensorModality.CAMERA.value, "camera")
        self.assertEqual(SensorModality.LIDAR.value, "lidar")
        self.assertEqual(SensorModality.RADAR.value, "radar")


class SensorConfigTests(unittest.TestCase):
    def test_creation(self):
        config = SensorConfig(
            sensor_id="cam1",
            modality=SensorModality.CAMERA,
            update_rate=30.0,
            noise_model={"std": 1.0},
        )
        self.assertEqual(config.sensor_id, "cam1")


class SensorDataTests(unittest.TestCase):
    def test_creation(self):
        data = SensorData(
            sensor_id="cam1",
            modality=SensorModality.CAMERA,
            timestamp=1.0,
            data=np.array([1.0, 2.0]),
            covariance=np.eye(2),
        )
        self.assertEqual(data.timestamp, 1.0)


class KalmanFilterTests(unittest.TestCase):
    def test_predict_and_update(self):
        kf = KalmanFilter(state_dim=9, meas_dim=2)
        kf.set_model(
            F=np.eye(9),
            Q=np.eye(9) * 0.01,
            H=np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            R=np.eye(2) * 0.1,
        )
        kf.predict(1.0)
        kf.update(np.array([1.0, 2.0]))
        x, P = kf.get_state()
        self.assertEqual(x.shape, (9,))
        self.assertEqual(P.shape, (9, 9))

    def test_get_state(self):
        kf = KalmanFilter(state_dim=9, meas_dim=2)
        x, P = kf.get_state()
        self.assertEqual(x.shape, (9,))
        self.assertEqual(P.shape, (9, 9))


class ExtendedKalmanFilterTests(unittest.TestCase):
    def test_predict_and_update(self):
        def f_func(x, dt):
            return x.copy()

        def F_func(x, dt):
            return np.eye(len(x))

        def h_func(x):
            return np.array([x[0]])

        def H_func(x):
            return np.array([[1.0, 0.0, 0.0]])

        ekf = ExtendedKalmanFilter(state_dim=3)
        ekf.predict(f_func, F_func, np.eye(3) * 0.01, 1.0)
        ekf.update(np.array([1.0]), h_func, H_func, np.array([[0.1]]))
        self.assertEqual(ekf.x.shape, (3,))


class UnscentedKalmanFilterTests(unittest.TestCase):
    def test_predict_and_update(self):
        def f_func(x, dt):
            return x.copy()

        def h_func(x):
            return np.array([x[0]])

        ukf = UnscentedKalmanFilter(state_dim=3)
        ukf.predict(f_func, np.eye(3) * 0.01, 1.0)
        ukf.update(np.array([1.0]), h_func, np.array([[0.1]]))
        self.assertEqual(ukf.x.shape, (3,))


class CameraModelTests(unittest.TestCase):
    def test_get_measurement_matrix(self):
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        model = CameraModel(K, noise_std=1.0)
        H = model.get_measurement_matrix()
        self.assertEqual(H.shape, (2, 15))

    def test_project(self):
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        model = CameraModel(K, noise_std=1.0)
        uv = model.project(np.array([1.0, 1.0, 1.0]))
        self.assertEqual(uv.shape, (2,))


class MultiSensorFusionTests(unittest.TestCase):
    def test_add_sensor_and_initialize(self):
        fusion = MultiSensorFusion()
        model = CameraModel(np.eye(3), noise_std=1.0)
        fusion.add_sensor("cam1", model)
        fusion.set_fusion_method("kalman")
        fusion.initialize(np.zeros(15))
        self.assertIsNotNone(fusion.kf)

    def test_process_empty_returns_none(self):
        fusion = MultiSensorFusion()
        result = fusion.process([])
        self.assertIsNone(result)


class LooselyCoupledFusionTests(unittest.TestCase):
    def test_initialize_and_process(self):
        fusion = LooselyCoupledFusion()
        fusion.initialize()

        data = SensorData(
            sensor_id="cam1",
            modality=SensorModality.CAMERA,
            timestamp=1.0,
            data=np.array([1.0, 2.0]),
            covariance=np.eye(2),
        )
        fusion.process_camera(data, 0.1)
        result = fusion.get_fused_state()
        self.assertIsNotNone(result)


class TightlyCoupledFusionTests(unittest.TestCase):
    def test_process_empty(self):
        fusion = TightlyCoupledFusion()
        result = fusion.process_measurements([])
        self.assertIsNotNone(result)


class FactorGraphFusionTests(unittest.TestCase):
    def test_add_and_optimize(self):
        fg = FactorGraphFusion()
        fg.add_variable("x1", 3, np.array([1.0, 2.0, 3.0]))
        fg.add_prior_factor("x1", np.array([0.0, 0.0, 0.0]), np.eye(3))
        result = fg.optimize(max_iterations=10)
        self.assertIn("x1", result)


class SensorCalibrationTests(unittest.TestCase):
    def test_calibrate_camera_lidar_insufficient_points(self):
        cal = SensorCalibration()
        cam_pts = np.random.default_rng(0).uniform(0, 1, size=(5, 3))
        lidar_pts = np.random.default_rng(0).uniform(0, 1, size=(5, 3))
        T = cal.calibrate_camera_lidar(cam_pts, lidar_pts)
        self.assertTrue(np.allclose(T, np.eye(4)))

    def test_calibrate_imu_camera(self):
        cal = SensorCalibration()
        R, t = cal.calibrate_imu_camera(np.zeros((10, 6)), np.zeros((10, 3)))
        self.assertTrue(np.allclose(R, np.eye(3)))


class SensorSyncTests(unittest.TestCase):
    def test_add_and_get_synchronized(self):
        sync = SensorSync(max_latency=0.1)

        data = SensorData(
            sensor_id="cam1",
            modality=SensorModality.CAMERA,
            timestamp=1.0,
            data=np.array([1.0]),
            covariance=np.eye(1),
        )
        sync.add_measurement("cam1", data)
        result = sync.get_synchronized(1.0)
        self.assertIn("cam1", result)


class OutlierRejectorTests(unittest.TestCase):
    def test_filter_returns_true_for_few_samples(self):
        rejector = OutlierRejector()
        result, accepted = rejector.filter("cam1", np.array([1.0, 2.0]))
        self.assertTrue(accepted)

    def test_filter_detects_outlier(self):
        rejector = OutlierRejector(threshold=2.0)
        for i in range(15):
            rejector.filter("cam1", np.array([1.0, 2.0]))
        result, accepted = rejector.filter("cam1", np.array([100.0, 200.0]))
        self.assertFalse(accepted)


class SensorHealthMonitorTests(unittest.TestCase):
    def test_record_and_get_health(self):
        monitor = SensorHealthMonitor()
        monitor.record_measurement("cam1", 0.01, True)
        monitor.record_measurement("cam1", 0.02, True)
        health = monitor.get_health("cam1")
        self.assertTrue(health["healthy"])

    def test_get_health_no_data(self):
        monitor = SensorHealthMonitor()
        health = monitor.get_health("unknown")
        self.assertFalse(health["healthy"])


if __name__ == "__main__":
    unittest.main()
