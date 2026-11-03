from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math
import time


@dataclass(frozen=True, slots=True)
class SensorData:
    sensor_id: str
    timestamp: float
    data: np.ndarray
    covariance: np.ndarray
    pose: Optional[np.ndarray] = None


@dataclass(frozen=True, slots=True)
class FusedState:
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    angular_velocity: np.ndarray
    covariance: np.ndarray
    sensor_contributions: Dict[str, float]


class SensorModel(ABC):
    @abstractmethod
    def predict(self, state: FusedState, dt: float) -> FusedState:
        pass

    @abstractmethod
    def update(self, state: FusedState, measurement: SensorData) -> FusedState:
        pass

    @abstractmethod
    def get_measurement_matrix(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_noise_covariance(self) -> np.ndarray:
        pass


class CameraModel(SensorModel):
    def __init__(self, intrinsics: np.ndarray, distortion: Optional[np.ndarray] = None):
        self.intrinsics = intrinsics
        self.distortion = distortion

    def predict(self, state: FusedState, dt: float) -> FusedState:
        return state

    def update(self, state: FusedState, measurement: SensorData) -> FusedState:
        return state

    def get_measurement_matrix(self) -> np.ndarray:
        H = np.zeros((2, 15))
        H[0, 0] = self.intrinsics[0, 0]
        H[0, 2] = self.intrinsics[0, 2]
        H[1, 1] = self.intrinsics[1, 1]
        H[1, 2] = self.intrinsics[1, 2]
        return H

    def get_noise_covariance(self) -> np.ndarray:
        return np.eye(2) * 1.0

    def project(self, point_3d: np.ndarray) -> np.ndarray:
        point_cam = point_3d / point_3d[2]
        uv = self.intrinsics @ point_cam
        return uv[:2]


class LidarModel(SensorModel):
    def __init__(self, noise_std: float = 0.05):
        self.noise_std = noise_std

    def predict(self, state: FusedState, dt: float) -> FusedState:
        return state

    def update(self, state: FusedState, measurement: SensorData) -> FusedState:
        return state

    def get_measurement_matrix(self) -> np.ndarray:
        H = np.zeros((3, 15))
        H[:3, :3] = np.eye(3)
        return H

    def get_noise_covariance(self) -> np.ndarray:
        return np.eye(3) * self.noise_std**2


class IMUModel(SensorModel):
    def __init__(self, accel_noise: float = 0.01, gyro_noise: float = 0.001):
        self.accel_noise = accel_noise
        self.gyro_noise = gyro_noise

    def predict(self, state: FusedState, dt: float) -> FusedState:
        return state

    def update(self, state: FusedState, measurement: SensorData) -> FusedState:
        return state

    def get_measurement_matrix(self) -> np.ndarray:
        H = np.zeros((6, 15))
        H[:3, 6:9] = np.eye(3)
        H[3:, 9:12] = np.eye(3)
        return H

    def get_noise_covariance(self) -> np.ndarray:
        return np.diag([self.accel_noise**2]*3 + [self.gyro_noise**2]*3)


class GPSModel(SensorModel):
    def __init__(self, noise_std: float = 1.0):
        self.noise_std = noise_std

    def predict(self, state: FusedState, dt: float) -> FusedState:
        return state

    def update(self, state: FusedState, measurement: SensorData) -> FusedState:
        return state

    def get_measurement_matrix(self) -> np.ndarray:
        H = np.zeros((3, 15))
        H[:3, :3] = np.eye(3)
        return H

    def get_noise_covariance(self) -> np.ndarray:
        return np.eye(3) * self.noise_std**2


class RadarModel(SensorModel):
    def __init__(self, range_noise: float = 0.1, angle_noise: float = 0.01):
        self.range_noise = range_noise
        self.angle_noise = angle_noise

    def predict(self, state: FusedState, dt: float) -> FusedState:
        return state

    def update(self, state: FusedState, measurement: SensorData) -> FusedState:
        return state

    def get_measurement_matrix(self) -> np.ndarray:
        H = np.zeros((4, 15))
        H[0, :3] = np.eye(1, 3, 0)
        H[1, :3] = np.eye(1, 3, 1)
        H[2, :3] = np.eye(1, 3, 2)
        H[3, 3:6] = np.eye(1, 3, 0)
        return H

    def get_noise_covariance(self) -> np.ndarray:
        return np.diag([self.range_noise**2, self.angle_noise**2, self.angle_noise**2, self.range_noise**2])


class KalmanFilter:
    def __init__(self, state_dim: int, meas_dim: int):
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 1000
        self.F = np.eye(state_dim)
        self.Q = np.eye(state_dim) * 0.01
        self.H = np.zeros((meas_dim, state_dim))
        self.R = np.eye(meas_dim)

    def set_model(self, F: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray):
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R

    def predict(self, dt: float):
        self.F[:3, 3:6] = np.eye(3) * dt
        self.F[3:6, 6:9] = np.eye(3) * dt

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray, H: np.ndarray = None, R: np.ndarray = None):
        if H is not None:
            self.H = H
        if R is not None:
            self.R = R

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P

    def get_state(self) -> np.ndarray:
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        return self.P.copy()


class ExtendedKalmanFilter:
    def __init__(self, state_dim: int):
        self.state_dim = state_dim
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 1000

    def predict(self, f_func, F_func, Q: np.ndarray, dt: float):
        self.x = f_func(self.x, dt)
        F = F_func(self.x, dt)
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, h_func, H_func, R: np.ndarray):
        h = h_func(self.x)
        H = H_func(self.x)

        y = z - h
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P


class UnscentedKalmanFilter:
    def __init__(
        self,
        state_dim: int,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        self.state_dim = state_dim
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambd = alpha**2 * (state_dim + kappa) - state_dim
        self.n_sigma = 2 * state_dim + 1

        self.Wm = np.full(self.n_sigma, 1 / (2 * (state_dim + self.lambd)))
        self.Wc = np.full(self.n_sigma, 1 / (2 * (state_dim + self.lambd)))
        self.Wm[0] = self.lambd / (state_dim + self.lambd)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)

        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 1000

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        sqrt_P = np.linalg.cholesky((self.state_dim + self.lambd) * P)
        sigmas = np.zeros((self.n_sigma, self.state_dim))
        sigmas[0] = x
        for i in range(self.state_dim):
            sigmas[i + 1] = x + sqrt_P[i]
            sigmas[self.state_dim + i + 1] = x - sqrt_P[i]
        return sigmas

    def predict(self, f_func, Q: np.ndarray, dt: float):
        sigmas = self._sigma_points(self.x, self.P)
        sigmas_f = np.array([f_func(s, dt) for s in sigmas])

        self.x = np.sum(self.Wm[:, None] * sigmas_f, axis=0)
        diff = sigmas_f - self.x
        self.P = diff.T @ np.diag(self.Wc) @ diff + Q

    def update(self, z: np.ndarray, h_func, R: np.ndarray):
        sigmas = self._sigma_points(self.x, self.P)
        sigmas_h = np.array([h_func(s) for s in sigmas])

        z_pred = np.sum(self.Wm[:, None] * sigmas_h, axis=0)
        diff_z = sigmas_h - z_pred
        diff_x = sigmas - self.x

        P_zz = diff_z.T @ np.diag(self.Wc) @ diff_z + R
        P_xz = diff_x.T @ np.diag(self.Wc) @ diff_z

        K = P_xz @ np.linalg.inv(P_zz)
        self.x = self.x + K @ (z - z_pred)
        self.P = self.P - K @ P_zz @ K.T


class MultiSensorFusion:
    def __init__(self):
        self.sensors: Dict[str, SensorModel] = {}
        self.kf: Optional[KalmanFilter] = None
        self.ekf: Optional[ExtendedKalmanFilter] = None
        self.ukf: Optional[UnscentedKalmanFilter] = None
        self.fusion_method: str = "kalman"
        self.last_timestamp = 0.0

    def add_sensor(self, sensor_id: str, sensor_model: SensorModel):
        self.sensors[sensor_id] = sensor_model

    def set_fusion_method(self, method: str):
        self.fusion_method = method

    def initialize(self, initial_state: np.ndarray):
        if self.fusion_method == "kalman":
            self.kf = KalmanFilter(15, 6)
            self.kf.x = initial_state
        elif self.fusion_method == "ekf":
            self.ekf = ExtendedKalmanFilter(15)
            self.ekf.x = initial_state
        elif self.fusion_method == "ukf":
            self.ukf = UnscentedKalmanFilter(15)
            self.ukf.x = initial_state

    def process(self, measurements: List) -> Optional[np.ndarray]:
        if not measurements:
            return None

        dt = measurements[0].timestamp - self.last_timestamp
        self.last_timestamp = measurements[0].timestamp

        if self.fusion_method == "kalman" and self.kf:
            self.kf.predict(dt)
            for meas in measurements:
                if meas.sensor_id in self.sensors:
                    sensor = self.sensors[meas.sensor_id]
                    H = sensor.get_measurement_matrix()
                    R = sensor.get_noise_covariance()
                    self.kf.update(meas.data, H, R)
            return self.kf.get_state()

        elif self.fusion_method == "ekf" and self.ekf:
            def f_func(x, dt):
                return x
            def F_func(x, dt):
                F = np.eye(15)
                F[:3, 3:6] = np.eye(3) * dt
                return F
            self.ekf.predict(f_func, F_func, np.eye(15)*0.01, dt)

            for meas in measurements:
                if meas.sensor_id in self.sensors:
                    sensor = self.sensors[meas.sensor_id]
                    def h_func(x):
                        return sensor.get_measurement_matrix() @ x
                    def H_func(x):
                        return sensor.get_measurement_matrix()
                    self.ekf.update(meas.data, h_func, H_func, sensor.get_noise_covariance())

        elif self.fusion_method == "ukf" and self.ukf:
            def f_func(x, dt):
                return x
            self.ukf.predict(f_func, np.eye(15)*0.01, dt)

            for meas in measurements:
                if meas.sensor_id in self.sensors:
                    sensor = self.sensors[meas.sensor_id]
                    def h_func(x):
                        return sensor.get_measurement_matrix() @ x
                    self.ukf.update(meas.data, h_func, sensor.get_noise_covariance())

        return None


class LooselyCoupledFusion:
    def __init__(self):
        self.camera_kf: Optional[KalmanFilter] = None
        self.lidar_kf: Optional[KalmanFilter] = None
        self.imu_kf: Optional[KalmanFilter] = None
        self.gps_kf: Optional[KalmanFilter] = None

    def initialize(self):
        self.camera_kf = KalmanFilter(15, 2)
        self.lidar_kf = KalmanFilter(15, 3)
        self.imu_kf = KalmanFilter(15, 6)
        self.gps_kf = KalmanFilter(15, 3)

    def process_camera(self, measurement, dt: float):
        if self.camera_kf:
            self.camera_kf.predict(dt)
            H = measurement.sensor_model.get_measurement_matrix() if hasattr(measurement, 'sensor_model') else np.zeros((2, 15))
            R = measurement.covariance
            self.camera_kf.update(measurement.data, H, R)

    def process_lidar(self, measurement, dt: float):
        if self.lidar_kf:
            self.lidar_kf.predict(dt)
            H = np.zeros((3, 15))
            H[:3, :3] = np.eye(3)
            R = measurement.covariance
            self.lidar_kf.update(measurement.data, H, R)

    def process_imu(self, measurement, dt: float):
        if self.imu_kf:
            self.imu_kf.predict(dt)
            H = np.zeros((6, 15))
            H[:3, 6:9] = np.eye(3)
            H[3:, 9:12] = np.eye(3)
            R = measurement.covariance
            self.imu_kf.update(measurement.data, H, R)

    def process_gps(self, measurement, dt: float):
        if self.gps_kf:
            self.gps_kf.predict(dt)
            H = np.zeros((3, 15))
            H[:3, :3] = np.eye(3)
            R = measurement.covariance
            self.gps_kf.update(measurement.data, H, R)

    def get_fused_state(self) -> np.ndarray:
        states = []
        covs = []

        for kf, weight in [
            (self.camera_kf, 0.2),
            (self.lidar_kf, 0.3),
            (self.imu_kf, 0.3),
            (self.gps_kf, 0.2),
        ]:
            if kf:
                state = kf.get_state()
                states.append(state[:3])
                covs.append(state.get_covariance()[:3, :3] * weight)

        if not states:
            return np.zeros(3)

        fused_pos = np.average(states, axis=0, weights=[0.2, 0.3, 0.3, 0.2][:len(states)])
        fused_cov = np.sum(covs, axis=0) if covs else np.eye(3)
        return fused_pos


class TightlyCoupledFusion:
    def __init__(self):
        self.kf = KalmanFilter(21, 6)
        self.last_timestamp = 0.0

    def initialize(self, initial_state: np.ndarray):
        self.kf = KalmanFilter(21, 6)
        self.kf.x[:15] = initial_state
        self.kf.P[:15, :15] = np.eye(15) * 1000
        self.last_timestamp = 0.0

    def process_measurements(self, measurements: List) -> np.ndarray:
        if not measurements:
            return self.kf.get_state()

        dt = measurements[0].timestamp - self.last_timestamp
        self.last_timestamp = measurements[0].timestamp

        self.kf.predict(dt)

        for meas in measurements:
            sensor = getattr(meas, 'sensor_model', None)
            if sensor:
                H = sensor.get_measurement_matrix()
                R = sensor.get_noise_covariance()
            else:
                H = np.zeros((meas.data.shape[0], 21))
                R = meas.covariance
            self.kf.update(meas.data, H, R)

        return self.kf.get_state()


class FactorGraphFusion:
    def __init__(self):
        self.factors: List[tuple] = []
        self.variables: Dict[str, np.ndarray] = {}
        self.variable_dim: Dict[str, int] = {}

    def add_variable(self, var_id: str, dim: int, initial: np.ndarray = None):
        self.variables[var_id] = initial if initial is not None else np.zeros(dim)
        self.variable_dim[var_id] = dim

    def add_prior_factor(self, var_id: str, mean: np.ndarray, covariance: np.ndarray):
        self.factors.append(('prior', var_id, mean, covariance))

    def add_between_factor(self, var1: str, var2: str, relative: np.ndarray, covariance: np.ndarray):
        self.factors.append(('between', var1, var2, relative, covariance))

    def add_measurement_factor(self, var_id: str, measurement: np.ndarray, H: np.ndarray, covariance: np.ndarray):
        self.factors.append(('measurement', var_id, measurement, H, covariance))

    def optimize(self, max_iterations: int = 50) -> Dict[str, np.ndarray]:
        var_ids = list(self.variables.keys())
        n = len(var_ids)
        dims = [self.variable_dim[vid] for vid in var_ids]
        total_dim = sum(dims)

        id_to_idx = {vid: i for i, vid in enumerate(var_ids)}
        idx_to_offset = {}
        offset = 0
        for i, vid in enumerate(var_ids):
            idx_to_offset[vid] = offset
            offset += dims[i]

        for _ in range(max_iterations):
            H = np.zeros((total_dim, total_dim))
            b = np.zeros(total_dim)

            for factor in self.factors:
                if factor[0] == 'prior':
                    _, var_id, mean, cov = factor
                    idx = id_to_idx[var_id]
                    offset = idx_to_offset[var_id]
                    dim = dims[idx]

                    H[offset:offset+dim, offset:offset+dim] += np.linalg.inv(cov)
                    b[offset:offset+dim] += np.linalg.inv(cov) @ mean

                elif factor[0] == 'between':
                    _, var1, var2, rel, cov = factor
                    idx1, idx2 = id_to_idx[var1], id_to_idx[var2]
                    off1, off2 = idx_to_offset[var1], idx_to_offset[var2]
                    dim1, dim2 = dims[idx1], dims[idx2]

                    J1 = np.eye(dim1)
                    J2 = -np.eye(dim1)

                    H[off1:off1+dim1, off1:off1+dim1] += J1.T @ np.linalg.inv(cov) @ J1
                    H[off2:off2+dim2, off2:off2+dim2] += J2.T @ np.linalg.inv(cov) @ J2
                    H[off1:off1+dim1, off2:off2+dim2] += J1.T @ np.linalg.inv(cov) @ J2
                    H[off2:off2+dim2, off1:off1+dim1] += J2.T @ np.linalg.inv(cov) @ J1

                    b[off1:off1+dim1] -= J1.T @ np.linalg.inv(cov) @ rel
                    b[off2:off2+dim2] -= J2.T @ np.linalg.inv(cov) @ rel

                elif factor[0] == 'measurement':
                    _, var_id, meas, H_mat, cov = factor
                    idx = id_to_idx[var_id]
                    offset = idx_to_offset[var_id]
                    dim = dims[idx]

                    H[offset:offset+dim, offset:offset+dim] += H_mat.T @ np.linalg.inv(cov) @ H_mat
                    b[offset:offset+dim] += H_mat.T @ np.linalg.inv(cov) @ meas

            try:
                dx = np.linalg.solve(H + 1e-6 * np.eye(total_dim), b)
            except np.linalg.LinAlgError:
                break

            for i, var_id in enumerate(var_ids):
                offset = idx_to_offset[var_id]
                dim = dims[i]
                self.variables[var_id] += dx[offset:offset+dim]

        return self.variables


class SensorCalibration:
    def __init__(self):
        self.extrinsics: Dict[str, np.ndarray] = {}
        self.intrinsics: Dict[str, np.ndarray] = {}
        self.time_offsets: Dict[str, float] = {}

    def set_extrinsic(self, sensor_id: str, transform: np.ndarray):
        self.extrinsics[sensor_id] = transform

    def set_intrinsic(self, sensor_id: str, matrix: np.ndarray):
        self.intrinsics[sensor_id] = matrix

    def set_time_offset(self, sensor_id: str, offset: float):
        self.time_offsets[sensor_id] = offset

    def calibrate_camera_lidar(self, camera_points: np.ndarray, lidar_points: np.ndarray) -> np.ndarray:
        if len(camera_points) != len(lidar_points) or len(camera_points) < 6:
            return np.eye(4)

        centroid_cam = np.mean(camera_points, axis=0)
        centroid_lidar = np.mean(lidar_points, axis=0)

        centered_cam = camera_points - centroid_cam
        centered_lidar = lidar_points - centroid_lidar

        H = centered_lidar.T @ centered_cam
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_cam - R @ centroid_lidar

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    def calibrate_imu_camera(self, imu_data: np.ndarray, camera_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.eye(3), np.zeros(3)


class SensorSync:
    def __init__(self, max_latency: float = 0.1):
        self.max_latency = max_latency
        self.buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.latest_timestamps: Dict[str, float] = {}

    def add_measurement(self, sensor_id: str, measurement):
        self.buffers[sensor_id].append(measurement)
        self.latest_timestamps[sensor_id] = measurement.timestamp

    def get_synchronized(self, target_time: float, tolerance: float = None) -> Dict[str, Any]:
        tol = tolerance or self.max_latency
        result = {}

        for sensor_id, buffer in self.buffers.items():
            best = None
            best_diff = float('inf')
            for meas in buffer:
                diff = abs(meas.timestamp - target_time)
                if diff < best_diff and diff <= tol:
                    best_diff = diff
                    best = meas
            if best:
                result[sensor_id] = best

        return result


class OutlierRejector:
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    def filter(self, sensor_id: str, measurement: np.ndarray) -> Tuple[np.ndarray, bool]:
        self.history[sensor_id].append(measurement)

        if len(self.history[sensor_id]) < 10:
            return measurement, True

        hist = np.array(self.history[sensor_id])
        mean = np.mean(hist, axis=0)
        std = np.std(hist, axis=0)
        std[std == 0] = 1.0

        z_scores = np.abs((measurement - mean) / std)
        is_outlier = np.any(z_scores > self.threshold)

        return measurement, not is_outlier


class SensorHealthMonitor:
    def __init__(self):
        self.sensor_stats: Dict[str, Dict] = defaultdict(lambda: {
            'count': 0,
            'errors': 0,
            'last_timestamp': 0,
            'latencies': deque(maxlen=100),
        })

    def record_measurement(self, sensor_id: str, latency: float, success: bool):
        stats = self.sensor_stats[sensor_id]
        stats['count'] += 1
        stats['last_timestamp'] = time.time()
        stats['latencies'].append(latency)
        if not success:
            stats['errors'] += 1

    def get_health(self, sensor_id: str) -> Dict:
        stats = self.sensor_stats[sensor_id]
        if stats['count'] == 0:
            return {'healthy': False, 'reason': 'no_data'}

        error_rate = stats['errors'] / stats['count']
        avg_latency = np.mean(stats['latencies']) if stats['latencies'] else 0

        healthy = error_rate < 0.05 and avg_latency < 0.1
        return {
            'healthy': healthy,
            'error_rate': error_rate,
            'avg_latency': avg_latency,
            'total_count': stats['count'],
        }

    def get_all_health(self) -> Dict[str, Dict]:
        return {sid: self.get_health(sid) for sid in self.sensor_stats}
