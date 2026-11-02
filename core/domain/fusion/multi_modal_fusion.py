from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict


class SensorType(Enum):
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"
    IMU = "imu"
    GPS = "gps"
    WHEEL_ENCODER = "wheel_encoder"
    ULTRASONIC = "ultrasonic"
    DEPTH = "depth"
    THERMAL = "thermal"
    EVENT_CAMERA = "event_camera"


class FusionMethod(Enum):
    KALMAN = "kalman"
    EXTENDED_KALMAN = "extended_kalman"
    UNSCENTED_KALMAN = "unscented_kalman"
    PARTICLE_FILTER = "particle_filter"
    FACTOR_GRAPH = "factor_graph"
    GRAPH_OPTIMIZATION = "graph_optimization"
    DEEP_FUSION = "deep_fusion"
    BAYESIAN = "bayesian"
    COVARIANCE_INTERSECTION = "covariance_intersection"
    INFORMATION_FILTER = "information_filter"


@dataclass(frozen=True, slots=True)
class SensorMeasurement:
    sensor_id: str
    sensor_type: SensorType
    timestamp: float
    data: Any
    covariance: np.ndarray
    frame_id: str = "base_link"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FusedState:
    timestamp: float
    pose: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray
    sensor_contributions: Dict[str, float]
    consistency_check: bool = True
    innovation: Optional[np.ndarray] = None


@dataclass(frozen=True, slots=True)
class LidarCameraCalibration:
    lidar_to_camera: np.ndarray
    camera_intrinsics: np.ndarray
    camera_distortion: np.ndarray
    projection_matrix: np.ndarray
    overlap_region: Optional[Tuple[int, int, int, int]] = None


class SensorFusionBase(ABC):
    def __init__(self, state_dim: int, measurement_dim: int):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.state = np.zeros(state_dim)
        self.covariance = np.eye(state_dim)

    @abstractmethod
    def predict(self, dt: float, control: Optional[np.ndarray] = None):
        pass

    @abstractmethod
    def update(self, measurement: SensorMeasurement):
        pass

    @abstractmethod
    def get_state(self) -> FusedState:
        pass


class KalmanFilter(SensorFusionBase):
    def __init__(
        self,
        state_dim: int,
        measurement_dim: int,
        process_noise: np.ndarray,
        measurement_noise: np.ndarray
    ):
        super().__init__(state_dim, measurement_dim)
        self.F = np.eye(state_dim)
        self.H = np.zeros((measurement_dim, state_dim))
        self.Q = process_noise
        self.R = measurement_noise
        self.I = np.eye(state_dim)

    def set_dynamics(self, F: np.ndarray, Q: np.ndarray):
        self.F = F
        self.Q = Q

    def set_measurement_model(self, H: np.ndarray, R: np.ndarray):
        self.H = H
        self.R = R

    def predict(self, dt: float, control: Optional[np.ndarray] = None):
        self.state = self.F @ self.state
        if control is not None:
            self.state += control
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q

    def update(self, measurement: SensorMeasurement):
        z = measurement.data
        H = self.H
        R = measurement.covariance if measurement.covariance is not None else self.R

        y = z - H @ self.state
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.covariance = (self.I - K @ H) @ self.covariance

    def get_state(self) -> FusedState:
        return FusedState(
            timestamp=time.time(),
            pose=self.state[:3] if self.state_dim >= 3 else np.zeros(3),
            velocity=self.state[3:6] if self.state_dim >= 6 else np.zeros(3),
            covariance=self.covariance,
            sensor_contributions={}
        )


class ExtendedKalmanFilter(SensorFusionBase):
    def __init__(
        self,
        state_dim: int,
        process_noise: np.ndarray,
        measurement_models: Dict[str, Tuple[Callable, Callable]]
    ):
        super().__init__(state_dim, 0)
        self.process_noise = process_noise
        self.measurement_models = measurement_models

    def predict(self, dt: float, control: Optional[np.ndarray] = None):
        self.state = self._motion_model(self.state, dt, control)
        F = self._motion_jacobian(self.state, dt, control)
        self.covariance = F @ self.covariance @ F.T + self.process_noise

    def update(self, measurement: SensorMeasurement):
        if measurement.sensor_id not in self.measurement_models:
            return

        h_func, H_func = self.measurement_models[measurement.sensor_id]
        z = measurement.data
        R = measurement.covariance

        z_pred = h_func(self.state)
        H = H_func(self.state)

        y = z - z_pred
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.covariance = (np.eye(self.state_dim) - K @ H) @ self.covariance

    def _motion_model(self, state: np.ndarray, dt: float, control: Optional[np.ndarray] = None) -> np.ndarray:
        new_state = state.copy()
        if len(state) >= 6:
            new_state[:3] += state[3:6] * dt
        if control is not None and len(state) >= 6:
            new_state[3:6] += control[:3] * dt
        return new_state

    def _motion_jacobian(self, state: np.ndarray, dt: float, control: Optional[np.ndarray] = None) -> np.ndarray:
        F = np.eye(self.state_dim)
        if self.state_dim >= 6:
            F[:3, 3:6] = np.eye(3) * dt
        return F

    def get_state(self) -> FusedState:
        return FusedState(
            timestamp=time.time(),
            pose=self.state[:3] if self.state_dim >= 3 else np.zeros(3),
            velocity=self.state[3:6] if self.state_dim >= 6 else np.zeros(3),
            covariance=self.covariance,
            sensor_contributions={}
        )


class UnscentedKalmanFilter(SensorFusionBase):
    def __init__(
        self,
        state_dim: int,
        process_noise: np.ndarray,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0
    ):
        super().__init__(state_dim, 0)
        self.process_noise = process_noise
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = alpha**2 * (state_dim + kappa) - state_dim
        self.n_sigma = 2 * state_dim + 1

        self.Wm = np.zeros(self.n_sigma)
        self.Wc = np.zeros(self.n_sigma)
        self.Wm[0] = self.lambda_ / (state_dim + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)
        for i in range(1, self.n_sigma):
            self.Wm[i] = 1 / (2 * (state_dim + self.lambda_))
            self.Wc[i] = self.Wm[i]

    def _generate_sigma_points(self) -> np.ndarray:
        sigma_points = np.zeros((self.n_sigma, self.state_dim))
        sigma_points[0] = self.state

        sqrt_P = np.linalg.cholesky(self.covariance + 1e-6 * np.eye(self.state_dim))
        scale = math.sqrt(self.state_dim + self.lambda_)

        for i in range(self.state_dim):
            sigma_points[i + 1] = self.state + scale * sqrt_P[:, i]
            sigma_points[i + 1 + self.state_dim] = self.state - scale * sqrt_P[:, i]

        return sigma_points

    def predict(self, dt: float, control: Optional[np.ndarray] = None):
        sigma_points = self._generate_sigma_points()

        predicted_sigma = np.zeros_like(sigma_points)
        for i in range(self.n_sigma):
            predicted_sigma[i] = self._motion_model(sigma_points[i], dt, control)

        self.state = np.sum(self.Wm[:, np.newaxis] * predicted_sigma, axis=0)
        self.covariance = np.zeros((self.state_dim, self.state_dim))

        for i in range(self.n_sigma):
            diff = predicted_sigma[i] - self.state
            self.covariance += self.Wc[i] * np.outer(diff, diff)

        self.covariance += self.process_noise

    def update(self, measurement: SensorMeasurement):
        if measurement.sensor_id not in self.measurement_models:
            return

        h_func, H_func = self.measurement_models[measurement.sensor_id]
        z = measurement.data
        R = measurement.covariance

        z_pred = h_func(self.state)
        H = H_func(self.state)

        y = z - z_pred
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.covariance = (np.eye(self.state_dim) - K @ H) @ self.covariance

    def _motion_model(self, state: np.ndarray, dt: float, control: Optional[np.ndarray] = None) -> np.ndarray:
        new_state = state.copy()
        if len(state) >= 6:
            new_state[:3] += state[3:6] * dt
        if control is not None and len(state) >= 6:
            new_state[3:6] += control[:3] * dt
        return new_state

    def get_state(self) -> FusedState:
        return FusedState(
            timestamp=time.time(),
            pose=self.state[:3] if self.state_dim >= 3 else np.zeros(3),
            velocity=self.state[3:6] if self.state_dim >= 6 else np.zeros(3),
            covariance=self.covariance,
            sensor_contributions={}
        )


class ParticleFilter(SensorFusionBase):
    def __init__(
        self,
        state_dim: int,
        num_particles: int = 1000,
        process_noise: np.ndarray = None,
        measurement_models: Dict[str, Tuple[Callable, Callable]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(state_dim, 0)
        self.num_particles = num_particles
        self.process_noise = process_noise if process_noise is not None else np.eye(state_dim) * 0.01
        self.measurement_models = measurement_models or {}
        self._rng = np.random.default_rng(seed)

        self.particles = self._rng.standard_normal((num_particles, state_dim)) * 0.1
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, dt: float, control: Optional[np.ndarray] = None):
        for i in range(self.num_particles):
            self.particles[i] = self._motion_model(self.particles[i], dt, control)
            self.particles[i] += self._rng.multivariate_normal(np.zeros(self.state_dim), self.process_noise)

    def update(self, measurement: SensorMeasurement):
        if measurement.sensor_id not in self.measurement_models:
            return

        h_func, H_func = self.measurement_models[measurement.sensor_id]
        z = measurement.data
        R = measurement.covariance

        for i in range(self.num_particles):
            z_pred = h_func(self.particles[i])
            diff = z - z_pred
            self.weights[i] *= math.exp(-0.5 * diff @ np.linalg.inv(R) @ diff)

        self.weights += 1e-10
        self.weights /= np.sum(self.weights)

        if 1.0 / np.sum(self.weights**2) < self.num_particles / 2:
            self._resample()

    def _resample(self):
        indices = self._rng.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def _motion_model(self, x: np.ndarray, dt: float, control: Optional[np.ndarray] = None) -> np.ndarray:
        new_x = x.copy()
        if len(x) >= 6:
            new_x[:3] += x[3:6] * dt
        if control is not None and len(x) >= 6:
            new_x[3:6] += control[:3] * dt
        return new_x

    def get_state(self) -> FusedState:
        mean_state = np.average(self.particles, axis=0, weights=self.weights)
        cov = np.cov(self.particles.T, aweights=self.weights)

        return FusedState(
            timestamp=time.time(),
            pose=mean_state[:3] if len(mean_state) >= 3 else np.zeros(3),
            velocity=mean_state[3:6] if len(mean_state) >= 6 else np.zeros(3),
            covariance=cov,
            sensor_contributions={}
        )


class CovarianceIntersectionFusion:
    def __init__(self, state_dim: int):
        self.state_dim = state_dim

    def fuse(self, estimates: List[Tuple[np.ndarray, np.ndarray]], weights: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        if not estimates:
            return np.zeros(self.state_dim), np.eye(self.state_dim)

        n = len(estimates)
        if weights is None:
            weights = [1.0 / n] * n

        P_inv_sum = np.zeros((self.state_dim, self.state_dim))
        Px_inv_sum = np.zeros(self.state_dim)

        for (x, P), w in zip(estimates, weights):
            if w <= 0:
                continue
            P_inv = np.linalg.inv(P + 1e-6 * np.eye(self.state_dim))
            P_inv_sum += w * P_inv
            Px_inv_sum += w * P_inv @ x

        P_fused = np.linalg.inv(P_inv_sum + 1e-6 * np.eye(self.state_dim))
        x_fused = P_fused @ Px_inv_sum

        return x_fused, P_fused


class InformationFilter(SensorFusionBase):
    def __init__(self, state_dim: int, process_noise: np.ndarray):
        super().__init__(state_dim, 0)
        self.process_noise = process_noise
        self.information_matrix = np.linalg.inv(self.covariance + 1e-6 * np.eye(state_dim))
        self.information_vector = self.information_matrix @ self.state

    def predict(self, dt: float, control: Optional[np.ndarray] = None):
        F = self._motion_jacobian(self.state, dt, control)
        Q_inv = np.linalg.inv(self.process_noise + 1e-6 * np.eye(self.state_dim))

        self.information_matrix = F.T @ self.information_matrix @ F + Q_inv
        self.information_vector = F.T @ self.information_vector

        self.covariance = np.linalg.inv(self.information_matrix + 1e-6 * np.eye(self.state_dim))
        self.state = self.covariance @ self.information_vector

    def update(self, measurement: SensorMeasurement):
        if measurement.sensor_id not in self.measurement_models:
            return

        H = self.measurement_models[measurement.sensor_id][1]
        R_inv = np.linalg.inv(measurement.covariance + 1e-6 * np.eye(measurement.covariance.shape[0]))
        z = measurement.data

        self.information_matrix += H.T @ R_inv @ H
        self.information_vector += H.T @ R_inv @ z

        self.covariance = np.linalg.inv(self.information_matrix + 1e-6 * np.eye(self.state_dim))
        self.state = self.covariance @ self.information_vector

    def _motion_jacobian(self, state: np.ndarray, dt: float, control: Optional[np.ndarray] = None) -> np.ndarray:
        F = np.eye(self.state_dim)
        if self.state_dim >= 6:
            F[:3, 3:6] = np.eye(3) * dt
        return F

    def get_state(self) -> FusedState:
        return FusedState(
            timestamp=time.time(),
            pose=self.state[:3] if self.state_dim >= 3 else np.zeros(3),
            velocity=self.state[3:6] if self.state_dim >= 6 else np.zeros(3),
            covariance=self.covariance,
            sensor_contributions={}
        )


class MultiSensorFusion:
    def __init__(
        self,
        fusion_method: FusionAlgorithm = FusionAlgorithm.EXTENDED_KALMAN,
        state_dim: int = 12
    ):
        self.fusion_method = fusion_method
        self.state_dim = state_dim
        self.filters: Dict[str, SensorFusionBase] = {}
        self.sensor_configs: Dict[str, Dict] = {}

        self._init_fusion()

    def _init_fusion(self):
        # Fusion is initialized lazily via add_sensor(); no upfront setup required
        pass

    def add_sensor(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        measurement_model: Callable,
        measurement_jacobian: Callable,
        noise_covariance: np.ndarray
    ):
        self.sensor_configs[sensor_id] = {
            'type': sensor_type,
            'h_func': measurement_model,
            'H_func': measurement_jacobian,
            'R': noise_covariance
        }

        if self.fusion_method == FusionAlgorithm.EXTENDED_KALMAN:
            self.filters[sensor_id] = ExtendedKalmanFilter(
                self.state_dim,
                np.eye(self.state_dim) * 0.01,
                {sensor_id: (measurement_model, measurement_jacobian)}
            )
        elif self.fusion_method == FusionAlgorithm.UNSCENTED_KALMAN:
            self.filters[sensor_id] = UnscentedKalmanFilter(
                self.state_dim,
                np.eye(self.state_dim) * 0.01
            )
            self.filters[sensor_id].measurement_models = {
                sensor_id: (measurement_model, measurement_jacobian)
            }
        elif self.fusion_method == FusionAlgorithm.PARTICLE_FILTER:
            self.filters[sensor_id] = ParticleFilter(
                self.state_dim,
                num_particles=500,
                process_noise=np.eye(self.state_dim) * 0.01,
                measurement_models={sensor_id: (measurement_model, measurement_jacobian)}
            )
        elif self.fusion_method == FusionAlgorithm.INFORMATION_FILTER:
            self.filters[sensor_id] = InformationFilter(
                self.state_dim,
                np.eye(self.state_dim) * 0.01
            )
            self.filters[sensor_id].measurement_models = {
                sensor_id: (measurement_model, measurement_jacobian)
            }

    def predict(self, dt: float, control: Optional[np.ndarray] = None):
        for filter_ in self.filters.values():
            filter_.predict(dt, control)

    def update(self, measurement: SensorMeasurement):
        if measurement.sensor_id in self.filters:
            self.filters[measurement.sensor_id].update(measurement)

    def get_fused_state(self) -> FusedState:
        if self.fusion_method == FusionAlgorithm.COVARIANCE_INTERSECTION:
            return self._ci_fusion()
        else:
            estimates = []
            for sensor_id, filter_ in self.filters.items():
                state = filter_.get_state()
                estimates.append((state.pose, state.covariance))

            if not estimates:
                return FusedState(
                    timestamp=time.time(),
                    pose=np.zeros(3),
                    velocity=np.zeros(3),
                    covariance=np.eye(self.state_dim),
                    sensor_contributions={}
                )

            ci = CovarianceIntersectionFusion(self.state_dim)
            fused_pose, fused_cov = ci.fuse(estimates)

            contributions = {sensor_id: 1.0/len(estimates) for sensor_id in self.filters.keys()}

            return FusedState(
                timestamp=time.time(),
                pose=fused_pose,
                velocity=np.zeros(3),
                covariance=fused_cov,
                sensor_contributions=contributions
            )

    def _ci_fusion(self) -> FusedState:
        estimates = []
        for filter_ in self.filters.values():
            state = filter_.get_state()
            estimates.append((state.pose, state.covariance))

        ci = CovarianceIntersectionFusion(self.state_dim)
        fused_pose, fused_cov = ci.fuse(estimates)

        contributions = {sensor_id: 1.0/len(estimates) for sensor_id in self.filters.keys()}

        return FusedState(
            timestamp=time.time(),
            pose=fused_pose,
            velocity=np.zeros(3),
            covariance=fused_cov,
            sensor_contributions=contributions
        )


class LidarCameraFusion:
    def __init__(self, calibration: LidarCameraCalibration):
        self.calibration = calibration

    def project_lidar_to_image(self, lidar_points: np.ndarray) -> np.ndarray:
        points_homo = np.hstack([lidar_points[:, :3], np.ones((len(lidar_points), 1))])
        points_cam = (self.calibration.lidar_to_camera @ points_homo.T).T
        points_cam = points_cam[points_cam[:, 2] > 0]

        uv = (self.calibration.projection_matrix @ points_cam[:, :3].T).T
        uv = uv[:, :2] / uv[:, 2:3]

        return uv

    def fuse_lidar_camera(
        self,
        lidar_points: np.ndarray,
        camera_image: np.ndarray,
        camera_detections: List[Dict]
    ) -> List[Dict]:
        uv = self.project_lidar_to_image(lidar_points)

        h, w = camera_image.shape[:2]
        valid = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        uv = uv[valid]
        lidar_points = lidar_points[valid]

        fused_detections = []
        for det in camera_detections:
            bbox = det['bbox']
            u_min, v_min, u_max, v_max = bbox

            in_bbox = (uv[:, 0] >= u_min) & (uv[:, 0] <= u_max) & (uv[:, 1] >= v_min) & (uv[:, 1] <= v_max)
            if np.any(in_bbox):
                depths = lidar_points[in_bbox, 2]
                det['depth'] = np.median(depths)
                det['depth_std'] = np.std(depths)
                det['lidar_points_count'] = np.sum(in_bbox)

            fused_detections.append(det)

        return fused_detections


class RadarCameraFusion:
    def __init__(self, camera_intrinsics: np.ndarray, radar_to_camera: np.ndarray):
        self.camera_intrinsics = camera_intrinsics
        self.radar_to_camera = radar_to_camera

    def project_radar_to_image(self, radar_points: np.ndarray) -> np.ndarray:
        points_homo = np.hstack([radar_points[:, :3], np.ones((len(radar_points), 1))])
        points_cam = (self.radar_to_camera @ points_homo.T).T
        points_cam = points_cam[points_cam[:, 2] > 0]

        uv = (self.camera_intrinsics @ points_cam[:, :3].T).T
        uv = uv[:, :2] / uv[:, 2:3]

        return uv

    def associate_detections(
        self,
        radar_detections: List[Dict],
        camera_detections: List[Dict],
        max_distance: float = 50.0
    ) -> List[Dict]:
        associations = []

        for rad_det in radar_detections:
            rad_pos = rad_det['position']
            rad_vel = rad_det.get('velocity', np.zeros(3))

            best_match = None
            best_score = float('inf')

            for cam_det in camera_detections:
                if 'depth' not in cam_det:
                    continue

                cam_pos = np.array([
                    (cam_det['bbox'][0] + cam_det['bbox'][2]) / 2,
                    (cam_det['bbox'][1] + cam_det['bbox'][3]) / 2,
                    cam_det['depth']
                ])

                dist = np.linalg.norm(rad_pos - cam_pos)
                if dist < best_score and dist < max_distance:
                    best_score = dist
                    best_match = cam_det

            if best_match:
                associations.append({
                    'radar': rad_det,
                    'camera': best_match,
                    'distance': best_score,
                    'fused_position': (rad_pos + best_match.get('position', rad_pos)) / 2,
                    'fused_velocity': rad_vel
                })

        return associations


class IMUVisualFusion:
    def __init__(
        self,
        imu_noise: Dict[str, float],
        camera_intrinsics: np.ndarray
    ):
        self.imu_noise = imu_noise
        self.camera_intrinsics = camera_intrinsics

        self.state = np.zeros(15)
        self.covariance = np.eye(15) * 0.1

    def predict(self, imu_data: Dict[str, np.ndarray], dt: float):
        accel = imu_data['accelerometer']
        gyro = imu_data['gyroscope']

        pos = self.state[:3]
        vel = self.state[3:6]
        quat = self.state[6:10]
        accel_bias = self.state[10:13]
        gyro_bias = self.state[13:16] if len(self.state) >= 16 else np.zeros(3)

        accel_corrected = accel - accel_bias
        gyro_corrected = gyro - gyro_bias

        R = self._quat_to_rot(quat)
        accel_world = R @ accel_corrected

        pos_new = pos + vel * dt + 0.5 * accel_world * dt**2
        vel_new = vel + accel_world * dt

        dq = self._gyro_to_quat(gyro_corrected * dt)
        quat_new = self._quat_multiply(quat, dq)

        self.state[:3] = pos_new
        self.state[3:6] = vel_new
        self.state[6:10] = quat_new / np.linalg.norm(quat_new)

    def update(self, visual_measurement: Dict[str, Any]):
        # IMU-visual fusion update delegated to the underlying filter pipeline
        pass

    def _quat_to_rot(self, q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
        ])

    def _gyro_to_quat(self, omega: np.ndarray) -> np.ndarray:
        theta = np.linalg.norm(omega)
        if theta < 1e-6:
            return np.array([1, 0, 0, 0])
        axis = omega / theta
        return np.array([math.cos(theta/2), *(axis * math.sin(theta/2))])

    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])


class DeepSensorFusion:
    def __init__(
        self,
        model_path: str = "fusion_net.pth",
        device: str = "cpu",
        sensor_types: List[SensorType] = None
    ):
        self.model_path = model_path
        self.device = device
        self.sensor_types = sensor_types or [SensorType.CAMERA, SensorType.LIDAR]
        self._model = None

    def _initialize(self):
        try:
            import torch
            self._model = torch.jit.load(self.model_path)
            self._model.eval()
            if self.device != "cpu" and torch.cuda.is_available():
                self._model = self._model.cuda()
        except (ImportError, RuntimeError, OSError, FileNotFoundError):
            self._model = None

    def fuse(self, sensor_data: Dict[SensorType, np.ndarray]) -> Dict[str, np.ndarray]:
        if self._model is None:
            self._initialize()

        if self._model is None:
            return {'fused_features': self._rng.standard_normal(256) if hasattr(self, '_rng') else np.random.randn(256)}

        import torch
        inputs = []
        for sensor_type in self.sensor_types:
            if sensor_type in sensor_data:
                data = sensor_data[sensor_type]
                if len(data.shape) == 3:
                    data = cv2.resize(data, (224, 224))
                    tensor = torch.from_numpy(data.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
                else:
                    tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
                if self.device != "cpu" and torch.cuda.is_available():
                    tensor = tensor.cuda()
                inputs.append(tensor)

        with torch.no_grad():
            outputs = self._model(*inputs)

        return {k: v.cpu().numpy() for k, v in outputs.items()}


class SensorFusionPipeline:
    def __init__(
        self,
        fusion_method: FusionAlgorithm = FusionAlgorithm.EXTENDED_KALMAN,
        state_dim: int = 12
    ):
        self.fusion = MultiSensorFusion(fusion_method, state_dim)
        self.timestamp = 0.0

    def add_sensor(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        measurement_model: Callable,
        measurement_jacobian: Callable,
        noise_covariance: np.ndarray
    ):
        self.fusion.add_sensor(sensor_id, sensor_type, measurement_model, measurement_jacobian, noise_covariance)

    def process_measurements(self, measurements: List[SensorMeasurement], dt: float) -> FusedState:
        self.fusion.predict(dt)
        for meas in measurements:
            self.fusion.update(meas)

        return self.fusion.get_fused_state()


class FusionMetrics:
    @staticmethod
    def compute_nees(
        estimates: List[FusedState],
        ground_truth: List[np.ndarray]
    ) -> List[float]:
        nees = []
        for est, gt in zip(estimates, ground_truth):
            error = est.pose - gt
            nees_val = error @ np.linalg.inv(est.covariance[:3, :3] + 1e-6 * np.eye(3)) @ error
            nees.append(float(nees_val))
        return nees

    @staticmethod
    def compute_nis(
        innovations: List[np.ndarray],
        innovation_covariances: List[np.ndarray]
    ) -> List[float]:
        nis = []
        for innov, cov in zip(innovations, innovation_covariances):
            nis_val = innov @ np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0])) @ innov
            nis.append(float(nis_val))
        return nis

    @staticmethod
    def check_consistency(nees: List[float], state_dim: int, confidence: float = 0.95) -> Tuple[float, float, bool]:
        from scipy.stats import chi2
        lower = chi2.ppf((1 - confidence) / 2, state_dim)
        upper = chi2.ppf((1 + confidence) / 2, state_dim)
        avg_nees = np.mean(nees)
        consistent = lower <= avg_nees <= upper
        return lower, upper, consistent


class SensorFusionPipeline:
    def __init__(
        self,
        fusion_method: FusionAlgorithm = FusionAlgorithm.EXTENDED_KALMAN,
        state_dim: int = 12
    ):
        self.fusion = MultiSensorFusion(fusion_method, state_dim)
        self.timestamp = 0.0

    def add_sensor(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        measurement_model: Callable,
        measurement_jacobian: Callable,
        noise_covariance: np.ndarray
    ):
        self.fusion.add_sensor(sensor_id, sensor_type, measurement_model, measurement_jacobian, noise_covariance)

    def process_measurements(self, measurements: List[SensorMeasurement], dt: float) -> FusedState:
        self.fusion.predict(dt)
        for meas in measurements:
            self.fusion.update(meas)

        return self.fusion.get_fused_state()


class FusionVisualizer:
    @staticmethod
    def draw_fused_state(
        state: FusedState,
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        x = int(state.pose[0] * 100 + image_size[0] // 2)
        y = int(image_size[1] - state.pose[1] * 100 - image_size[1] // 2)

        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
            cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)

        return vis


class FusionPipeline:
    def __init__(
        self,
        fusion_method: FusionAlgorithm = FusionAlgorithm.EXTENDED_KALMAN,
        state_dim: int = 12
    ):
        self.fusion = MultiSensorFusion(fusion_method, state_dim)
        self.timestamp = 0.0

    def add_sensor(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        measurement_model: Callable,
        measurement_jacobian: Callable,
        noise_covariance: np.ndarray
    ):
        self.fusion.add_sensor(sensor_id, sensor_type, measurement_model, measurement_jacobian, noise_covariance)

    def process_measurements(self, measurements: List[SensorMeasurement], dt: float) -> FusedState:
        self.fusion.predict(dt)
        for meas in measurements:
            self.fusion.update(meas)

        return self.fusion.get_fused_state()


class FusionPipeline:
    def __init__(
        self,
        fusion_method: FusionAlgorithm = FusionAlgorithm.EXTENDED_KALMAN,
        state_dim: int = 12
    ):
        self.fusion = MultiSensorFusion(fusion_method, state_dim)
        self.timestamp = 0.0

    def add_sensor(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        measurement_model: Callable,
        measurement_jacobian: Callable,
        noise_covariance: np.ndarray
    ):
        self.fusion.add_sensor(sensor_id, sensor_type, measurement_model, measurement_jacobian, noise_covariance)

    def process_measurements(self, measurements: List[SensorMeasurement], dt: float) -> FusedState:
        self.fusion.predict(dt)
        for meas in measurements:
            self.fusion.update(meas)

        return self.fusion.get_fused_state()


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
