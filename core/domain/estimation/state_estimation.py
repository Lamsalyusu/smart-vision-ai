from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict, deque
import copy


class StateEstimationMethod(Enum):
    KALMAN = "kalman"
    EXTENDED_KALMAN = "extended_kalman"
    UNSCENTED_KALMAN = "unscented_kalman"
    PARTICLE_FILTER = "particle_filter"
    FACTOR_GRAPH = "factor_graph"
    GRAPH_OPTIMIZATION = "graph_optimization"
    COMPLEMENTARY = "complementary"
    MAHONY = "mahony"
    MADGWICK = "madgwick"


@dataclass(frozen=True, slots=True)
class StateEstimate:
    timestamp: float
    pose: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray
    bias_accel: np.ndarray
    bias_gyro: np.ndarray
    covariance_diag: np.ndarray


@dataclass(frozen=True, slots=True)
class IMUMeasurement:
    accelerometer: np.ndarray
    gyroscope: np.ndarray
    timestamp: float
    temperature: float = 0.0


@dataclass(frozen=True, slots=True)
class SensorMeasurement:
    sensor_id: str
    measurement: np.ndarray
    covariance: np.ndarray
    timestamp: float


class BaseStateEstimator(ABC):
    @abstractmethod
    def predict(self, dt: float, imu_data: Optional[IMUMeasurement] = None) -> StateEstimate:
        pass

    @abstractmethod
    def update(self, measurement: SensorMeasurement) -> StateEstimate:
        pass

    @abstractmethod
    def get_state(self) -> StateEstimate:
        pass


class KalmanFilter:
    def __init__(
        self,
        state_dim: int,
        measurement_dim: int,
        process_noise: np.ndarray,
        measurement_noise: np.ndarray
    ):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.F = np.eye(state_dim)
        self.H = np.zeros((measurement_dim, state_dim))
        self.Q = process_noise
        self.R = measurement_noise
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim)
        self.I = np.eye(state_dim)

    def set_dynamics(self, F: np.ndarray, Q: np.ndarray):
        self.F = F
        self.Q = Q

    def set_measurement_model(self, H: np.ndarray, R: np.ndarray):
        self.H = H
        self.R = R

    def predict(self, dt: float, u: np.ndarray = None):
        self.x = self.F @ self.x
        if u is not None:
            self.x += u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray, R: np.ndarray = None, H: np.ndarray = None):
        R = R if R is not None else self.R
        H = H if H is not None else self.H

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (self.I - K @ H) @ self.P

    def get_state(self) -> StateEstimate:
        return StateEstimate(
            timestamp=time.time(),
            pose=self.x[:3] if len(self.x) >= 3 else np.zeros(3),
            velocity=self.x[3:6] if len(self.x) >= 6 else np.zeros(3),
            covariance=self.P,
            bias_accel=np.zeros(3),
            bias_gyro=np.zeros(3),
            covariance_diag=np.diag(self.P)
        )


class ExtendedKalmanFilter:
    def __init__(
        self,
        state_dim: int,
        process_noise: np.ndarray,
        measurement_models: Dict[str, Tuple[Callable, Callable]]
    ):
        self.state_dim = state_dim
        self.process_noise = process_noise
        self.measurement_models = measurement_models
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim)
        self.I = np.eye(state_dim)

    def predict(self, dt: float, u: np.ndarray = None):
        self.x = self._motion_model(self.x, dt, u)
        F = self._motion_jacobian(self.x, dt, u)
        self.P = F @ self.P @ F.T + self.process_noise

    def update(self, measurement: np.ndarray, sensor_id: str):
        if sensor_id not in self.measurement_models:
            return

        h_func, H_func = self.measurement_models[sensor_id]
        z = measurement
        R = np.eye(len(z)) * 0.01

        z_pred = h_func(self.x)
        H = H_func(self.x)

        y = z - z_pred
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (self.I - K @ H) @ self.P

    def _motion_model(self, x: np.ndarray, dt: float, u: np.ndarray = None) -> np.ndarray:
        new_x = x.copy()
        if len(x) >= 6:
            new_x[:3] += x[3:6] * dt
        if u is not None and len(x) >= 6:
            new_x[3:6] += u[:3] * dt
        return new_x

    def _motion_jacobian(self, x: np.ndarray, dt: float, u: np.ndarray = None) -> np.ndarray:
        F = np.eye(self.state_dim)
        if self.state_dim >= 6:
            F[:3, 3:6] = np.eye(3) * dt
        return F

    def get_state(self) -> StateEstimate:
        return StateEstimate(
            timestamp=time.time(),
            pose=self.x[:3] if len(self.x) >= 3 else np.zeros(3),
            velocity=self.x[3:6] if len(self.x) >= 6 else np.zeros(3),
            covariance=self.P,
            bias_accel=np.zeros(3),
            bias_gyro=np.zeros(3),
            covariance_diag=np.diag(self.P)
        )


class UnscentedKalmanFilter:
    def __init__(
        self,
        state_dim: int,
        process_noise: np.ndarray,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0
    ):
        self.state_dim = state_dim
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

        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim)

    def _generate_sigma_points(self) -> np.ndarray:
        sigma_points = np.zeros((self.n_sigma, self.state_dim))
        sigma_points[0] = self.x

        sqrt_P = np.linalg.cholesky(self.P + 1e-6 * np.eye(self.state_dim))
        scale = math.sqrt(self.state_dim + self.lambda_)

        for i in range(self.state_dim):
            sigma_points[i + 1] = self.x + scale * sqrt_P[:, i]
            sigma_points[i + 1 + self.state_dim] = self.x - scale * sqrt_P[:, i]

        return sigma_points

    def predict(self, dt: float, u: np.ndarray = None):
        sigma_points = self._generate_sigma_points()

        predicted_sigma = np.zeros_like(sigma_points)
        for i in range(self.n_sigma):
            predicted_sigma[i] = self._motion_model(sigma_points[i], dt, u)

        self.x = np.sum(self.Wm[:, np.newaxis] * predicted_sigma, axis=0)

        self.P = np.zeros((self.state_dim, self.state_dim))
        for i in range(self.n_sigma):
            diff = predicted_sigma[i] - self.x
            self.P += self.Wc[i] * np.outer(diff, diff)

        self.P += self.process_noise

    def update(self, measurement: np.ndarray, h_func: Callable, R: np.ndarray):
        sigma_points = self._generate_sigma_points()
        meas_dim = len(measurement)
        predicted_meas = np.zeros((self.n_sigma, meas_dim))
        for i in range(self.n_sigma):
            predicted_meas[i] = h_func(sigma_points[i])

        z_pred = np.sum(self.Wm[:, np.newaxis] * predicted_meas, axis=0)

        S = np.zeros((meas_dim, meas_dim))
        Pxz = np.zeros((self.state_dim, meas_dim))
        for i in range(self.n_sigma):
            diff_x = sigma_points[i] - self.x
            diff_z = predicted_meas[i] - z_pred
            S += self.Wc[i] * np.outer(diff_z, diff_z)
            Pxz += self.Wc[i] * np.outer(diff_x, diff_z)

        S += R
        K = Pxz @ np.linalg.inv(S)

        y = measurement - z_pred
        self.x = self.x + K @ y
        self.P = self.P - K @ S @ K.T

    def _motion_model(self, x: np.ndarray, dt: float, u: np.ndarray = None) -> np.ndarray:
        new_x = x.copy()
        if len(x) >= 6:
            new_x[:3] += x[3:6] * dt
        if u is not None and len(x) >= 6:
            new_x[3:6] += u[:3] * dt
        return new_x

    def get_state(self) -> StateEstimate:
        return StateEstimate(
            timestamp=time.time(),
            pose=self.x[:3] if len(self.x) >= 3 else np.zeros(3),
            velocity=self.x[3:6] if len(self.x) >= 6 else np.zeros(3),
            covariance=self.P,
            bias_accel=np.zeros(3),
            bias_gyro=np.zeros(3),
            covariance_diag=np.diag(self.P)
        )


class ParticleFilter:
    def __init__(
        self,
        state_dim: int,
        num_particles: int = 1000,
        process_noise: np.ndarray = None,
        measurement_models: Dict[str, Tuple[Callable, Callable]] = None
    ):
        self.state_dim = state_dim
        self.num_particles = num_particles
        self.process_noise = process_noise if process_noise is not None else np.eye(state_dim) * 0.01
        self.measurement_models = measurement_models or {}

        self.particles = np.random.randn(num_particles, state_dim) * 0.1
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, dt: float, u: np.ndarray = None):
        for i in range(self.num_particles):
            self.particles[i] = self._motion_model(self.particles[i], dt, u)
            self.particles[i] += np.random.multivariate_normal(np.zeros(self.state_dim), self.process_noise)

    def update(self, measurement: np.ndarray, sensor_id: str):
        if sensor_id not in self.measurement_models:
            return

        h_func = self.measurement_models[sensor_id][0]
        z = measurement
        R = np.eye(len(z)) * 0.01

        for i in range(self.num_particles):
            z_pred = h_func(self.particles[i])
            diff = z - z_pred
            self.weights[i] *= math.exp(-0.5 * diff @ np.linalg.inv(R) @ diff)

        self.weights += 1e-10
        self.weights /= np.sum(self.weights)

        if 1.0 / np.sum(self.weights**2) < self.num_particles / 2:
            self._resample()

    def _resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def _motion_model(self, x: np.ndarray, dt: float, u: np.ndarray = None) -> np.ndarray:
        new_x = x.copy()
        if len(x) >= 6:
            new_x[:3] += x[3:6] * dt
        if u is not None and len(x) >= 6:
            new_x[3:6] += u[:3] * dt
        return new_x

    def get_state(self) -> StateEstimate:
        mean_state = np.average(self.particles, axis=0, weights=self.weights)
        cov = np.cov(self.particles.T, aweights=self.weights)

        return StateEstimate(
            timestamp=time.time(),
            pose=mean_state[:3] if len(mean_state) >= 3 else np.zeros(3),
            velocity=mean_state[3:6] if len(mean_state) >= 6 else np.zeros(3),
            covariance=cov,
            bias_accel=np.zeros(3),
            bias_gyro=np.zeros(3),
            covariance_diag=np.diag(cov)
        )


class IMUStateEstimator:
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


class StateEstimationPipeline:
    def __init__(
        self,
        estimator_type: StateEstimationMethod = StateEstimationMethod.EXTENDED_KALMAN,
        state_dim: int = 15
    ):
        self.estimator_type = estimator_type
        self.state_dim = state_dim
        self.estimators: Dict[str, BaseStateEstimator] = {}
        self.fused_state: Optional[StateEstimate] = None

    def add_estimator(
        self,
        name: str,
        estimator: BaseStateEstimator
    ):
        self.estimators[name] = estimator

    def predict(self, dt: float, imu_data: Optional[IMUMeasurement] = None):
        for est in self.estimators.values():
            est.predict(dt, imu_data)

    def update(self, measurements: List[SensorMeasurement]):
        for meas in measurements:
            if meas.sensor_id in self.estimators:
                self.estimators[meas.sensor_id].update(meas)

        self.fused_state = self._fuse_estimates()

    def _fuse_estimates(self) -> StateEstimate:
        if not self.estimators:
            return StateEstimate(
                timestamp=time.time(),
                pose=np.zeros(3),
                velocity=np.zeros(3),
                covariance=np.eye(15),
                bias_accel=np.zeros(3),
                bias_gyro=np.zeros(3),
                covariance_diag=np.ones(15)
            )

        poses = []
        velocities = []
        covariances = []

        for est in self.estimators.values():
            state = est.get_state()
            poses.append(state.pose)
            velocities.append(state.velocity)
            covariances.append(state.covariance)

        fused_pose = np.mean(poses, axis=0) if poses else np.zeros(3)
        fused_vel = np.mean(velocities, axis=0) if velocities else np.zeros(3)

        try:
            fused_cov = np.mean(covariances, axis=0) if covariances else np.eye(15)
        except:
            fused_cov = np.eye(15)

        return StateEstimate(
            timestamp=time.time(),
            pose=fused_pose,
            velocity=fused_vel,
            covariance=fused_cov,
            bias_accel=np.zeros(3),
            bias_gyro=np.zeros(3),
            covariance_diag=np.diag(fused_cov)
        )

    def get_fused_state(self) -> Optional[StateEstimate]:
        return self.fused_state


class StateEstimationMetrics:
    @staticmethod
    def compute_nees(
        estimates: List[StateEstimate],
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


class StateEstimationVisualizer:
    @staticmethod
    def draw_trajectory(
        trajectory: List[np.ndarray],
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if len(trajectory) < 2:
            return vis

        points = []
        for pose in trajectory:
            t = pose[:3, 3]
            x = int(t[0] * 10 + image_size[0] // 2)
            y = int(-t[1] * 10 + image_size[1] // 2)
            if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                points.append((x, y))

        for i in range(len(points) - 1):
            cv2.line(vis, points[i], points[i+1], (255, 255, 255), 1)

        return vis

    @staticmethod
    def draw_covariance_ellipse(
        covariance: np.ndarray,
        center: np.ndarray,
        image_size: Tuple[int, int] = (800, 600),
        scale: float = 100.0,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if covariance.shape[0] < 2:
            return vis

        cov_2d = covariance[:2, :2]
        eigenvals, eigenvecs = np.linalg.eigh(cov_2d)

        angle = math.atan2(eigenvecs[1, 0], eigenvecs[0, 0])
        width = 2 * math.sqrt(eigenvals[0]) * scale
        height = 2 * math.sqrt(eigenvals[1]) * scale

        center = (int(center[0] * scale + image_size[0] // 2),
                  int(image_size[1] // 2 - center[1] * scale))

        cv2.ellipse(vis, center, (int(width), int(height)),
                   int(angle * 180 / math.pi), 0, 360, (0, 255, 0), 2)

        return vis


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet