from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict


class FusionAlgorithm(Enum):
    KALMAN = "kalman"
    EXTENDED_KALMAN = "extended_kalman"
    UNSCENTED_KALMAN = "unscented_kalman"
    PARTICLE_FILTER = "particle_filter"
    CENTRALIZED = "centralized"
    DISTRIBUTED = "distributed"
    CONSENSUS = "consensus"
    COVARIANCE_INTERSECTION = "covariance_intersection"
    INFORMATION_FILTER = "information_filter"
    FACTOR_GRAPH = "factor_graph"
    GRAPH_OPTIMIZATION = "graph_optimization"
    DEEP_FUSION = "deep_fusion"
    BAYESIAN = "bayesian"


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
class Track:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    sensor_ids: Set[str]
    age: int
    hits: int
    time_since_update: int
    label: Optional[str] = None
    score: float = 0.0


class KalmanFilterBase(ABC):
    def __init__(self, state_dim: int, measurement_dim: int):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim)

    @abstractmethod
    def predict(self, dt: float, u: Optional[np.ndarray] = None):
        pass

    @abstractmethod
    def update(self, z: np.ndarray, R: np.ndarray, H: np.ndarray):
        pass

    @abstractmethod
    def get_state(self) -> FusedState:
        pass


class LinearKalmanFilter(KalmanFilterBase):
    def __init__(
        self,
        state_dim: int,
        measurement_dim: int,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray
    ):
        super().__init__(state_dim, measurement_dim)
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R

    def predict(self, dt: float, u: Optional[np.ndarray] = None):
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
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

    def get_state(self) -> FusedState:
        return FusedState(
            timestamp=time.time(),
            pose=self.x[:3] if self.state_dim >= 3 else np.zeros(3),
            velocity=self.x[3:6] if self.state_dim >= 6 else np.zeros(3),
            covariance=self.P,
            sensor_contributions={}
        )


class ExtendedKalmanFilter(KalmanFilterBase):
    def __init__(
        self,
        state_dim: int,
        process_noise: np.ndarray,
        measurement_models: Dict[str, Tuple[Callable, Callable]]
    ):
        super().__init__(state_dim, 0)
        self.process_noise = process_noise
        self.measurement_models = measurement_models

    def predict(self, dt: float, u: Optional[np.ndarray] = None):
        self.x = self._motion_model(self.x, dt, u)
        F = self._motion_jacobian(self.x, dt, u)
        self.P = F @ self.P @ F.T + self.process_noise

    def update(self, z: np.ndarray, R: np.ndarray, H: np.ndarray = None, sensor_id: str = None):
        if sensor_id and sensor_id in self.measurement_models:
            h_func, H_func = self.measurement_models[sensor_id]
            h = h_func(self.x)
            H = H_func(self.x)
        else:
            h = z
            if H is None:
                raise ValueError("H matrix required if sensor_id not in measurement_models")

        y = z - h
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

    def _motion_model(self, x: np.ndarray, dt: float, u: Optional[np.ndarray]) -> np.ndarray:
        new_x = x.copy()
        if len(x) >= 6:
            new_x[:3] += x[3:6] * dt
        if u is not None and len(x) >= 6:
            new_x[3:6] += u[:3] * dt
        return new_x

    def _motion_jacobian(self, x: np.ndarray, dt: float, u: Optional[np.ndarray]) -> np.ndarray:
        F = np.eye(self.state_dim)
        if self.state_dim >= 6:
            F[:3, 3:6] = np.eye(3) * dt
        return F

    def get_state(self) -> FusedState:
        return FusedState(
            timestamp=time.time(),
            pose=self.x[:3] if self.state_dim >= 3 else np.zeros(3),
            velocity=self.x[3:6] if self.state_dim >= 6 else np.zeros(3),
            covariance=self.P,
            sensor_contributions={}
        )


class UnscentedKalmanFilter(KalmanFilterBase):
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
        sigma_points[0] = self.x

        sqrt_P = np.linalg.cholesky(self.P + 1e-6 * np.eye(self.state_dim))
        scale = math.sqrt(self.state_dim + self.lambda_)

        for i in range(self.state_dim):
            sigma_points[i + 1] = self.x + scale * sqrt_P[:, i]
            sigma_points[i + 1 + self.state_dim] = self.x - scale * sqrt_P[:, i]

        return sigma_points

    def predict(self, dt: float, u: Optional[np.ndarray] = None):
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

    def update(self, z: np.ndarray, R: np.ndarray, h_func: Callable):
        sigma_points = self._generate_sigma_points()
        meas_dim = len(z)
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

        y = z - z_pred
        self.x = self.x + K @ y
        self.P = self.P - K @ S @ K.T

    def _motion_model(self, x: np.ndarray, dt: float, u: Optional[np.ndarray]) -> np.ndarray:
        new_x = x.copy()
        if len(x) >= 6:
            new_x[:3] += x[3:6] * dt
        if u is not None and len(x) >= 6:
            new_x[3:6] += u[:3] * dt
        return new_x

    def get_state(self) -> FusedState:
        return FusedState(
            timestamp=time.time(),
            pose=self.x[:3] if self.state_dim >= 3 else np.zeros(3),
            velocity=self.x[3:6] if self.state_dim >= 6 else np.zeros(3),
            covariance=self.P,
            sensor_contributions={}
        )


class ParticleFilter(KalmanFilterBase):
    def __init__(
        self,
        state_dim: int,
        num_particles: int = 1000,
        process_noise: np.ndarray = None
    ):
        super().__init__(state_dim, 0)
        self.num_particles = num_particles
        self.process_noise = process_noise if process_noise is not None else np.eye(state_dim) * 0.01

        self.particles = np.random.randn(num_particles, state_dim) * 0.1
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, dt: float, u: Optional[np.ndarray] = None):
        for i in range(self.num_particles):
            self.particles[i] = self._motion_model(self.particles[i], dt, u)
            self.particles[i] += np.random.multivariate_normal(np.zeros(self.state_dim), self.process_noise)

    def update(self, z: np.ndarray, R: np.ndarray, h_func: Callable):
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

    def _motion_model(self, x: np.ndarray, dt: float, u: Optional[np.ndarray]) -> np.ndarray:
        new_x = x.copy()
        if len(x) >= 6:
            new_x[:3] += x[3:6] * dt
        if u is not None and len(x) >= 6:
            new_x[3:6] += u[:3] * dt
        return new_x

    def get_state(self) -> FusedState:
        mean_state = np.average(self.particles, axis=0, weights=self.weights)
        cov = np.cov(self.particles.T, aweights=self.weights)

        return FusedState(
            timestamp=time.time(),
            pose=mean_state[:3] if self.state_dim >= 3 else np.zeros(3),
            velocity=mean_state[3:6] if self.state_dim >= 6 else np.zeros(3),
            covariance=cov,
            sensor_contributions={}
        )


class CovarianceIntersection:
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


class InformationFilter(KalmanFilterBase):
    def __init__(self, state_dim: int, process_noise: np.ndarray):
        super().__init__(state_dim, 0)
        self.process_noise = process_noise
        self.Y = np.linalg.inv(self.P + 1e-6 * np.eye(state_dim))
        self.y = self.Y @ self.x

    def predict(self, dt: float, u: Optional[np.ndarray] = None):
        F = self._motion_jacobian(self.x, dt, u)
        Q_inv = np.linalg.inv(self.process_noise + 1e-6 * np.eye(self.state_dim))

        self.Y = F.T @ self.Y @ F + Q_inv
        self.y = F.T @ self.y

        self.P = np.linalg.inv(self.Y + 1e-6 * np.eye(self.state_dim))
        self.x = self.P @ self.y

    def update(self, z: np.ndarray, R: np.ndarray, H: np.ndarray):
        R_inv = np.linalg.inv(R + 1e-6 * np.eye(R.shape[0]))

        self.Y += H.T @ R_inv @ H
        self.y += H.T @ R_inv @ z

        self.P = np.linalg.inv(self.Y + 1e-6 * np.eye(self.state_dim))
        self.x = self.P @ self.y

    def _motion_jacobian(self, x: np.ndarray, dt: float, u: Optional[np.ndarray]) -> np.ndarray:
        F = np.eye(self.state_dim)
        if self.state_dim >= 6:
            F[:3, 3:6] = np.eye(3) * dt
        return F

    def get_state(self) -> FusedState:
        return FusedState(
            timestamp=time.time(),
            pose=self.x[:3] if self.state_dim >= 3 else np.zeros(3),
            velocity=self.x[3:6] if self.state_dim >= 6 else np.zeros(3),
            covariance=self.P,
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
        self.filters: Dict[str, KalmanFilterBase] = {}
        self.sensor_configs: Dict[str, Dict] = {}
        self._init_fusion()

    def _init_fusion(self):
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
                process_noise=np.eye(self.state_dim) * 0.01
            )
            self.filters[sensor_id].measurement_models = {
                sensor_id: (measurement_model, measurement_jacobian)
            }
        elif self.fusion_method == FusionAlgorithm.INFORMATION_FILTER:
            self.filters[sensor_id] = InformationFilter(
                self.state_dim,
                np.eye(self.state_dim) * 0.01
            )
            self.filters[sensor_id].measurement_models = {
                sensor_id: (measurement_model, measurement_jacobian)
            }

    def predict(self, dt: float, u: Optional[np.ndarray] = None):
        for filter_ in self.filters.values():
            filter_.predict(dt, u)

    def update(self, measurement: SensorMeasurement):
        if measurement.sensor_id in self.filters:
            self.filters[measurement.sensor_id].update(
                measurement.data,
                measurement.covariance,
                sensor_id=measurement.sensor_id
            )

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

            ci = CovarianceIntersection(self.state_dim)
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

        ci = CovarianceIntersection(self.state_dim)
        fused_pose, fused_cov = ci.fuse(estimates)

        contributions = {sensor_id: 1.0/len(estimates) for sensor_id in self.filters.keys()}

        return FusedState(
            timestamp=time.time(),
            pose=fused_pose,
            velocity=np.zeros(3),
            covariance=fused_cov,
            sensor_contributions=contributions
        )


class Tracker:
    def __init__(
        self,
        state_dim: int = 6,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        self.state_dim = state_dim
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, Track] = {}
        self.track_id_counter = 0

    def update(self, detections: List[Detection3D]) -> List[Track]:
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id].age += 1
            self.tracks[track_id].time_since_update += 1
            if self.tracks[track_id].age > self.max_age:
                del self.tracks[track_id]

        if not detections:
            return list(self.tracks.values())

        matched = self._associate_detections_to_tracks(detections)
        unmatched_detections = [i for i in range(len(detections)) if i not in [m[1] for m in matched]]
        unmatched_tracks = [tid for tid in self.tracks.keys() if tid not in [m[0] for m in matched]]

        for track_id, det_idx in matched:
            track = self.tracks[track_id]
            det = detections[det_idx]
            track.hits += 1
            track.time_since_update = 0
            track.state = self._update_track_state(track.state, det)
            track.sensor_ids.update(det.metadata.get('sensor_ids', set()))
            track.score = (track.score * track.hits + det.box.score) / (track.hits + 1)

        for det_idx in unmatched_detections:
            det = detections[det_idx]
            self._create_new_track(det)

        return list(self.tracks.values())

    def _associate_detections_to_tracks(self, detections: List[Detection3D]) -> List[Tuple[int, int]]:
        if not self.tracks:
            return []

        track_ids = list(self.tracks.keys())
        track_states = np.array([self.tracks[tid].state[:3] for tid in track_ids])
        det_positions = np.array([det.box.center for det in detections])

        cost_matrix = np.zeros((len(track_ids), len(detections)))
        for i, t_state in enumerate(track_states):
            for j, d_pos in enumerate(det_positions):
                cost_matrix[i, j] = np.linalg.norm(t_state - d_pos)

        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = []
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.iou_threshold:
                matched.append((list(self.tracks.keys())[i], j))

        return matched

    def _update_track_state(self, old_state: np.ndarray, detection: Detection3D) -> np.ndarray:
        new_state = old_state.copy()
        new_state[:3] = detection.box.center
        if detection.box.velocity is not None:
            new_state[3:6] = detection.box.velocity
        return new_state

    def _create_new_track(self, detection: Detection3D):
        track_id = self.track_id_counter
        self.track_id_counter += 1

        state = np.zeros(self.state_dim)
        state[:3] = detection.box.center
        if detection.box.velocity is not None:
            state[3:6] = detection.box.velocity

        track = Track(
            track_id=track_id,
            state=state,
            covariance=np.eye(self.state_dim) * 0.1,
            sensor_ids=detection.metadata.get('sensor_ids', set()),
            age=0,
            hits=1,
            time_since_update=0,
            label=detection.box.label,
            score=detection.box.score
        )
        self.tracks[track_id] = track


class MultiObjectTracker:
    def __init__(
        self,
        sensor_fusion: MultiSensorFusion,
        tracker: Tracker
    ):
        self.sensor_fusion = sensor_fusion
        self.tracker = tracker

    def process(self, measurements: List[SensorMeasurement]) -> Tuple[FusedState, List[Track]]:
        fused_state = self.sensor_fusion.get_fused_state()

        detections = self._measurements_to_detections(measurements)
        tracks = self.tracker.update(detections)

        return fused_state, tracks

    def _measurements_to_detections(self, measurements: List[SensorMeasurement]) -> List[Detection3D]:
        detections = []

        for meas in measurements:
            if meas.sensor_type in [SensorType.CAMERA, SensorType.LIDAR, SensorType.RADAR]:
                det = self._convert_measurement_to_detection(meas)
                if det:
                    detections.append(det)

        return detections

    def _convert_measurement_to_detection(self, measurement: SensorMeasurement) -> Optional[Detection3D]:
        return None


class SensorFusionPipeline:
    def __init__(
        self,
        fusion_method: FusionAlgorithm = FusionAlgorithm.EXTENDED_KALMAN,
        state_dim: int = 12
    ):
        self.fusion = MultiSensorFusion(fusion_method, state_dim)
        self.tracker = MultiObjectTracker(
            MultiSensorFusion(fusion_method, state_dim),
            Tracker(state_dim=6)
        )

    def add_sensor(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        measurement_model: Callable,
        measurement_jacobian: Callable,
        noise_covariance: np.ndarray
    ):
        self.fusion.add_sensor(sensor_id, sensor_type, measurement_model, measurement_jacobian, noise_covariance)

    def process_measurements(self, measurements: List[SensorMeasurement], dt: float) -> Tuple[FusedState, List[Track]]:
        self.fusion.predict(dt)
        for meas in measurements:
            self.fusion.update(meas)

        return self.tracker.process(measurements)


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

    @staticmethod
    def compute_track_quality(tracks: List[Track]) -> Dict[str, float]:
        if not tracks:
            return {}

        total_age = sum(t.age for t in tracks)
        total_hits = sum(t.hits for t in tracks)
        active_tracks = sum(1 for t in tracks if t.time_since_update == 0)

        return {
            'avg_age': total_age / len(tracks),
            'avg_hits': total_hits / len(tracks),
            'active_ratio': active_tracks / len(tracks),
            'avg_score': np.mean([t.score for t in tracks])
        }


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet