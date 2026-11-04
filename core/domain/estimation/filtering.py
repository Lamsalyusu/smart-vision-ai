from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
import random
from collections import defaultdict
import copy


class MotionModel(Enum):
    CONSTANT_VELOCITY = "constant_velocity"
    CONSTANT_ACCELERATION = "constant_acceleration"
    CONSTANT_TURN_RATE = "constant_turn_rate"
    SINGER = "singer"
    NEARLY_CONSTANT_VELOCITY = "nearly_constant_velocity"
    BICYCLE = "bicycle"
    ACKERMANN = "ackermann"
    DIFFERENTIAL_DRIVE = "differential_drive"


class MeasurementModel(Enum):
    POSITION = "position"
    POSITION_VELOCITY = "position_velocity"
    BEARING_RANGE = "bearing_range"
    BEARING_ONLY = "bearing_only"
    DOPPLER = "doppler"
    IMAGE_PLANE = "image_plane"


@dataclass(frozen=True, slots=True)
class State:
    x: np.ndarray
    P: np.ndarray
    timestamp: float


@dataclass(frozen=True, slots=True)
class Measurement:
    z: np.ndarray
    R: np.ndarray
    timestamp: float
    sensor_id: str


class BaseFilter(ABC):
    @abstractmethod
    def predict(self, dt: float) -> State:
        pass

    @abstractmethod
    def update(self, measurement: Measurement) -> State:
        pass


class KalmanFilter:
    def __init__(
        self,
        state_dim: int,
        measurement_dim: int,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray
    ):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R

        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim)

    def predict(self, dt: float = 1.0) -> State:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        return State(
            x=self.x.copy(),
            P=self.P.copy(),
            timestamp=time.time()
        )

    def update(self, z: np.ndarray, R: np.ndarray = None, H: np.ndarray = None) -> State:
        R = R if R is not None else self.R
        H = H if H is not None else self.H

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

        return State(
            x=self.x.copy(),
            P=self.P.copy(),
            timestamp=time.time()
        )

    def get_state(self) -> State:
        return State(
            x=self.x.copy(),
            P=self.P.copy(),
            timestamp=time.time()
        )


class ExtendedKalmanFilter:
    def __init__(
        self,
        state_dim: int,
        process_noise: np.ndarray,
        motion_model: Callable[[np.ndarray, float], np.ndarray],
        motion_jacobian: Callable[[np.ndarray, float], np.ndarray],
        measurement_models: Dict[str, Tuple[Callable, Callable]]
    ):
        self.state_dim = state_dim
        self.process_noise = process_noise
        self.motion_model = motion_model
        self.motion_jacobian = motion_jacobian
        self.measurement_models = measurement_models

        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim)

    def predict(self, dt: float) -> State:
        self.x = self.motion_model(self.x, dt)
        F = self.motion_jacobian(self.x, dt)
        self.P = F @ self.P @ F.T + self.process_noise

        return State(
            x=self.x.copy(),
            P=self.P.copy(),
            timestamp=time.time()
        )

    def update(self, measurement: np.ndarray, sensor_id: str) -> State:
        if sensor_id not in self.measurement_models:
            raise ValueError(f"No measurement model for sensor: {sensor_id}")

        h_func, H_func = self.measurement_models[sensor_id]

        z_pred = h_func(self.x)
        H = H_func(self.x)

        R = np.eye(len(measurement)) * 0.1

        y = measurement - z_pred
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

        return State(
            x=self.x.copy(),
            P=self.P.copy(),
            timestamp=time.time()
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

    def predict(self, dt: float, motion_model: Callable) -> State:
        sigma_points = self._generate_sigma_points()
        predicted_sigma = np.zeros_like(sigma_points)

        for i in range(self.n_sigma):
            predicted_sigma[i] = motion_model(sigma_points[i], dt)

        self.x = np.sum(self.Wm[:, np.newaxis] * predicted_sigma, axis=0)
        self.P = np.zeros((self.state_dim, self.state_dim))

        for i in range(self.n_sigma):
            diff = predicted_sigma[i] - self.x
            self.P += self.Wc[i] * np.outer(diff, diff)

        self.P += np.eye(self.state_dim) * 0.01

        return State(
            x=self.x.copy(),
            P=self.P.copy(),
            timestamp=time.time()
        )

    def update(self, measurement: np.ndarray, R: np.ndarray, h_func: Callable) -> State:
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

        return State(
            x=self.x.copy(),
            P=self.P.copy(),
            timestamp=time.time()
        )


class ParticleFilter:
    def __init__(
        self,
        state_dim: int,
        num_particles: int = 1000,
        process_noise: np.ndarray = None,
        seed: Optional[int] = None,
    ):
        self.state_dim = state_dim
        self.num_particles = num_particles
        self.process_noise = process_noise if process_noise is not None else np.eye(state_dim) * 0.01
        self._rng = np.random.default_rng(seed)

        self.particles = self._rng.standard_normal((num_particles, state_dim)) * 0.1
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, dt: float, motion_model: Callable) -> State:
        for i in range(self.num_particles):
            self.particles[i] = motion_model(self.particles[i], dt)
            self.particles[i] += self._rng.multivariate_normal(np.zeros(self.state_dim), self.process_noise)

        mean_state = np.average(self.particles, axis=0, weights=self.weights)
        cov = np.cov(self.particles.T, aweights=self.weights)

        return State(
            x=mean_state,
            P=cov,
            timestamp=time.time()
        )

    def update(self, measurement: np.ndarray, R: np.ndarray, h_func: Callable) -> State:
        for i in range(self.num_particles):
            z_pred = h_func(self.particles[i])
            diff = measurement - z_pred
            self.weights[i] *= math.exp(-0.5 * diff @ np.linalg.inv(R) @ diff)

        self.weights += 1e-10
        self.weights /= np.sum(self.weights)

        if 1.0 / np.sum(self.weights**2) < self.num_particles / 2:
            self._resample()

        mean_state = np.average(self.particles, axis=0, weights=self.weights)
        cov = np.cov(self.particles.T, aweights=self.weights)

        return State(
            x=mean_state,
            P=cov,
            timestamp=time.time()
        )

    def _resample(self):
        indices = self._rng.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles


class KalmanSmoother:
    def __init__(self, filter_: KalmanFilter):
        self.filter = filter_
        self.states: List[State] = []

    def smooth(self, measurements: List[np.ndarray]) -> List[State]:
        forward_states = []

        for z in measurements:
            self.filter.predict(1.0)
            state = self.filter.update(z)
            forward_states.append(state)

        smoothed_states = [forward_states[-1]]

        for i in range(len(forward_states) - 2, -1, -1):
            F = np.eye(len(forward_states[i].x))
            P = forward_states[i].P
            x = forward_states[i].x

            next_P = smoothed_states[0].P
            next_x = smoothed_states[0].x

            C = P @ F.T @ np.linalg.inv(next_P)

            x_smooth = x + C @ (next_x - F @ x)
            P_smooth = P + C @ (smoothed_states[0].P - next_P) @ C.T

            smoothed_states.insert(0, State(x=x_smooth, P=P_smooth, timestamp=0))

        return smoothed_states


class InformationFilter:
    def __init__(self, state_dim: int, process_noise: np.ndarray):
        self.state_dim = state_dim
        self.process_noise = process_noise

        self.Y = np.eye(state_dim)
        self.y = np.zeros(state_dim)

    def predict(self, dt: float, F: np.ndarray = None, Q: np.ndarray = None) -> State:
        F = F if F is not None else np.eye(self.state_dim)
        Q = Q if Q is not None else self.process_noise

        Q_inv = np.linalg.inv(Q + 1e-6 * np.eye(self.state_dim))

        self.Y = F.T @ self.Y @ F + Q_inv
        self.y = F.T @ self.y

        self.P = np.linalg.inv(self.Y + 1e-6 * np.eye(self.state_dim))
        self.x = self.P @ self.y

        return State(
            x=self.x.copy(),
            P=self.P.copy(),
            timestamp=time.time()
        )

    def update(self, z: np.ndarray, R: np.ndarray, H: np.ndarray) -> State:
        R_inv = np.linalg.inv(R + 1e-6 * np.eye(R.shape[0]))

        self.Y += H.T @ R_inv @ H
        self.y += H.T @ R_inv @ z

        self.P = np.linalg.inv(self.Y + 1e-6 * np.eye(self.state_dim))
        self.x = self.P @ self.y

        return State(
            x=self.x.copy(),
            P=self.P.copy(),
            timestamp=time.time()
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


class InteractingMultipleModel:
    def __init__(
        self,
        models: List[KalmanFilter],
        transition_matrix: np.ndarray,
        initial_mode_probs: np.ndarray = None
    ):
        self.models = models
        self.transition_matrix = transition_matrix
        self.num_models = len(models)

        if initial_mode_probs is None:
            self.mode_probs = np.ones(self.num_models) / self.num_models
        else:
            self.mode_probs = initial_mode_probs

        self.mixed_state = np.zeros(models[0].state_dim)
        self.mixed_cov = np.eye(models[0].state_dim)

    def mix(self):
        self.mixed_state = np.zeros(self.models[0].state_dim)
        self.mixed_cov = np.zeros((self.models[0].state_dim, self.models[0].state_dim))

        for i in range(self.num_models):
            self.mixed_state += self.mode_probs[i] * self.models[i].x

        for i in range(self.num_models):
            diff = self.models[i].x - self.mixed_state
            self.mixed_cov += self.mode_probs[i] * (self.models[i].P + np.outer(diff, diff))

    def predict(self, dt: float = 1.0):
        self.mix()

        for i, model in enumerate(self.models):
            model.x = self.mixed_state.copy()
            model.P = self.mixed_cov.copy()
            model.predict(dt)

    def update(self, z: np.ndarray, R: np.ndarray, H: np.ndarray = None):
        likelihoods = np.zeros(self.num_models)

        for i, model in enumerate(self.models):
            H = H if H is not None else model.H
            R = R if R is not None else model.R

            y = z - H @ model.x
            S = H @ model.P @ H.T + R
            K = model.P @ H.T @ np.linalg.inv(S)

            model.x = model.x + K @ y
            model.P = (np.eye(model.state_dim) - K @ H) @ model.P

            likelihoods[i] = np.exp(-0.5 * y @ np.linalg.inv(S) @ y) / np.sqrt(np.linalg.det(2 * np.pi * S) + 1e-10)

        self.mode_probs = self.mode_probs * likelihoods
        self.mode_probs /= np.sum(self.mode_probs)

    def get_state(self) -> State:
        x = np.zeros(self.models[0].state_dim)
        P = np.zeros((self.models[0].state_dim, self.models[0].state_dim))

        for i, model in enumerate(self.models):
            x += self.mode_probs[i] * model.x
            diff = model.x - x
            P += self.mode_probs[i] * (model.P + np.outer(diff, diff))

        return State(
            x=x,
            P=P,
            timestamp=time.time()
        )


class FixedLagSmoother:
    def __init__(self, filter_: KalmanFilter, lag: int = 10):
        self.filter = filter_
        self.lag = lag
        self.history: List[Tuple[State, np.ndarray]] = []

    def step(self, z: np.ndarray) -> State:
        state = self.filter.predict(1.0)
        state = self.filter.update(z)

        self.history.append((state, z))

        if len(self.history) > self.lag:
            oldest_state, oldest_z = self.history[0]
            self._smooth_step(oldest_state, oldest_z)
            self.history.pop(0)

        return self.history[-1][0] if self.history else self.filter.get_state()

    def _smooth_step(self, state: State, z: np.ndarray):
        # Fixed-lag smoothing step: adjust past state with new measurement
        # Simple implementation: blend measurement into state history
        alpha = 0.3
        smoothed = (1 - alpha) * state.position + alpha * z[:3] if len(z) >= 3 else state.position
        return smoothed


class RTS_Smoother:
    def __init__(self, filter_: KalmanFilter):
        self.filter = filter_
        self.forward_states: List[State] = []

    def forward_step(self, z: np.ndarray) -> State:
        state = self.filter.predict(1.0)
        state = self.filter.update(z)
        self.forward_states.append(state)
        return state

    def backward_pass(self) -> List[State]:
        if not self.forward_states:
            return []

        smoothed = [self.forward_states[-1]]

        for i in range(len(self.forward_states) - 2, -1, -1):
            F = np.eye(len(self.forward_states[i].x))
            P = self.forward_states[i].P
            x = self.forward_states[i].x

            next_x = smoothed[0].x
            next_P = smoothed[0].P

            C = P @ F.T @ np.linalg.inv(next_P)

            x_smooth = x + C @ (next_x - F @ x)
            P_smooth = P + C @ (next_P - F @ P @ F.T) @ C.T

            smoothed.insert(0, State(x=x_smooth, P=P_smooth, timestamp=0))

        return smoothed


class KalmanFilterBank:
    def __init__(self, filters: List[KalmanFilter]):
        self.filters = filters
        self.num_filters = len(filters)

    def predict(self, dt: float = 1.0):
        for f in self.filters:
            f.predict(dt)

    def update(self, z: np.ndarray, R: np.ndarray = None, H: np.ndarray = None):
        for f in self.filters:
            f.update(z, R, H)

    def get_best_estimate(self) -> State:
        best_filter = min(self.filters, key=lambda f: np.trace(f.P))
        return best_filter.get_state()


class AdaptiveKalmanFilter:
    def __init__(
        self,
        filter_: KalmanFilter,
        innovation_window: int = 10,
        adaptation_rate: float = 0.1
    ):
        self.filter = filter_
        self.innovation_window = innovation_window
        self.adaptation_rate = adaptation_rate
        self.innovations: List[np.ndarray] = []
        self.innovation_covs: List[np.ndarray] = []

    def step(self, z: np.ndarray, R: np.ndarray = None, H: np.ndarray = None) -> State:
        R = R if R is not None else self.filter.R
        H = H if H is not None else self.filter.H

        self.filter.predict(1.0)
        state = self.filter.update(z, R, H)

        H = H if H is not None else self.filter.H
        innovation = z - H @ self.filter.x
        S = H @ self.filter.P @ H.T + R

        self.innovations.append(innovation)
        self.innovation_covs.append(S)

        if len(self.innovations) > self.innovation_window:
            self.innovations.pop(0)
            self.innovation_covs.pop(0)

        if len(self.innovations) >= 5:
            self._adapt_noise()

        return state

    def _adapt_noise(self):
        empirical_cov = np.cov(np.array(self.innovations).T)
        predicted_cov = np.mean(self.innovation_covs, axis=0)

        diff = empirical_cov - predicted_cov
        new_Q = self.filter.Q + self.adaptation_rate * diff
        self.filter.Q = np.maximum(new_Q, np.eye(self.filter.state_dim) * 1e-6)


class GridFilter:
    def __init__(
        self,
        state_space: Tuple[np.ndarray, np.ndarray],
        grid_resolution: float
    ):
        self.state_space = state_space
        self.grid_resolution = grid_resolution

        self.grid_shape = tuple(
            ((state_space[1][i] - state_space[0][i]) / grid_resolution).astype(int)
            for i in range(len(state_space[0]))
        )

        self.belief = np.zeros(self.grid_shape)
        self.belief[self.grid_shape[0] // 2, self.grid_shape[1] // 2] = 1.0

    def predict(self, transition_model: Callable):
        new_belief = np.zeros_like(self.belief)

        for idx in np.ndindex(self.grid_shape):
            prob = self.belief[idx]
            if prob > 0:
                state = np.array(idx) * self.grid_resolution
                next_state = transition_model(state)
                next_idx = tuple(
                    (next_state[i] / self.grid_resolution).astype(int)
                    for i in range(len(next_state))
                )
                if all(0 <= next_idx[i] < self.grid_shape[i] for i in range(len(next_idx))):
                    new_belief[next_idx] += prob

        self.belief = new_belief / (np.sum(self.belief) + 1e-10)

    def update(self, likelihood: np.ndarray):
        self.belief *= likelihood
        self.belief /= (np.sum(self.belief) + 1e-10)

    def get_state(self) -> State:
        max_idx = np.unravel_index(np.argmax(self.belief), self.grid_shape)
        x = np.array(max_idx) * self.grid_resolution
        P = np.eye(2) * self.grid_resolution**2
        return State(x=x, P=P, timestamp=time.time())


class FilterMetrics:
    @staticmethod
    def nees(state: State, truth: np.ndarray) -> float:
        error = state.x - truth
        return float(error @ np.linalg.inv(state.P + 1e-6 * np.eye(len(state.x))) @ error)

    @staticmethod
    def nis(innovation: np.ndarray, innovation_cov: np.ndarray) -> float:
        return float(innovation @ np.linalg.inv(innovation_cov + 1e-6 * np.eye(len(innovation))) @ innovation)

    @staticmethod
    def consistency_test(nees_values: List[float], state_dim: int, confidence: float = 0.95) -> Tuple[bool, float, float]:
        from scipy.stats import chi2
        avg_nees = np.mean(nees_values)
        lower = chi2.ppf((1 - confidence) / 2, state_dim)
        upper = chi2.ppf((1 + confidence) / 2, state_dim)
        return lower <= avg_nees <= upper, lower, upper

    @staticmethod
    def rmse(estimates: List[np.ndarray], truths: List[np.ndarray]) -> float:
        errors = [np.linalg.norm(e - t) for e, t in zip(estimates, truths)]
        return float(np.sqrt(np.mean(np.array(errors)**2)))

    @staticmethod
    def anees(estimates: List[np.ndarray], truths: List[np.ndarray], covariances: List[np.ndarray]) -> float:
        nees = []
        for e, t, P in zip(estimates, truths, covariances):
            error = e - t
            nees_val = error @ np.linalg.inv(P + 1e-6 * np.eye(len(P))) @ error
            nees.append(float(nees_val))
        return float(np.mean(nees))


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
