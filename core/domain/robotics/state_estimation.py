from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math
import time


@dataclass(frozen=True, slots=True)
class StateEstimate:
    mean: np.ndarray
    covariance: np.ndarray
    timestamp: float
    innovation: Optional[np.ndarray] = None
    innovation_cov: Optional[np.ndarray] = None
    log_likelihood: float = 0.0


class StateEstimator(ABC):
    @abstractmethod
    def predict(self, dt: float) -> None:
        pass

    @abstractmethod
    def update(self, measurement: np.ndarray) -> None:
        pass

    @abstractmethod
    def get_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        pass


class ExtendedKalmanFilter:
    def __init__(
        self,
        state_dim: int,
        meas_dim: int,
        process_noise: np.ndarray = None,
        measurement_noise: np.ndarray = None,
    ):
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 1000

        self.Q = process_noise if process_noise is not None else np.eye(state_dim) * 0.01
        self.R = measurement_noise if measurement_noise is not None else np.eye(1) * 0.1

        self.F = np.eye(state_dim)
        self.H = np.zeros((1, state_dim))

        self.f_func = None
        self.h_func = None
        self.F_func = None
        self.H_func = None

    def set_dynamics(self, f_func: Callable, F_func: Callable):
        self.f_func = f_func
        self.F_func = F_func

    def set_measurement(self, h_func: Callable, H_func: Callable):
        self.h_func = h_func
        self.H_func = H_func

    def predict(self, dt: float = 1.0):
        if self.f_func is not None:
            self.x = self.f_func(self.x, dt)
        if self.F_func is not None:
            self.F = self.F_func(self.x, dt)
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        if self.H_func is not None:
            self.H = self.H_func(self.x)
        if self.h_func is not None:
            z_pred = self.h_func(self.x)
        else:
            z_pred = self.H @ self.x

        y = z - z_pred
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - z_pred)
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P

    def get_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x.copy(), self.P.copy()


class UnscentedKalmanFilter:
    def __init__(
        self,
        state_dim: int,
        meas_dim: int,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        self.state_dim = state_dim
        self.meas_dim = meas_dim
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
        self.Q = np.eye(state_dim) * 0.01
        self.R = np.eye(1) * 0.1

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        sqrt_P = np.linalg.cholesky((self.state_dim + self.lambd) * P)
        sigmas = np.zeros((self.n_sigma, self.state_dim))
        sigmas[0] = x
        for i in range(self.state_dim):
            sigmas[i + 1] = x + sqrt_P[i]
            sigmas[self.state_dim + i + 1] = x - sqrt_P[i]
        return sigmas

    def predict(self, f_func: Callable, dt: float):
        sigmas = self._sigma_points(self.x, self.P)
        sigmas_f = np.array([f_func(s, dt) for s in sigmas])

        self.x = np.sum(self.Wm[:, None] * sigmas_f, axis=0)
        diff = sigmas_f - self.x
        self.P = diff.T @ np.diag(self.Wc) @ diff + self.Q

    def update(self, z: float, h_func: Callable):
        sigmas = self._sigma_points(self.x, self.P)
        sigmas_z = np.array([h_func(s) for s in sigmas])

        z_pred = np.sum(self.Wm[:, None] * sigmas_z, axis=0)
        diff_z = sigmas_z - z_pred
        diff_x = sigmas - self.x

        P_zz = diff_z.T @ np.diag(self.Wc) @ diff_z + self.R
        P_xz = diff_x.T @ np.diag(self.Wc) @ diff_z

        K = P_xz @ np.linalg.inv(P_zz)
        self.x = self.x + K @ (z - z_pred)
        self.P = self.P - K @ P_zz @ K.T


class ParticleFilter:
    def __init__(
        self,
        num_particles: int = 1000,
        state_dim: int = 4,
        process_noise: float = 0.1,
        measurement_noise: float = 0.1,
    ):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self.particles = np.random.randn(num_particles, state_dim)
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, f_func: Callable, dt: float = 1.0):
        for i in range(self.num_particles):
            self.particles[i] = f_func(self.particles[i], dt) + np.random.randn(self.state_dim) * self.process_noise * dt

    def update(self, z: float, h_func: Callable):
        for i in range(self.num_particles):
            z_pred = h_func(self.particles[i])
            diff = z - z_pred
            self.weights[i] = np.exp(-0.5 * diff**2 / self.measurement_noise**2)

        self.weights /= self.weights.sum()

        if 1.0 / np.sum(self.weights**2) < self.num_particles / 2:
            self._resample()

    def _resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.average(self.particles, axis=0, weights=self.weights)
        P = np.cov(self.particles.T, aweights=self.weights)
        return x, P


class IMMFilter:
    def __init__(self, filters: List, mode_transition_matrix: np.ndarray):
        self.filters = filters
        self.mode_probs = np.ones(len(filters)) / len(filters)
        self.transition_matrix = mode_transition_matrix

    def predict(self, dt: float):
        for f in self.filters:
            f.predict(dt)

    def update(self, z: float, H: np.ndarray = None, R: np.ndarray = None):
        likelihoods = []
        for f in self.filters:
            y = z - f.H @ f.x
            S = f.H @ f.P @ f.H.T + f.R
            likelihood = np.exp(-0.5 * y.T @ np.linalg.inv(S) @ y) / np.sqrt(np.linalg.det(2 * np.pi * S))
            likelihoods.append(likelihood)

        likelihoods = np.array(likelihoods)
        self.mode_probs = self.mode_probs * likelihoods
        self.mode_probs /= self.mode_probs.sum()

        for i, f in enumerate(self.filters):
            f.update(z, f.H, f.R)

        self.mode_probs = self.transition_matrix.T @ self.mode_probs

    def get_combined_state(self):
        x = np.sum(self.mode_probs[:, None] * np.array([f.x for f in self.filters]), axis=0)
        P = np.zeros_like(self.filters[0].P)
        for i, f in enumerate(self.filters):
            diff = f.x - self.get_combined_state()[0]
            P += self.mode_probs[i] * (f.P + np.outer(diff, diff))
        return self.get_combined_state()


class ParticleFilter:
    def __init__(
        self,
        num_particles: int = 1000,
        process_noise: float = 0.1,
        measurement_noise: float = 0.1,
    ):
        self.num_particles = num_particles
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.particles = np.random.randn(1000, 4)
        self.weights = np.ones(1000) / 1000

    def predict(self, f_func: Callable, dt: float = 1.0):
        for i in range(self.num_particles):
            self.particles[i] = f_func(self.particles[i], dt) + np.random.randn(4) * self.process_noise * dt

    def update(self, z: float, h_func: Callable):
        for i in range(self.num_particles):
            z_pred = h_func(self.particles[i])
            diff = z - z_pred
            self.weights[i] = np.exp(-0.5 * (diff**2) / self.measurement_noise**2)

        self.weights /= self.weights.sum()

        if 1.0 / np.sum(self.weights**2) < self.num_particles / 2:
            self._resample()

    def _resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_estimate(self):
        x = np.average(self.particles, axis=0, weights=self.weights)
        P = np.cov(self.particles.T, aweights=self.weights)
        return x, P


class IMMFilter:
    def __init__(self, filters: List, mode_transition_matrix: np.ndarray):
        self.filters = filters
        self.mode_probs = np.ones(len(filters)) / len(filters)
        self.transition_matrix = mode_transition_matrix

    def predict(self, dt: float):
        for f in self.filters:
            f.predict(dt)

    def update(self, z: np.ndarray, H: np.ndarray = None, R: np.ndarray = None):
        likelihoods = []
        for f in self.filters:
            y = z - f.H @ f.x
            S = f.H @ f.P @ f.H.T + f.R
            likelihood = np.exp(-0.5 * y.T @ np.linalg.inv(S) @ y) / np.sqrt(np.linalg.det(2 * np.pi * S))
            likelihoods.append(likelihood)

        likelihoods = np.array(likelihoods)
        self.mode_probs = self.mode_probs * likelihoods
        self.mode_probs /= self.mode_probs.sum()

        for i, f in enumerate(self.filters):
            f.update(z, f.H, f.R)

        self.mode_probs = self.transition_matrix.T @ self.mode_probs

    def get_combined_state(self):
        x = np.sum(self.mode_probs[:, None] * np.array([f.x for f in self.filters]), axis=0)
        P = np.zeros_like(self.filters[0].P)
        for i, f in enumerate(self.filters):
            diff = f.x - self.get_combined_state()[0]
            P += self.mode_probs[i] * (f.P + np.outer(diff, diff))
        return self.get_combined_state()


class ParticleFilter:
    def __init__(
        self,
        num_particles: int = 1000,
        state_dim: int = 4,
        process_noise: float = 0.1,
        measurement_noise: float = 0.1,
    ):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.particles = np.random.randn(num_particles, state_dim)
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, f_func: Callable, dt: float = 1.0):
        for i in range(self.num_particles):
            self.particles[i] = f_func(self.particles[i], dt) + np.random.randn(self.state_dim) * self.process_noise * dt

    def update(self, z: float, h_func: Callable):
        for i in range(self.num_particles):
            z_pred = h_func(self.particles[i])
            diff = z - z_pred
            self.weights[i] = np.exp(-0.5 * (diff**2) / self.measurement_noise**2)

        self.weights /= self.weights.sum()

        if 1.0 / np.sum(self.weights**2) < self.num_particles / 2:
            self._resample()

    def _resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_estimate(self):
        x = np.average(self.particles, axis=0, weights=self.weights)
        P = np.cov(self.particles.T, aweights=self.weights)
        return x, P


class IMMFilter:
    def __init__(self, filters: List, mode_transition_matrix: np.ndarray):
        self.filters = filters
        self.mode_probs = np.ones(len(filters)) / len(filters)
        self.transition_matrix = mode_transition_matrix

    def predict(self, dt: float):
        for f in self.filters:
            f.predict(dt)

    def update(self, z: float, H: np.ndarray = None, R: np.ndarray = None):
        likelihoods = []
        for f in self.filters:
            y = z - f.H @ f.x
            S = f.H @ f.P @ f.H.T + f.R
            likelihood = np.exp(-0.5 * y.T @ np.linalg.inv(S) @ y) / np.sqrt(np.linalg.det(2 * np.pi * S))
            likelihoods.append(likelihood)

        likelihoods = np.array(likelihoods)
        self.mode_probs = self.mode_probs * likelihoods
        self.mode_probs /= self.mode_probs.sum()

        for i, f in enumerate(self.filters):
            f.update(z, f.H, f.R)

        self.mode_probs = self.transition_matrix.T @ self.mode_probs

    def get_combined_state(self):
        x = np.sum(self.mode_probs[:, None] * np.array([f.x for f in self.filters]), axis=0)
        P = np.zeros_like(self.filters[0].P)
        for i, f in enumerate(self.filters):
            diff = f.x - self.get_combined_state()[0]
            P += self.mode_probs[i] * (f.P + np.outer(diff, diff))
        return self.get_combined_state()


class ParticleFilter:
    def __init__(
        self,
        num_particles: int = 1000,
        state_dim: int = 4,
        process_noise: float = 0.1,
        measurement_noise: float = 0.1,
    ):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.particles = np.random.randn(num_particles, state_dim)
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, f_func: Callable, dt: float = 1.0):
        for i in range(self.num_particles):
            self.particles[i] = f_func(self.particles[i], dt) + np.random.randn(self.state_dim) * self.process_noise * dt

    def update(self, z: float, h_func: Callable):
        for i in range(self.num_particles):
            z_pred = h_func(self.particles[i])
            diff = z - z_pred
            self.weights[i] = np.exp(-0.5 * (diff**2) / self.measurement_noise**2)

        self.weights /= self.weights.sum()

        if 1.0 / np.sum(self.weights**2) < self.num_particles / 2:
            self._resample()

    def _resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_estimate(self):
        x = np.average(self.particles, axis=0, weights=self.weights)
        P = np.cov(self.particles.T, aweights=self.weights)
        return x, P


class IMMFilter:
    def __init__(self, filters: List, mode_transition_matrix: np.ndarray):
        self.filters = filters
        self.mode_probs = np.ones(len(filters)) / len(filters)
        self.transition_matrix = mode_transition_matrix

    def predict(self, dt: float):
        for f in self.filters:
            f.predict(dt)

    def update(self, z: float, H: np.ndarray = None, R: np.ndarray = None):
        likelihoods = []
        for f in self.filters:
            y = z - f.H @ f.x
            S = f.H @ f.P @ f.H.T + f.R
            likelihood = np.exp(-0.5 * y.T @ np.linalg.inv(S) @ y) / np.sqrt(np.linalg.det(2 * np.pi * S))
            likelihoods.append(likelihood)

        likelihoods = np.array(likelihoods)
        self.mode_probs = self.mode_probs * likelihoods
        self.mode_probs /= self.mode_probs.sum()

        for i, f in enumerate(self.filters):
            f.update(z, f.H, f.R)

        self.mode_probs = self.transition_matrix.T @ self.mode_probs

    def get_combined_state(self):
        x = np.sum(self.mode_probs[:, None] * np.array([f.x for f in self.filters]), axis=0)
        P = np.zeros_like(self.filters[0].P)
        for i, f in enumerate(self.filters):
            diff = f.x - self.get_combined_state()[0]
            P += self.mode_probs[i] * (f.P + np.outer(diff, diff))
        return self.get_combined_state()


class ParticleFilter:
    def __init__(
        self,
        num_particles: int = 1000,
        state_dim: int = 4,
        process_noise: float = 0.1,
        measurement_noise: float = 0.1,
    ):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.particles = np.random.randn(num_particles, state_dim)
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, f_func: Callable, dt: float = 1.0):
        for i in range(self.num_particles):
            self.particles[i] = f_func(self.particles[i], dt) + np.random.randn(self.state_dim) * self.process_noise * dt

    def update(self, z: float, h_func: Callable):
        for i in range(self.num_particles):
            z_pred = h_func(self.particles[i])
            diff = z - z_pred
            self.weights[i] = np.exp(-0.5 * diff**2 / self.measurement_noise**2)

        self.weights /= self.weights.sum()

        if 1.0 / np.sum(self.weights**2) < self.num_particles / 2:
            self._resample()

    def _resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_estimate(self):
        x = np.average(self.particles, axis=0, weights=self.weights)
        P = np.cov(self.particles.T, aweights=self.weights)
        return x, P


class IMMFilter:
    def __init__(self, filters: List, mode_transition_matrix: np.ndarray):
        self.filters = filters
        self.mode_probs = np.ones(len(filters)) / len(filters)
        self.transition_matrix = mode_transition_matrix

    def predict(self, dt: float):
        for f in self.filters:
            f.predict(dt)

    def update(self, z: float, H: np.ndarray = None, R: np.ndarray = None):
        likelihoods = []
        for f in self.filters:
            y = z - f.H @ f.x
            S = f.H @ f.P @ f.H.T + f.R
            likelihood = np.exp(-0.5 * y.T @ np.linalg.inv(S) @ y) / np.sqrt(np.linalg.det(2 * np.pi * S))
            likelihoods.append(likelihood)

        likelihoods = np.array(likelihoods)
        self.mode_probs = self.mode_probs * likelihoods
        self.mode_probs /= self.mode_probs.sum()

        for i, f in enumerate(self.filters):
            f.update(z, f.H, f.R)

        self.mode_probs = self.transition_matrix.T @ self.mode_probs

    def get_combined_state(self):
        x = np.sum(self.mode_probs[:, None] * np.array([f.x for f in self.filters]), axis=0)
        P = np.zeros_like(self.filters[0].P)
        for i, f in enumerate(self.filters):
            diff = f.x - self.get_combined_state()[0]
            P += self.mode_probs[i] * (f.P + np.outer(diff, diff))
        return self.get_combined_state()


class ParticleFilter:
    def __init__(
        self,
        num_particles: int = 1000,
        state_dim: int = 4,
        process_noise: float = 0.1,
        measurement_noise: float = 0.1,
    ):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.particles = np.random.randn(num_particles, state_dim)
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, f_func: Callable, dt: float = 1.0):
        for i in range(self.num_particles):
            self.particles[i] = f_func(self.particles[i], dt) + np.random.randn(self.state_dim) * self.process_noise * dt

    def update(self, z: float, h_func: Callable):
        for i in range(self.num_particles):
            z_pred = h_func(self.particles[i])
            diff = z - z_pred
            self.weights[i] = np.exp(-0.5 * diff**2 / self.measurement_noise**2)

        self.weights /= self.weights.sum()

        if 1.0 / np.sum(self.weights**2) < self.num_particles / 2:
            self._resample()

    def _resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_estimate(self):
        x = np.average(self.particles, axis=0, weights=self.weights)
        P = np.cov(self.particles.T, aweights=self.weights)
        return x, P


class IMMFilter:
    def __init__(self, filters: List, mode_transition_matrix: np.ndarray):
        self.filters = filters
        self.mode_probs = np.ones(len(filters)) / len(filters)
        self.transition_matrix = mode_transition_matrix

    def predict(self, dt: float):
        for f in self.filters:
            f.predict(dt)

    def update(self, z: float, H: np.ndarray = None, R: np.ndarray = None):
        likelihoods = []
        for f in self.filters:
            y = z - f.H @ f.x
            S = f.H @ f.P @ f.H.T + f.R
            likelihood = np.exp(-0.5 * y.T @ np.linalg.inv(S) @ y) / np.sqrt(np.linalg.det(2 * np.pi * S))
            likelihoods.append(likelihood)

        likelihoods = np.array(likelihoods)
        self.mode_probs = self.mode_probs * likelihoods
        self.mode_probs /= self.mode_probs.sum()

        for i, f in enumerate(self.filters):
            f.update(z, f.H, f.R)

        self.mode_probs = self.transition_matrix.T @ self.mode_probs

    def get_combined_state(self):
        x = np.sum(self.mode_probs[:, None] * np.array([f.x for f in self.filters]), axis=0)
        P = np.zeros_like(self.filters[0].P)
        for i, f in enumerate(self.filters):
            diff = f.x - self.get_combined_state()[0]
            P += self.mode_probs[i] * (f.P + np.outer(diff, diff))
        return self.get_combined_state()


class ParticleFilter:
    def __init__(
        self,
        num_particles: int = 1000,
        state_dim: int = 4,
        process_noise: float = 0.1,
        measurement_noise: float = 0.1,
    ):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.particles = np.random.randn(num_particles, state_dim)
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, f_func: Callable, dt: float = 1.0):
        for i in range(self.num_particles):
            self.particles[i] = f_func(self.particles[i], dt) + np.random.randn(self.state_dim) * self.process_noise * dt

    def update(self, z: float, h_func: Callable):
        for i in range(self.num_particles):
            z_pred = h_func(self.particles[i])
            diff = z - z_pred
            self.weights[i] = np.exp(-0.5 * diff**2 / self.measurement_noise**2)

        self.weights /= self.weights.sum()

        if 1.0 / np.sum(self.weights**2) < self.num_particles / 2:
            self._resample()

    def _resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_estimate(self):
        x = np.average(self.particles, axis=0, weights=self.weights)
        P = np.cov(self.particles.T, aweights=self.weights)
        return x, P


class IMMFilter:
    def __init__(self, filters: List, mode_transition_matrix: np.ndarray):
        self.filters = filters
        self.mode_probs = np.ones(len(filters)) / len(filters)
        self.transition_matrix = mode_transition_matrix

    def predict(self, dt: float):
        for f in self.filters:
            f.predict(dt)

    def update(self, z: float, H: np.ndarray = None, R: np.ndarray = None):
        likelihoods = []
        for f in self.filters:
            y = z - f.H @ f.x
            S = f.H @ f.P @ f.H.T + f.R
            likelihood = np.exp(-0.5 * y.T @ np.linalg.inv(S) @ y) / np.sqrt(np.linalg.det(2 * np.pi * S))
            likelihoods.append(likelihood)

        likelihoods = np.array(likelihoods)
        self.mode_probs = self.mode_probs * likelihoods
        self.mode_probs /= self.mode_probs.sum()

        for i, f in enumerate(self.filters):
            f.update(z, f.H, f.R)

        self.mode_probs = self.transition_matrix.T @ self.mode_probs

    def get_combined_state(self):
        x = np.sum(self.mode_probs[:, None] * np.array([f.x for f in self.filters]), axis=0)
        P = np.zeros_like(self.filters[0].P)
        for i, f in enumerate(self.filters):
            diff = f.x - self.get_combined_state()[0]
            P += self.mode_probs[i] * (f.P + np.outer(diff, diff))
        return self.get_combined_state()


class ParticleFilter:
    def __init__(
        self,
        num_particles: int = 1000,
        state_dim: int = 4,
        process_noise: float = 0.1,
        measurement_noise: float = 0.1,
    ):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.particles = np.random.randn(num_particles, state_dim)
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, f_func: Callable, dt: float = 1.0):
        for i in range(self.num_particles):
            self.particles[i] = f_func(self.particles[i], dt) + np.random.randn(self.state_dim) * self.process_noise * dt

    def update(self, z: float, h_func: Callable):
        for i in range(self.num_particles):
            z_pred = h_func(self.particles[i])
            diff = z - z_pred
            self.weights[i] = np.exp(-0.5 * diff**2 / self.measurement_noise**2)

        self.weights /= self.weights.sum()

        if 1.0 / np.sum(self.weights**2) < self.num_particles / 2:
            self._resample()

    def _resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_estimate(self):
        x = np.average(self.particles, axis=0, weights=self.weights)
        P = np.cov(self.particles.T, aweights=self.weights)
        return x, P


class IMMFilter:
    def __init__(self, filters: List, mode_transition_matrix: np.ndarray):
        self.filters = filters
        self.mode_probs = np.ones(len(filters)) / len(filters)
        self.transition_matrix = mode_transition_matrix

    def predict(self, dt: float):
        for f in self.filters:
            f.predict(dt)

    def update(self, z: float, H: np.ndarray = none, R: np.ndarray = none):
        likelihoods = []
        for f in self.filters:
            y = z - f.H @ f.x
            S = f.H @ f.P @ f.H.T + f.R
            likelihood = np.exp(-0.5 * y.T @ np.linalg.inv(S) @ y) / np.sqrt(np.linalg.det(2 * np.pi * S))
            likelihoods.append(likelihood)

        likelihoods = np.array(likelihoods)
        self.mode_probs = self.mode_probs * likelihoods
        self.mode_probs /= self.mode_probs.sum()

        for i, f in enumerate(self.filters):
            f.update(z, f.H, f.R)

        self.mode_probs = self.transition_matrix.T @ self.mode_probs

    def get_combined_state(self):
        x = np.sum(self.mode_probs[:, None] * np.array([f.x for f in self.filters]), axis=0)
        P = np.zeros_like(self.filters[0].P)
        for i, f in enumerate(self.filters):
            diff = f.x - self.get_combined_state()[0]
            P += self.mode_probs[i] * (f.P + np.outer(diff, diff))
        return self.get_combined_state()