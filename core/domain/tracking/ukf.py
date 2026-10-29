from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .state import TrackState3D


class UnscentedKalmanFilter3D:
    """Eight-state constant-velocity UKF with position and heading observations."""

    STATE_SIZE = 8
    MEASUREMENT_INDICES = (0, 1, 2, 6)

    def __init__(
        self,
        initial_state: TrackState3D,
        covariance: Sequence[Sequence[float]] | None = None,
        process_noise: Sequence[Sequence[float]] | None = None,
        measurement_noise: Sequence[Sequence[float]] | None = None,
        alpha: float = 0.5,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not math.isfinite(beta) or beta < 0 or not math.isfinite(kappa):
            raise ValueError("beta and kappa must be finite and beta non-negative")

        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa
        self._state = np.asarray(initial_state.as_vector(), dtype=np.float64)
        self._covariance = self._matrix(
            covariance,
            self.STATE_SIZE,
            np.eye(self.STATE_SIZE, dtype=np.float64),
            "state covariance",
        )
        self._process_noise = self._matrix(
            process_noise,
            self.STATE_SIZE,
            np.eye(self.STATE_SIZE, dtype=np.float64) * 0.01,
            "process noise",
        )
        default_measurement_noise = np.diag([0.25, 0.25, 0.25, 0.05])
        self._measurement_noise = self._matrix(
            measurement_noise,
            len(self.MEASUREMENT_INDICES),
            default_measurement_noise,
            "measurement noise",
        )
        self._validate_covariance(self._covariance, "state covariance")
        self._validate_covariance(self._process_noise, "process noise", allow_semidefinite=True)
        self._validate_covariance(self._measurement_noise, "measurement noise")

    @staticmethod
    def _matrix(
        values: Sequence[Sequence[float]] | None,
        size: int,
        default: np.ndarray,
        name: str,
    ) -> np.ndarray:
        if values is None:
            return default
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.shape != (size, size) or not np.isfinite(matrix).all():
            raise ValueError(f"{name} must be a finite {size}x{size} matrix")
        return matrix

    @staticmethod
    def _validate_covariance(
        covariance: np.ndarray, name: str, allow_semidefinite: bool = False
    ) -> None:
        if not np.allclose(covariance, covariance.T, rtol=1e-9, atol=1e-12):
            raise ValueError(f"{name} must be symmetric")
        try:
            if allow_semidefinite:
                if np.linalg.eigvalsh(covariance).min() < -1e-10:
                    raise ValueError(f"{name} must be positive semidefinite")
            else:
                np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"{name} must be positive definite") from exc

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    @classmethod
    def _state_residual(cls, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        residual = left - right
        residual[6] = cls._wrap_angle(residual[6])
        return residual

    @classmethod
    def _weighted_mean(cls, sigma_points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        mean = sigma_points @ weights
        mean[6] = math.atan2(
            float(np.dot(weights, np.sin(sigma_points[6]))),
            float(np.dot(weights, np.cos(sigma_points[6]))),
        )
        return mean

    def _sigma_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        spread = self._alpha**2 * (self.STATE_SIZE + self._kappa)
        scale = spread
        if scale <= 0:
            raise ValueError("alpha and kappa produce invalid sigma-point spread")
        lambda_value = spread - self.STATE_SIZE
        try:
            root = np.linalg.cholesky(self._covariance * scale)
        except np.linalg.LinAlgError as exc:
            raise ValueError("state covariance lost positive definiteness") from exc

        points = np.empty((self.STATE_SIZE, 2 * self.STATE_SIZE + 1), dtype=np.float64)
        points[:, 0] = self._state
        for index in range(self.STATE_SIZE):
            points[:, index + 1] = self._state + root[:, index]
            points[:, index + 1 + self.STATE_SIZE] = self._state - root[:, index]

        mean_weights = np.full(
            2 * self.STATE_SIZE + 1,
            1.0 / (2.0 * spread),
            dtype=np.float64,
        )
        covariance_weights = mean_weights.copy()
        mean_weights[0] = lambda_value / spread
        covariance_weights[0] = mean_weights[0] + (1.0 - self._alpha**2 + self._beta)
        return points, mean_weights, covariance_weights

    @staticmethod
    def _transition(state: np.ndarray, elapsed_seconds: float) -> np.ndarray:
        predicted = state.copy()
        predicted[0:3] += state[3:6] * elapsed_seconds
        predicted[6] = (state[6] + state[7] * elapsed_seconds + math.pi) % (2 * math.pi) - math.pi
        return predicted

    @property
    def state(self) -> TrackState3D:
        return TrackState3D(*(float(value) for value in self._state))

    @property
    def covariance(self) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(float(value) for value in row) for row in self._covariance)

    def predict(self, elapsed_seconds: float) -> TrackState3D:
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("elapsed time must be finite and non-negative")
        sigma_points, mean_weights, covariance_weights = self._sigma_points()
        propagated = np.column_stack(
            [self._transition(sigma_points[:, index], elapsed_seconds)
             for index in range(sigma_points.shape[1])]
        )
        predicted_mean = self._weighted_mean(propagated, mean_weights)
        predicted_covariance = np.zeros_like(self._covariance)
        for index, weight in enumerate(covariance_weights):
            residual = self._state_residual(propagated[:, index], predicted_mean)
            predicted_covariance += weight * np.outer(residual, residual)
        predicted_covariance += self._process_noise * elapsed_seconds
        self._state = predicted_mean
        self._covariance = (predicted_covariance + predicted_covariance.T) * 0.5
        self._validate_covariance(self._covariance, "predicted covariance")
        return self.state

    def update(self, measurement: Sequence[float]) -> TrackState3D:
        observed = np.asarray(measurement, dtype=np.float64)
        if observed.shape != (len(self.MEASUREMENT_INDICES),) or not np.isfinite(observed).all():
            raise ValueError("measurement must contain finite x, y, z, and heading values")
        observed[3] = self._wrap_angle(float(observed[3]))

        sigma_points, mean_weights, covariance_weights = self._sigma_points()
        measurement_points = sigma_points[list(self.MEASUREMENT_INDICES), :].copy()
        measurement_points[3, :] = np.vectorize(self._wrap_angle)(measurement_points[3, :])
        measurement_mean = measurement_points @ mean_weights
        measurement_mean[3] = math.atan2(
            float(np.dot(mean_weights, np.sin(measurement_points[3]))),
            float(np.dot(mean_weights, np.cos(measurement_points[3]))),
        )

        measurement_covariance = self._measurement_noise.copy()
        cross_covariance = np.zeros(
            (self.STATE_SIZE, len(self.MEASUREMENT_INDICES)), dtype=np.float64
        )
        for index, weight in enumerate(covariance_weights):
            state_residual = self._state_residual(sigma_points[:, index], self._state)
            measurement_residual = measurement_points[:, index] - measurement_mean
            measurement_residual[3] = self._wrap_angle(measurement_residual[3])
            measurement_covariance += weight * np.outer(
                measurement_residual, measurement_residual
            )
            cross_covariance += weight * np.outer(state_residual, measurement_residual)

        try:
            gain = np.linalg.solve(measurement_covariance, cross_covariance.T).T
        except np.linalg.LinAlgError as exc:
            raise ValueError("measurement covariance is singular") from exc

        innovation = observed - measurement_mean
        innovation[3] = self._wrap_angle(innovation[3])
        self._state += gain @ innovation
        self._state[6] = self._wrap_angle(self._state[6])
        self._covariance -= gain @ measurement_covariance @ gain.T
        self._covariance = (self._covariance + self._covariance.T) * 0.5
        self._validate_covariance(self._covariance, "updated covariance")
        return self.state