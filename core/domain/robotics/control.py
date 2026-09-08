from __future__ import annotations

import math
from typing import Tuple
import numpy as np


class PIDController:
    """Discrete Proportional-Integral-Derivative (PID) controller with anti-windup."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        integral_limit: float = 100.0,
        output_limit: float = 100.0,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit

        self._integral = 0.0
        self._prev_error: float | None = None

    def update(self, setpoint: float, measured: float, dt: float) -> float:
        """Compute control effort."""
        error = setpoint - measured
        dt = max(1e-4, dt)

        # Proportional term
        p = self.kp * error

        # Integral term with clamping
        self._integral += error * dt
        self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))
        i = self.ki * self._integral

        # Derivative term
        if self._prev_error is None:
            d = 0.0
        else:
            d = self.kd * (error - self._prev_error) / dt
        self._prev_error = error

        u = p + i + d
        return max(-self.output_limit, min(self.output_limit, u))

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None


class LinearQuadraticRegulator:
    """Discrete-time LQR optimal state-feedback controller solving the Algebraic Riccati Equation (DARE)."""

    @staticmethod
    def solve_dare(a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray, max_iter: int = 150, tol: float = 1e-6) -> np.ndarray:
        """Solve Discrete Algebraic Riccati Equation: P = A^T * P * A - (A^T * P * B) * (R + B^T * P * B)^-1 * (B^T * P * A) + Q."""
        p = q.copy()
        for _ in range(max_iter):
            bp = b.T @ p
            r_bpb = r + bp @ b
            k_term = np.linalg.solve(r_bpb, bp @ a)
            p_next = a.T @ p @ a - a.T @ p @ b @ k_term + q
            if np.max(np.abs(p_next - p)) < tol:
                p = p_next
                break
            p = p_next
        return p

    @classmethod
    def compute_gain(cls, a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Computes optimal feedback gain matrix K: u = -K * x."""
        p = cls.solve_dare(a, b, q, r)
        k = np.linalg.inv(r + b.T @ p @ b) @ (b.T @ p @ a)
        return k
