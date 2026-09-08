from __future__ import annotations

import math
from typing import List, Tuple
import numpy as np


class PolynomialTrajectoryGenerator:
    """Smooth joint-space and Cartesian trajectory generation."""

    @staticmethod
    def cubic_spline(
        q0: float,
        q1: float,
        v0: float,
        v1: float,
        duration: float,
    ) -> np.ndarray:
        """Generates coefficients [a0, a1, a2, a3] for cubic trajectory: q(t) = a0 + a1*t + a2*t^2 + a3*t^3."""
        t = max(1e-4, duration)
        a0 = q0
        a1 = v0
        a2 = (3.0 * (q1 - q0) - (2.0 * v0 + v1) * t) / (t ** 2)
        a3 = (-2.0 * (q1 - q0) + (v0 + v1) * t) / (t ** 3)
        return np.array([a0, a1, a2, a3], dtype=np.float64)

    @staticmethod
    def quintic_spline(
        q0: float,
        q1: float,
        v0: float,
        v1: float,
        acc0: float,
        acc1: float,
        duration: float,
    ) -> np.ndarray:
        """Generates coefficients [a0, a1, a2, a3, a4, a5] for minimum-jerk trajectory with continuous acceleration."""
        t = max(1e-4, duration)
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t

        a0 = q0
        a1 = v0
        a2 = 0.5 * acc0

        # Boundary condition matrix for higher terms
        m = np.array([
            [t3,       t4,       t5],
            [3.0 * t2, 4.0 * t3, 5.0 * t4],
            [6.0 * t,  12.0 * t2, 20.0 * t3],
        ], dtype=np.float64)

        b = np.array([
            q1 - a0 - a1 * t - a2 * t2,
            v1 - a1 - 2.0 * a2 * t,
            acc1 - 2.0 * a2,
        ], dtype=np.float64)

        coeffs_345 = np.linalg.solve(m, b)
        return np.array([a0, a1, a2, coeffs_345[0], coeffs_345[1], coeffs_345[2]], dtype=np.float64)

    @staticmethod
    def evaluate_polynomial(coeffs: np.ndarray, t: float) -> Tuple[float, float, float]:
        """Evaluates position q(t), velocity v(t), and acceleration a(t)."""
        deg = len(coeffs) - 1
        pos = sum(coeffs[i] * (t ** i) for i in range(len(coeffs)))
        vel = sum(i * coeffs[i] * (t ** (i - 1)) for i in range(1, len(coeffs)))
        acc = sum(i * (i - 1) * coeffs[i] * (t ** (i - 2)) for i in range(2, len(coeffs)))
        return float(pos), float(vel), float(acc)


class TrapezoidalVelocityProfile:
    """Bang-Coast-Bang velocity profile for actuator position control."""

    def __init__(self, max_vel: float, max_acc: float) -> None:
        self.v_max = abs(max_vel)
        self.a_max = abs(max_acc)

    def plan(self, q0: float, q1: float) -> Tuple[float, float, float]:
        """Returns acceleration time ta, coasting time tc, and total duration tf."""
        d = q1 - q0
        dist = abs(d)

        # Time to reach max velocity
        ta = self.v_max / self.a_max
        dist_acc = self.a_max * (ta ** 2)

        if dist >= dist_acc:
            # Triangular profile reaches v_max and coasts
            tc = (dist - dist_acc) / self.v_max
            tf = 2.0 * ta + tc
        else:
            # Distance too short to reach v_max (pure triangle)
            ta = math.sqrt(dist / self.a_max)
            tc = 0.0
            tf = 2.0 * ta

        return ta, tc, tf
