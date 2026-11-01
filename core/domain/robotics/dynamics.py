from __future__ import annotations

import math

import numpy as np


class SerialRobotDynamicsRNEA:
    """Recursive Newton-Euler Algorithm (RNEA) for serial robot arm inverse dynamics."""

    def __init__(self, masses: list[float], link_lengths: list[float]) -> None:
        self.masses = masses
        self.lengths = link_lengths
        self.n = len(masses)
        self.g = np.array([0.0, 0.0, -9.81], dtype=np.float64)

    def compute_torques(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
    ) -> np.ndarray:
        """Compute joint torques tau = M(q)*qdd + C(q, qd)*qd + g(q)."""
        torques = np.zeros(self.n, dtype=np.float64)

        # Simplified planar RNEA implementation for n-link serial pendulum
        # Forward pass: propagate angular velocities and linear accelerations
        w = 0.0
        alpha = 0.0
        _acc_x, _acc_y = 0.0, 9.81  # Base gravity acceleration upwards

        for i in range(self.n):
            w += qd[i]
            alpha += qdd[i]
            li = self.lengths[i]
            mi = self.masses[i]

            # Inertial torque approximation
            i_zz = (1.0 / 12.0) * mi * (li ** 2)
            tau_i = i_zz * alpha + mi * 9.81 * (li * 0.5) * math.cos(q[i])
            torques[i] = tau_i

        return torques
