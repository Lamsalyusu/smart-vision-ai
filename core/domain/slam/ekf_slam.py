from __future__ import annotations

import math
from typing import Dict, List, Tuple
import numpy as np


class ExtendedKalmanFilterSLAM:
    """Full 2D/3D Extended Kalman Filter SLAM with dynamic landmark state augmentation."""

    def __init__(self, process_noise_std: float = 0.05, measurement_noise_std: float = 0.1) -> None:
        # State vector: [x, y, theta, l1_x, l1_y, l2_x, l2_y, ...]
        self.state = np.zeros(3, dtype=np.float64)
        # Covariance matrix P
        self.p = np.diag([1e-3, 1e-3, 1e-3]).astype(np.float64)

        self.q = np.diag([process_noise_std ** 2, process_noise_std ** 2, (process_noise_std * 0.5) ** 2])
        self.r = np.diag([measurement_noise_std ** 2, (measurement_noise_std * 0.5) ** 2])
        self.landmark_map: Dict[int, int] = {}  # landmark_id -> state_vector_offset

    def predict(self, v: float, omega: float, dt: float) -> None:
        """EKF Motion Model Prediction step for differential drive robot."""
        theta = self.state[2]
        if abs(omega) < 1e-5:
            dx = v * math.cos(theta) * dt
            dy = v * math.sin(theta) * dt
            dtheta = 0.0
            g_robot = np.array([
                [1.0, 0.0, -v * math.sin(theta) * dt],
                [0.0, 1.0,  v * math.cos(theta) * dt],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
        else:
            dx = (-v / omega) * math.sin(theta) + (v / omega) * math.sin(theta + omega * dt)
            dy = (v / omega) * math.cos(theta) - (v / omega) * math.cos(theta + omega * dt)
            dtheta = omega * dt
            g_robot = np.array([
                [1.0, 0.0, (v / omega) * (-math.cos(theta) + math.cos(theta + omega * dt))],
                [0.0, 1.0, (v / omega) * (-math.sin(theta) + math.sin(theta + omega * dt))],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)

        self.state[0] += dx
        self.state[1] += dy
        self.state[2] = (self.state[2] + dtheta + math.pi) % (2.0 * math.pi) - math.pi

        # Expand Jacobian to full state size
        dim = len(self.state)
        g = np.eye(dim, dtype=np.float64)
        g[:3, :3] = g_robot

        # Covariance update: P = G * P * G^T + Q
        self.p = g @ self.p @ g.T
        self.p[:3, :3] += self.q

    def update_measurement(self, landmark_id: int, r_meas: float, phi_meas: float) -> None:
        """EKF Measurement Update with automatic landmark initialization."""
        rx, ry, rtheta = self.state[0], self.state[1], self.state[2]

        if landmark_id not in self.landmark_map:
            lx = rx + r_meas * math.cos(rtheta + phi_meas)
            ly = ry + r_meas * math.sin(rtheta + phi_meas)

            idx = len(self.state)
            self.landmark_map[landmark_id] = idx
            self.state = np.concatenate([self.state, [lx, ly]])

            # Expand covariance matrix
            old_p = self.p
            new_p = np.eye(len(self.state), dtype=np.float64) * 10.0
            new_p[:len(old_p), :len(old_p)] = old_p
            self.p = new_p
            return

        # Known landmark update
        idx = self.landmark_map[landmark_id]
        lx = self.state[idx]
        ly = self.state[idx + 1]

        delta_x = lx - rx
        delta_y = ly - ry
        q = delta_x ** 2 + delta_y ** 2
        sqrt_q = math.sqrt(max(1e-8, q))

        # Expected measurement
        z_hat = np.array([
            sqrt_q,
            (math.atan2(delta_y, delta_x) - rtheta + math.pi) % (2.0 * math.pi) - math.pi
        ])

        # Measurement Jacobian H
        dim = len(self.state)
        h = np.zeros((2, dim), dtype=np.float64)

        h[0, 0] = -delta_x / sqrt_q
        h[0, 1] = -delta_y / sqrt_q
        h[0, 2] = 0.0
        h[0, idx] = delta_x / sqrt_q
        h[0, idx + 1] = delta_y / sqrt_q

        h[1, 0] = delta_y / q
        h[1, 1] = -delta_x / q
        h[1, 2] = -1.0
        h[1, idx] = -delta_y / q
        h[1, idx + 1] = delta_x / q

        # Innovation residual
        y_residual = np.array([
            r_meas - z_hat[0],
            (phi_meas - z_hat[1] + math.pi) % (2.0 * math.pi) - math.pi
        ])

        # Innovation covariance S = H * P * H^T + R
        s = h @ self.p @ h.T + self.r
        # Kalman Gain K = P * H^T * S^-1
        k_gain = self.p @ h.T @ np.linalg.inv(s)

        # State and covariance update
        self.state += k_gain @ y_residual
        self.state[2] = (self.state[2] + math.pi) % (2.0 * math.pi) - math.pi
        self.p = (np.eye(dim) - k_gain @ h) @ self.p
