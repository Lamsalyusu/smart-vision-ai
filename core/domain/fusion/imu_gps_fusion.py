from __future__ import annotations

import math

import numpy as np


class ErrorStateKalmanFilter:
    """Tightly-Coupled Error-State Kalman Filter (ESKF) for IMU and GPS/Visual Odometry Fusion."""

    def __init__(self, accel_noise: float = 0.1, gyro_noise: float = 0.01) -> None:
        # Nominal state: position (3), velocity (3), orientation quaternion (4), accel_bias (3), gyro_bias (3)
        self.p = np.zeros(3, dtype=np.float64)
        self.v = np.zeros(3, dtype=np.float64)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.ba = np.zeros(3, dtype=np.float64)
        self.bg = np.zeros(3, dtype=np.float64)

        # 15-dimensional error covariance matrix: [delta_p, delta_v, delta_theta, delta_ba, delta_bg]
        self.cov = np.eye(15, dtype=np.float64) * 1e-3
        self.q_noise = np.eye(12, dtype=np.float64) * (accel_noise ** 2)

    def predict_imu(self, accel: np.ndarray, gyro: np.ndarray, dt: float) -> None:
        """Propagate nominal kinematics and error state covariance using IMU measurements."""
        # Subtract biases
        a_unbiased = accel - self.ba
        w_unbiased = gyro - self.bg

        # Rotate acceleration to world frame (with gravity compensation)
        g = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        # Quaternion to rotation matrix
        qw, qx, qy, qz = self.q
        r_mat = np.array([
            [1.0 - 2.0*(qy*qy + qz*qz), 2.0*(qx*qy - qz*qw),       2.0*(qx*qz + qy*qw)],
            [2.0*(qx*qy + qz*qw),       1.0 - 2.0*(qx*qx + qz*qz), 2.0*(qy*qz - qx*qw)],
            [2.0*(qx*qz - qy*qw),       2.0*(qy*qz + qx*qw),       1.0 - 2.0*(qx*qx + qy*qy)]
        ])

        accel_world = r_mat @ a_unbiased + g

        # Update position and velocity
        self.p += self.v * dt + 0.5 * accel_world * (dt ** 2)
        self.v += accel_world * dt

        # Update quaternion
        angle = np.linalg.norm(w_unbiased) * dt
        if angle > 1e-7:
            axis = w_unbiased / np.linalg.norm(w_unbiased)
            dq = np.array([math.cos(angle * 0.5), *(axis * math.sin(angle * 0.5))])
            # Quat multiply dq * q
            w1, x1, y1, z1 = dq
            w2, x2, y2, z2 = self.q
            self.q = np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ])
            self.q /= np.linalg.norm(self.q)

    def update_position_measurement(self, pos_measured: np.ndarray, noise_std: float = 0.5) -> None:
        """Update filter state using external position measurement (GPS / Visual Odometry)."""
        # Residual
        delta_z = pos_measured - self.p

        # Measurement matrix H (3x15): only position error observed
        h = np.zeros((3, 15), dtype=np.float64)
        h[:3, :3] = np.eye(3)

        v_noise = np.eye(3, dtype=np.float64) * (noise_std ** 2)
        s = h @ self.cov @ h.T + v_noise
        k_gain = self.cov @ h.T @ np.linalg.inv(s)

        # Error state correction
        delta_x = k_gain @ delta_z

        # Correct nominal state
        self.p += delta_x[:3]
        self.v += delta_x[3:6]
        self.ba += delta_x[9:12]
        self.bg += delta_x[12:15]

        # Covariance update
        self.cov = (np.eye(15) - k_gain @ h) @ self.cov
