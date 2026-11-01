from __future__ import annotations

import math

import numpy as np


class SO3:
    """Special Orthogonal Group SO(3) representing 3D rotations and its Lie algebra so(3)."""

    @staticmethod
    def hat(omega: np.ndarray) -> np.ndarray:
        """Skew-symmetric matrix operator (wedge/hat): maps R^3 vector to 3x3 so(3) Lie algebra."""
        w1, w2, w3 = float(omega[0]), float(omega[1]), float(omega[2])
        return np.array([
            [ 0.0, -w3,   w2],
            [ w3,   0.0, -w1],
            [-w2,   w1,   0.0]
        ], dtype=np.float64)

    @staticmethod
    def vee(omega_hat: np.ndarray) -> np.ndarray:
        """Inverse hat operator: maps 3x3 so(3) skew-symmetric matrix back to R^3 vector."""
        return np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]], dtype=np.float64)

    @classmethod
    def exp(cls, omega: np.ndarray) -> np.ndarray:
        """Exponential map: maps tangent vector omega in so(3) to rotation matrix R in SO(3) via Rodrigues formula."""
        theta = np.linalg.norm(omega)
        if theta < 1e-7:
            # Taylor series expansion for numerical stability near origin
            return np.eye(3) + cls.hat(omega)

        axis = omega / theta
        k = cls.hat(axis)
        # Rodrigues formula: exp(omega) = I + sin(theta)*K + (1 - cos(theta))*K^2
        return np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)

    @classmethod
    def log(cls, r_mat: np.ndarray) -> np.ndarray:
        """Logarithmic map: maps rotation matrix R in SO(3) to tangent vector omega in so(3)."""
        trace = np.trace(r_mat)
        cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        theta = math.acos(cos_theta)

        if theta < 1e-7:
            return cls.vee(r_mat - np.eye(3))

        sin_theta = math.sin(theta)
        if abs(sin_theta) < 1e-6:
            # Near pi singularity: use symmetric diagonal elements
            diag = np.diagonal(r_mat)
            idx = np.argmax(diag)
            col = r_mat[:, idx]
            col[idx] += 1.0
            axis = col / np.linalg.norm(col)
            return axis * theta

        skew = (r_mat - r_mat.T) / (2.0 * sin_theta)
        return cls.vee(skew) * theta

    @classmethod
    def left_jacobian(cls, omega: np.ndarray) -> np.ndarray:
        """Left Jacobian of SO(3): Jl(omega) = I + (1 - cos(theta))/theta^2 * K + (theta - sin(theta))/theta^3 * K^2."""
        theta = np.linalg.norm(omega)
        if theta < 1e-6:
            return np.eye(3) + 0.5 * cls.hat(omega)

        k = cls.hat(omega)
        k2 = k @ k
        theta2 = theta * theta
        theta3 = theta2 * theta

        c1 = (1.0 - math.cos(theta)) / theta2
        c2 = (theta - math.sin(theta)) / theta3
        return np.eye(3) + c1 * k + c2 * k2


class SE3:
    """Special Euclidean Group SE(3) representing 3D rigid poses [R | t] and its Lie algebra se(3)."""

    def __init__(self, rotation: np.ndarray, translation: np.ndarray) -> None:
        self.r = np.array(rotation, dtype=np.float64)
        self.t = np.array(translation, dtype=np.float64).reshape(3)

    @property
    def matrix(self) -> np.ndarray:
        """4x4 homogeneous transformation matrix."""
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = self.r
        mat[:3, 3] = self.t
        return mat

    def inverse(self) -> SE3:
        """Inverse pose: T^-1 = [R^T | -R^T * t]."""
        r_inv = self.r.T
        t_inv = -r_inv @ self.t
        return SE3(r_inv, t_inv)

    def compose(self, other: SE3) -> SE3:
        """Group composition: T1 * T2 = [R1*R2 | R1*t2 + t1]."""
        r_new = self.r @ other.r
        t_new = self.r @ other.t + self.t
        return SE3(r_new, t_new)

    def transform_point(self, point_3d: np.ndarray) -> np.ndarray:
        """Applies rigid transformation to 3D point: p' = R * p + t."""
        return self.r @ point_3d + self.t

    @classmethod
    def exp(cls, xi: np.ndarray) -> SE3:
        """Exponential map: maps 6D tangent vector xi = [v, omega] in se(3) to SE(3)."""
        v = xi[:3]
        omega = xi[3:]
        theta = np.linalg.norm(omega)

        if theta < 1e-7:
            return cls(np.eye(3), v)

        r = SO3.exp(omega)
        jl = SO3.left_jacobian(omega)
        t = jl @ v
        return cls(r, t)

    def log(self) -> np.ndarray:
        """Logarithmic map: maps SE(3) to 6D tangent vector xi = [v, omega]."""
        omega = SO3.log(self.r)
        theta = np.linalg.norm(omega)

        if theta < 1e-7:
            jl_inv = np.eye(3) - 0.5 * SO3.hat(omega)
        else:
            k = SO3.hat(omega)
            k2 = k @ k
            theta2 = theta * theta
            # Analytical inverse of left Jacobian
            coeff = (1.0 / theta2) - ((1.0 + math.cos(theta)) / (2.0 * theta * math.sin(theta)))
            jl_inv = np.eye(3) - 0.5 * k + coeff * k2

        v = jl_inv @ self.t
        return np.concatenate([v, omega])

    def adjoint(self) -> np.ndarray:
        """6x6 Adjoint representation of SE(3): Ad(T) = [R,  hat(t)*R; 0,  R]."""
        adj = np.zeros((6, 6), dtype=np.float64)
        adj[:3, :3] = self.r
        adj[:3, 3:] = SO3.hat(self.t) @ self.r
        adj[3:, 3:] = self.r
        return adj
