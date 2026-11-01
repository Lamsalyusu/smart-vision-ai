from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CameraProjectionMatrix:
    camera_id: str
    projection_3x4: np.ndarray  # P = K [R | t]


class StereoTriangulationEngine:
    """Rigorous 3D Reconstruction using Multi-View Epipolar Geometry and DLT SVD."""

    @staticmethod
    def compute_essential_matrix(F: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
        """Compute Essential Matrix E = K2^T * F * K1."""
        return np.dot(K2.T, np.dot(F, K1))

    @staticmethod
    def decompose_essential_matrix(E: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Decompose Essential Matrix into 4 possible (R, t) hypothesis solutions via SVD."""
        U, _S, Vt = np.linalg.svd(E)
        if np.linalg.det(U) < 0:
            U *= -1.0
        if np.linalg.det(Vt) < 0:
            Vt *= -1.0

        W = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=np.float64)

        R1 = np.dot(U, np.dot(W, Vt))
        R2 = np.dot(U, np.dot(W.T, Vt))
        t1 = U[:, 2].reshape(3, 1)
        t2 = -U[:, 2].reshape(3, 1)

        solutions = [
            (R1, t1),
            (R1, t2),
            (R2, t1),
            (R2, t2),
        ]
        return solutions

    @staticmethod
    def triangulate_point_dlt(points_2d: list[tuple[float, float]], cameras: list[CameraProjectionMatrix]) -> np.ndarray:
        """Triangulate a 3D point from multiple 2D camera observations using SVD Direct Linear Transformation (DLT).

        Args:
            points_2d: List of (u, v) normalized or pixel coordinates
            cameras: List of CameraProjectionMatrix with P = 3x4 matrix
        Returns:
            np.ndarray: [X, Y, Z] 3D world coordinate
        """
        if len(points_2d) < 2 or len(points_2d) != len(cameras):
            raise ValueError("Triangulation requires at least 2 camera views matching 2D points")

        num_views = len(points_2d)
        A = np.zeros((2 * num_views, 4), dtype=np.float64)

        for i in range(num_views):
            u, v = points_2d[i]
            P = cameras[i].projection_3x4
            A[2 * i] = u * P[2, :] - P[0, :]
            A[2 * i + 1] = v * P[2, :] - P[1, :]

        # Solve AX = 0 via SVD
        _, _, Vt = np.linalg.svd(A)
        X_homo = Vt[-1]

        if abs(X_homo[3]) < 1e-8:
            return X_homo[:3]

        return X_homo[:3] / X_homo[3]

    @classmethod
    def select_cheirality_valid_solution(
        cls,
        E: np.ndarray,
        K1: np.ndarray,
        K2: np.ndarray,
        pt1: tuple[float, float],
        pt2: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select the unique (R, t) solution among 4 candidates where 3D point has positive depth in both views."""
        solutions = cls.decompose_essential_matrix(E)
        P1 = np.dot(K1, np.hstack([np.eye(3), np.zeros((3, 1))]))
        cam1 = CameraProjectionMatrix("cam1", P1)

        best_R, best_t, best_X = solutions[0][0], solutions[0][1], np.zeros(3)
        max_positive_depths = -1

        for R, t in solutions:
            P2 = np.dot(K2, np.hstack([R, t]))
            cam2 = CameraProjectionMatrix("cam2", P2)
            try:
                X_3d = cls.triangulate_point_dlt([pt1, pt2], [cam1, cam2])
                # Depth in camera 1: Z
                depth1 = X_3d[2]
                # Depth in camera 2: R * X + t
                X_cam2 = np.dot(R, X_3d.reshape(3, 1)) + t
                depth2 = float(X_cam2[2, 0])

                pos_count = (1 if depth1 > 0 else 0) + (1 if depth2 > 0 else 0)
                if pos_count > max_positive_depths:
                    max_positive_depths = pos_count
                    best_R, best_t, best_X = R, t, X_3d
            except (ValueError, np.linalg.LinAlgError):
                continue

        return best_R, best_t, best_X
