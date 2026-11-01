from __future__ import annotations

import math
from typing import List, Tuple
import numpy as np

from .lie_algebra import SO3, SE3


class VisualOdometryEstimator:
    """Monocular Visual Odometry using Essential Matrix decomposition and PnP pose recovery."""

    def __init__(self, focal_length: float = 718.856, principal_point: Tuple[float, float] = (607.1928, 185.2157)) -> None:
        self.fx = focal_length
        self.fy = focal_length
        self.cx, self.cy = principal_point
        self.current_pose = SE3(np.eye(3), np.zeros(3))

    @property
    def intrinsics(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def normalize_keypoints(self, pts: np.ndarray) -> np.ndarray:
        """Convert pixel points (u, v) to calibrated normalized coordinates."""
        k_inv = np.linalg.inv(self.intrinsics)
        homog = np.column_stack([pts, np.ones(len(pts))])
        return (k_inv @ homog.T).T[:, :2]

    def estimate_essential_matrix_8point(self, pts1_norm: np.ndarray, pts2_norm: np.ndarray) -> np.ndarray:
        """Normalized 8-point algorithm for Essential Matrix E."""
        n = len(pts1_norm)
        if n < 8:
            raise ValueError("Essential matrix estimation requires at least 8 point correspondences")

        a = np.zeros((n, 9), dtype=np.float64)
        for i in range(n):
            x1, y1 = pts1_norm[i]
            x2, y2 = pts2_norm[i]
            a[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.0]

        _, _, vh = np.linalg.svd(a)
        e_raw = vh[-1].reshape((3, 3))

        # Enforce internal constraint: singular values must be [s, s, 0]
        u, s, vh = np.linalg.svd(e_raw)
        s_avg = (s[0] + s[1]) * 0.5
        e = u @ np.diag([s_avg, s_avg, 0.0]) @ vh
        return e

    def decompose_essential_matrix(self, e: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Decompose Essential Matrix into 4 possible (R, t) hypothesis solutions."""
        u, _, vh = np.linalg.svd(e)
        if np.linalg.det(u) < 0:
            u *= -1.0
        if np.linalg.det(vh) < 0:
            vh *= -1.0

        w = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0]
        ], dtype=np.float64)

        r1 = u @ w @ vh
        r2 = u @ w.T @ vh
        t1 = u[:, 2]
        t2 = -u[:, 2]

        return [(r1, t1), (r1, t2), (r2, t1), (r2, t2)]

    def recover_pose(self, pts1: np.ndarray, pts2: np.ndarray) -> SE3:
        """Triangulate points and recover true physical (R, t) by enforcing chierality (positive depth)."""
        pts1_norm = self.normalize_keypoints(pts1)
        pts2_norm = self.normalize_keypoints(pts2)

        e = self.estimate_essential_matrix_8point(pts1_norm, pts2_norm)
        hypotheses = self.decompose_essential_matrix(e)

        best_count = -1
        best_r = np.eye(3)
        best_t = np.zeros(3)

        # Chierality test
        p1 = np.hstack([np.eye(3), np.zeros((3, 1))])

        for r_cand, t_cand in hypotheses:
            p2 = np.hstack([r_cand, t_cand.reshape(3, 1)])
            valid_depth_count = 0

            for i in range(min(50, len(pts1_norm))):
                x1, y1 = pts1_norm[i]
                x2, y2 = pts2_norm[i]

                # Linear triangulation DLT
                a = np.array([
                    x1 * p1[2] - p1[0],
                    y1 * p1[2] - p1[1],
                    x2 * p2[2] - p2[0],
                    y2 * p2[2] - p2[1],
                ])
                _, _, vh = np.linalg.svd(a)
                x_3d = vh[-1]
                x_3d /= max(1e-8, x_3d[3])

                # Check depth in both camera frames
                z1 = x_3d[2]
                z2 = (r_cand @ x_3d[:3] + t_cand)[2]

                if z1 > 0.0 and z2 > 0.0:
                    valid_depth_count += 1

            if valid_depth_count > best_count:
                best_count = valid_depth_count
                best_r = r_cand
                best_t = t_cand

        relative_se3 = SE3(best_r, best_t)
        self.current_pose = self.current_pose.compose(relative_se3)
        return self.current_pose
