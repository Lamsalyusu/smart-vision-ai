from __future__ import annotations

import numpy as np


class IterativeClosestPoint:
    """Rigid 3D Point-to-Point and Point-to-Plane ICP Registration."""

    @staticmethod
    def find_nearest_neighbors(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Find Euclidean closest point correspondences from src to dst."""
        diff = src[:, np.newaxis, :] - dst[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        indices = np.argmin(dist_sq, axis=-1)
        distances = np.sqrt(np.min(dist_sq, axis=-1))
        return indices, distances

    @classmethod
    def register_point_to_point(
        cls,
        source: np.ndarray,
        target: np.ndarray,
        max_iterations: int = 50,
        tolerance: float = 1e-5,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Solves optimal rigid rotation R and translation t using Horn's SVD quaternion/matrix method."""
        src = source.copy()
        r_cum = np.eye(3, dtype=np.float64)
        t_cum = np.zeros(3, dtype=np.float64)

        prev_error = float("inf")

        for _ in range(max_iterations):
            indices, dists = cls.find_nearest_neighbors(src, target)
            matched_target = target[indices]
            mean_error = float(np.mean(dists))

            if abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error

            # Centroids
            centroid_src = np.mean(src, axis=0)
            centroid_tgt = np.mean(matched_target, axis=0)

            # Center points
            src_centered = src - centroid_src
            tgt_centered = matched_target - centroid_tgt

            # Cross-covariance matrix H
            h = src_centered.T @ tgt_centered
            u, _, vh = np.linalg.svd(h)
            r_step = vh.T @ u.T

            # Special reflection check
            if np.linalg.det(r_step) < 0:
                vh[2, :] *= -1.0
                r_step = vh.T @ u.T

            t_step = centroid_tgt - r_step @ centroid_src

            # Update transformed points and cumulative transformation
            src = (r_step @ src.T).T + t_step
            r_cum = r_step @ r_cum
            t_cum = r_step @ t_cum + t_step

        return r_cum, t_cum, prev_error
