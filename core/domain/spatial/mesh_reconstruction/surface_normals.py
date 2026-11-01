from __future__ import annotations

import numpy as np


class PointCloudNormalEstimator:
    """Estimates oriented surface normals for 3D point clouds using Covariance PCA.

    For each point in the point cloud:
    1. Gathers k-nearest spatial neighbors.
    2. Builds the 3x3 local covariance tensor.
    3. Solves the eigen-decomposition: normal = eigenvector of smallest eigenvalue.
    4. Orients normals consistently towards viewpoint (sensor origin).
    """

    def __init__(self, k_neighbors: int = 10, viewpoint: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self.k = max(3, k_neighbors)
        self.viewpoint = np.array(viewpoint, dtype=np.float64)

    def estimate_normals(self, points: np.ndarray) -> np.ndarray:
        """Estimate (N, 3) normalized surface normals for (N, 3) point cloud.

        Args:
            points: np.ndarray of shape (N, 3)
        Returns:
            normals: np.ndarray of shape (N, 3)
        """
        n_pts = len(points)
        if n_pts < 3:
            return np.zeros((n_pts, 3), dtype=np.float64)

        normals = np.zeros((n_pts, 3), dtype=np.float64)
        k = min(self.k, n_pts)

        # Pairwise Euclidean distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)

        for i in range(n_pts):
            # Find k nearest neighbors
            if k < n_pts:
                neighbor_indices = np.argpartition(dist_sq[i], k)[:k]
            else:
                neighbor_indices = np.arange(n_pts)
            neighbors = points[neighbor_indices]

            # Compute local mean and covariance
            centroid = np.mean(neighbors, axis=0)
            shifted = neighbors - centroid
            cov = np.dot(shifted.T, shifted) / max(1, k - 1)

            # Eigen-decomposition (np.linalg.eigh returns eigenvalues in ascending order)
            _eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normal = eigenvectors[:, 0]  # Smallest eigenvalue vector

            # Consistent orientation towards sensor viewpoint
            to_viewpoint = self.viewpoint - points[i]
            if np.dot(normal, to_viewpoint) < 0.0:
                normal = -normal

            norm = np.linalg.norm(normal)
            normals[i] = normal / max(1e-8, norm)

        return normals
