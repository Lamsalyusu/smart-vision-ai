import math
from dataclasses import dataclass

import numpy as np

from core.domain.spatial.transforms import Vector3


@dataclass(frozen=True, slots=True)
class BoundingBox3D:
    min_bound: Vector3
    max_bound: Vector3

    @property
    def extents(self) -> Vector3:
        return Vector3(
            self.max_bound.x - self.min_bound.x,
            self.max_bound.y - self.min_bound.y,
            self.max_bound.z - self.min_bound.z,
        )


class PointCloud3D:
    """Dense 3D point cloud container with geometric filtering and downsampling primitives."""

    def __init__(
        self,
        points: np.ndarray,
        colors: np.ndarray | None = None,
    ) -> None:
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points array must be of shape (N, 3)")
        if not np.isfinite(pts).all():
            raise ValueError("point coordinates must be finite")

        self._points = pts
        if colors is not None:
            cols = np.asarray(colors, dtype=np.uint8)
            if cols.shape != pts.shape:
                raise ValueError("colors shape must match points shape (N, 3)")
            self._colors: np.ndarray | None = cols
        else:
            self._colors = None

    @property
    def points(self) -> np.ndarray:
        return self._points

    @property
    def colors(self) -> np.ndarray | None:
        return self._colors

    def __len__(self) -> int:
        return int(self._points.shape[0])

    def get_bounding_box(self) -> BoundingBox3D:
        if len(self) == 0:
            return BoundingBox3D(Vector3(0, 0, 0), Vector3(0, 0, 0))
        min_xyz = np.min(self._points, axis=0)
        max_xyz = np.max(self._points, axis=0)
        return BoundingBox3D(
            min_bound=Vector3(
                float(min_xyz[0]), float(min_xyz[1]), float(min_xyz[2])
            ),
            max_bound=Vector3(
                float(max_xyz[0]), float(max_xyz[1]), float(max_xyz[2])
            ),
        )

    def voxel_downsample(self, voxel_size_m: float) -> "PointCloud3D":
        """Downsample point cloud by voxel grid centroid aggregation."""
        if not math.isfinite(voxel_size_m) or voxel_size_m <= 0:
            raise ValueError("voxel size must be finite and positive")
        if len(self) == 0:
            return PointCloud3D(self._points.copy())

        # Quantize points to discrete voxel coordinates
        voxel_indices = np.floor(self._points / voxel_size_m).astype(np.int64)

        # Hash coordinates for group by
        # Use structured array view for efficient unique key grouping
        unique_keys, inverse_indices = np.unique(
            voxel_indices, axis=0, return_inverse=True
        )

        num_voxels = unique_keys.shape[0]
        downsampled_points = np.zeros((num_voxels, 3), dtype=np.float32)
        counts = np.zeros(num_voxels, dtype=np.float32)

        np.add.at(downsampled_points, inverse_indices, self._points)
        np.add.at(counts, inverse_indices, 1.0)
        downsampled_points /= counts[:, None]

        downsampled_colors: np.ndarray | None = None
        if self._colors is not None:
            color_sums = np.zeros((num_voxels, 3), dtype=np.float32)
            np.add.at(color_sums, inverse_indices, self._colors.astype(np.float32))
            downsampled_colors = np.clip(
                color_sums / counts[:, None], 0, 255
            ).astype(np.uint8)

        return PointCloud3D(downsampled_points, downsampled_colors)

    def remove_statistical_outliers(
        self,
        k_neighbors: int = 16,
        std_ratio: float = 2.0,
    ) -> "PointCloud3D":
        """Filter noisy outlier points whose mean neighbor distance exceeds threshold."""
        if len(self) < k_neighbors or k_neighbors <= 1:
            return PointCloud3D(
                self._points.copy(),
                self._colors.copy() if self._colors is not None else None,
            )

        pts = self._points
        # Compute distances to neighbors using sample subsample or k-d chunking
        # For lightweight robust filtering: calculate sample distance matrix
        num_pts = len(self)
        mean_dists = np.zeros(num_pts, dtype=np.float32)

        # Vectorized batch neighbor distance approximation
        step = max(1, num_pts // 1000)
        for i in range(0, num_pts, 256):
            batch = pts[i : i + 256]
            # Pairwise squared distances to subset
            diff = batch[:, None, :] - pts[::step, :]
            sq_dists = np.sum(diff**2, axis=-1)
            # Take k smallest
            k_actual = min(k_neighbors, sq_dists.shape[1] - 1)
            part = np.partition(sq_dists, k_actual, axis=1)[:, 1 : k_actual + 1]
            mean_dists[i : i + 256] = np.mean(np.sqrt(part), axis=1)

        global_mean = float(np.mean(mean_dists))
        global_std = float(np.std(mean_dists))
        threshold = global_mean + std_ratio * global_std

        inliers = mean_dists <= threshold
        filtered_pts = self._points[inliers]
        filtered_cols = (
            self._colors[inliers] if self._colors is not None else None
        )

        return PointCloud3D(filtered_pts, filtered_cols)
