from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class VoxelGridMetadata:
    origin: np.ndarray      # [min_x, min_y, min_z]
    voxel_size: float
    dimensions: tuple[int, int, int]  # (nx, ny, nz)


class VolumetricTSDFBuilder:
    """Constructs a 3D Truncated Signed Distance Field (TSDF) from an oriented point cloud."""

    def __init__(
        self,
        voxel_size: float = 0.05,
        truncation_distance: float = 0.15,
        padding: float = 0.1,
    ) -> None:
        self.voxel_size = max(0.005, voxel_size)
        self.truncation = max(0.01, truncation_distance)
        self.padding = padding

    def build_tsdf(self, points: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, VoxelGridMetadata]:
        """Compute the 3D scalar TSDF volume from 3D points and surface normals.

        Args:
            points: (N, 3) coordinates
            normals: (N, 3) normalized surface normals
        Returns:
            (grid: (nx, ny, nz) float64 array of signed distances, metadata: VoxelGridMetadata)
        """
        if len(points) == 0:
            raise ValueError("Point cloud is empty; cannot construct TSDF")

        min_bound = np.min(points, axis=0) - self.padding
        max_bound = np.max(points, axis=0) + self.padding

        nx = max(4, int(np.ceil((max_bound[0] - min_bound[0]) / self.voxel_size)))
        ny = max(4, int(np.ceil((max_bound[1] - min_bound[1]) / self.voxel_size)))
        nz = max(4, int(np.ceil((max_bound[2] - min_bound[2]) / self.voxel_size)))

        # Constrain memory footprint for fast high-res volumetric processing
        nx = min(nx, 48)
        ny = min(ny, 48)
        nz = min(nz, 48)

        metadata = VoxelGridMetadata(
            origin=min_bound,
            voxel_size=self.voxel_size,
            dimensions=(nx, ny, nz),
        )

        # Generate 3D grid voxel coordinates
        xs = min_bound[0] + np.arange(nx) * self.voxel_size
        ys = min_bound[1] + np.arange(ny) * self.voxel_size
        zs = min_bound[2] + np.arange(nz) * self.voxel_size

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        voxel_positions = np.stack([X, Y, Z], axis=-1)  # (nx, ny, nz, 3)

        np.full((nx, ny, nz), self.truncation, dtype=np.float64)

        # Vectorized projection onto nearest surface points
        flat_voxels = voxel_positions.reshape(-1, 3)
        n_voxels = len(flat_voxels)

        # Process in batches for high cache locality
        batch_size = 2048
        flat_tsdf = np.full(n_voxels, self.truncation, dtype=np.float64)

        for b_start in range(0, n_voxels, batch_size):
            b_end = min(n_voxels, b_start + batch_size)
            batch_voxels = flat_voxels[b_start:b_end]

            # Distances from batch voxels to all points
            diff = batch_voxels[:, np.newaxis, :] - points[np.newaxis, :, :]  # (B, N, 3)
            dist_sq = np.sum(diff ** 2, axis=-1)  # (B, N)

            nearest_indices = np.argmin(dist_sq, axis=-1)  # (B,)
            nearest_dists = np.sqrt(np.min(dist_sq, axis=-1))  # (B,)

            nearest_points = points[nearest_indices]
            nearest_normals = normals[nearest_indices]

            vec_to_voxel = batch_voxels - nearest_points
            dot_sign = np.sum(vec_to_voxel * nearest_normals, axis=-1)
            sign = np.where(dot_sign >= 0.0, 1.0, -1.0)

            signed_dist = sign * nearest_dists
            # Truncate
            truncated = np.clip(signed_dist, -self.truncation, self.truncation)
            flat_tsdf[b_start:b_end] = truncated

        return flat_tsdf.reshape(nx, ny, nz), metadata
