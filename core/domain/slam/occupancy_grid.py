from __future__ import annotations

import math
from typing import Tuple
import numpy as np


class OccupancyGrid3D:
    """3D Probabilistic Voxel Grid with Log-Odds Bayesian Mapping and 3D Raycasting."""

    def __init__(
        self,
        bounds_min: Tuple[float, float, float] = (-5.0, -5.0, -1.0),
        bounds_max: Tuple[float, float, float] = (5.0, 5.0, 3.0),
        voxel_size: float = 0.1,
        p_occ: float = 0.7,
        p_free: float = 0.4,
    ) -> None:
        self.origin = np.array(bounds_min, dtype=np.float64)
        self.bounds_max = np.array(bounds_max, dtype=np.float64)
        self.voxel_size = max(0.01, voxel_size)

        dims = np.ceil((self.bounds_max - self.origin) / self.voxel_size).astype(np.int32)
        self.nx, self.ny, self.nz = int(dims[0]), int(dims[1]), int(dims[2])

        # Clamp max voxel allocation for performance
        self.nx = min(self.nx, 100)
        self.ny = min(self.ny, 100)
        self.nz = min(self.nz, 40)

        # Log-odds representation: l = log(p / (1 - p))
        self.log_odds = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        self.l_occ = math.log(p_occ / (1.0 - p_occ))
        self.l_free = math.log(p_free / (1.0 - p_free))
        self.l_min = -5.0
        self.l_max = 5.0

    def world_to_voxel(self, pt_world: np.ndarray) -> Tuple[int, int, int]:
        """Convert world coordinates (x, y, z) to discrete grid indices."""
        idx = np.floor((pt_world - self.origin) / self.voxel_size).astype(np.int32)
        return int(idx[0]), int(idx[1]), int(idx[2])

    def voxel_to_world(self, ix: int, iy: int, iz: int) -> np.ndarray:
        return self.origin + (np.array([ix, iy, iz], dtype=np.float64) + 0.5) * self.voxel_size

    def is_in_bounds(self, ix: int, iy: int, iz: int) -> bool:
        return (0 <= ix < self.nx) and (0 <= iy < self.ny) and (0 <= iz < self.nz)

    def integrate_ray(self, origin_world: np.ndarray, endpoint_world: np.ndarray) -> None:
        """Bresenham 3D raycasting: update traversed voxels as free, endpoint as occupied."""
        x0, y0, z0 = self.world_to_voxel(origin_world)
        x1, y1, z1 = self.world_to_voxel(endpoint_world)

        # 3D line drawing
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        dz = abs(z1 - z0)

        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        sz = 1 if z1 > z0 else -1

        cx, cy, cz = x0, y0, z0

        if dx >= dy and dx >= dz:
            p1 = 2 * dy - dx
            p2 = 2 * dz - dx
            while cx != x1:
                if self.is_in_bounds(cx, cy, cz):
                    self.log_odds[cx, cy, cz] = np.clip(self.log_odds[cx, cy, cz] + self.l_free, self.l_min, self.l_max)
                cx += sx
                if p1 >= 0:
                    cy += sy
                    p1 -= 2 * dx
                if p2 >= 0:
                    cz += sz
                    p2 -= 2 * dx
                p1 += 2 * dy
                p2 += 2 * dz
        elif dy >= dx and dy >= dz:
            p1 = 2 * dx - dy
            p2 = 2 * dz - dy
            while cy != y1:
                if self.is_in_bounds(cx, cy, cz):
                    self.log_odds[cx, cy, cz] = np.clip(self.log_odds[cx, cy, cz] + self.l_free, self.l_min, self.l_max)
                cy += sy
                if p1 >= 0:
                    cx += sx
                    p1 -= 2 * dy
                if p2 >= 0:
                    cz += sz
                    p2 -= 2 * dy
                p1 += 2 * dx
                p2 += 2 * dz
        else:
            p1 = 2 * dy - dz
            p2 = 2 * dx - dz
            while cz != z1:
                if self.is_in_bounds(cx, cy, cz):
                    self.log_odds[cx, cy, cz] = np.clip(self.log_odds[cx, cy, cz] + self.l_free, self.l_min, self.l_max)
                cz += sz
                if p1 >= 0:
                    cy += sy
                    p1 -= 2 * dz
                if p2 >= 0:
                    cx += sx
                    p2 -= 2 * dz
                p1 += 2 * dy
                p2 += 2 * dx

        # Update endpoint as occupied
        if self.is_in_bounds(x1, y1, z1):
            self.log_odds[x1, y1, z1] = np.clip(self.log_odds[x1, y1, z1] + self.l_occ, self.l_min, self.l_max)

    def get_probability_grid(self) -> np.ndarray:
        """Convert log-odds grid to standard probabilities: p = 1 / (1 + exp(-l))."""
        return 1.0 / (1.0 + np.exp(-self.log_odds))
