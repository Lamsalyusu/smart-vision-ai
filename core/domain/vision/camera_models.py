from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(slots=True)
class PinholeCameraModel:
    """Standard Pinhole Camera Model with Brown-Conrady radial & tangential lens distortion."""

    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def project_3d_to_pixel(self, pt_camera: np.ndarray) -> np.ndarray:
        """Projects 3D camera frame point (X, Y, Z) to distorted 2D pixel coordinate (u, v)."""
        x, y, z = pt_camera[0], pt_camera[1], pt_camera[2]
        if abs(z) < 1e-6:
            z = 1e-6

        # Normalized image plane coordinates
        xn = x / z
        yn = y / z

        # Radial distortion
        r2 = xn * xn + yn * yn
        r4 = r2 * r2
        r6 = r4 * r2
        radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6

        # Tangential distortion
        dx = 2.0 * self.p1 * xn * yn + self.p2 * (r2 + 2.0 * xn * xn)
        dy = self.p1 * (r2 + 2.0 * yn * yn) + 2.0 * self.p2 * xn * yn

        xd = xn * radial + dx
        yd = yn * radial + dy

        u = self.fx * xd + self.cx
        v = self.fy * yd + self.cy
        return np.array([u, v], dtype=np.float64)

    def unproject_pixel_to_ray(self, u: float, v: float, max_iter: int = 10) -> np.ndarray:
        """Unprojects distorted pixel (u, v) into normalized 3D unit ray via iterative Gauss-Newton inversion."""
        xd = (u - self.cx) / self.fx
        yd = (v - self.cy) / self.fy

        # Initial estimate
        xn = xd
        yn = yd

        # Invert distortion iteratively
        for _ in range(max_iter):
            r2 = xn * xn + yn * yn
            radial = 1.0 + self.k1 * r2 + self.k2 * (r2 ** 2) + self.k3 * (r2 ** 3)
            dx = 2.0 * self.p1 * xn * yn + self.p2 * (r2 + 2.0 * xn * xn)
            dy = self.p1 * (r2 + 2.0 * yn * yn) + 2.0 * self.p2 * xn * yn

            xn = (xd - dx) / radial
            yn = (yd - dy) / radial

        ray = np.array([xn, yn, 1.0], dtype=np.float64)
        return ray / np.linalg.norm(ray)


@dataclass(slots=True)
class FisheyeKannalaBrandtModel:
    """Kannala-Brandt generic fisheye camera model for ultra-wide field of view lenses."""

    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0

    def project(self, pt_camera: np.ndarray) -> np.ndarray:
        x, y, z = pt_camera[0], pt_camera[1], pt_camera[2]
        r = math.sqrt(x * x + y * y)
        if r < 1e-8:
            return np.array([self.cx, self.cy], dtype=np.float64)

        theta = math.atan2(r, z)
        theta_d = (
            theta +
            self.k1 * (theta ** 3) +
            self.k2 * (theta ** 5) +
            self.k3 * (theta ** 7) +
            self.k4 * (theta ** 9)
        )

        scale = theta_d / r
        u = self.fx * (x * scale) + self.cx
        v = self.fy * (y * scale) + self.cy
        return np.array([u, v], dtype=np.float64)
