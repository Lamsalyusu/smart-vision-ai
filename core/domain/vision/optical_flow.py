from __future__ import annotations

import math
from typing import Tuple
import numpy as np

from .filters import ImageFilterKernel, SpatialConvolutionEngine


class LucasKanadeOpticalFlow:
    """Differential Lucas-Kanade optical flow estimator with spatial window least-squares."""

    def __init__(self, window_size: int = 15, regularization: float = 1e-4) -> None:
        self.w_size = window_size
        self.reg = regularization

    def compute_flow(self, prev_img: np.ndarray, next_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute (u, v) dense optical flow fields.

        Args:
            prev_img: (H, W) previous grayscale frame
            next_img: (H, W) current grayscale frame
        Returns:
            (u, v): flow velocities in x and y directions
        """
        i1 = prev_img.astype(np.float64)
        i2 = next_img.astype(np.float64)
        h, w = i1.shape

        # Temporal difference It
        it = i2 - i1
        # Spatial gradients Ix, Iy averaged across the two frames
        avg_img = (i1 + i2) * 0.5
        ix = SpatialConvolutionEngine.convolve2d(avg_img, ImageFilterKernel.sobel_x())
        iy = SpatialConvolutionEngine.convolve2d(avg_img, ImageFilterKernel.sobel_y())

        u = np.zeros((h, w), dtype=np.float64)
        v = np.zeros((h, w), dtype=np.float64)
        half = self.w_size // 2

        # Solve [Ix^2   Ix*Iy] [u] = -[Ix*It]
        #       [Ix*Iy  Iy^2 ] [v]   -[Iy*It]
        # in local neighborhood windows
        g_kernel = np.ones((self.w_size, self.w_size), dtype=np.float64)
        s_ix2 = SpatialConvolutionEngine.convolve2d(ix * ix, g_kernel)
        s_iy2 = SpatialConvolutionEngine.convolve2d(iy * iy, g_kernel)
        s_ixiy = SpatialConvolutionEngine.convolve2d(ix * iy, g_kernel)
        s_ixit = SpatialConvolutionEngine.convolve2d(ix * it, g_kernel)
        s_iyit = SpatialConvolutionEngine.convolve2d(iy * it, g_kernel)

        det = (s_ix2 + self.reg) * (s_iy2 + self.reg) - (s_ixiy ** 2)
        valid = det > 1e-6

        # 2x2 matrix inversion
        u[valid] = (-s_iy2[valid] * s_ixit[valid] + s_ixiy[valid] * s_iyit[valid]) / det[valid]
        v[valid] = (s_ixiy[valid] * s_ixit[valid] - s_ix2[valid] * s_iyit[valid]) / det[valid]

        return u, v


class HornSchunckOpticalFlow:
    """Global variational Horn-Schunck optical flow with smoothness constraint."""

    def __init__(self, alpha: float = 1.0, max_iter: int = 50, tol: float = 1e-4) -> None:
        self.alpha = max(1e-4, alpha)
        self.max_iter = max_iter
        self.tol = tol

    def compute_flow(self, prev_img: np.ndarray, next_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        i1 = prev_img.astype(np.float64)
        i2 = next_img.astype(np.float64)
        h, w = i1.shape

        # Gradients Ix, Iy, It using 4-neighborhood forward-backward differences
        ix = np.zeros((h, w), dtype=np.float64)
        iy = np.zeros((h, w), dtype=np.float64)
        it = i2 - i1

        ix[:, :-1] = np.diff((i1 + i2) * 0.5, axis=1)
        iy[:-1, :] = np.diff((i1 + i2) * 0.5, axis=0)

        u = np.zeros((h, w), dtype=np.float64)
        v = np.zeros((h, w), dtype=np.float64)

        # 4-neighbor averaging kernel
        kernel = np.array([
            [0.0, 0.25, 0.0],
            [0.25, 0.0, 0.25],
            [0.0, 0.25, 0.0]
        ], dtype=np.float64)

        alpha_sq = self.alpha ** 2
        denom = alpha_sq + ix ** 2 + iy ** 2

        for _ in range(self.max_iter):
            u_avg = SpatialConvolutionEngine.convolve2d(u, kernel)
            v_avg = SpatialConvolutionEngine.convolve2d(v, kernel)

            numerator = ix * u_avg + iy * v_avg + it
            update = numerator / denom

            new_u = u_avg - ix * update
            new_v = v_avg - iy * update

            diff = np.max(np.abs(new_u - u)) + np.max(np.abs(new_v - v))
            u = new_u
            v = new_v

            if diff < self.tol:
                break

        return u, v
