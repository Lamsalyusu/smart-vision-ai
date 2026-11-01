from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union
import numpy as np


class ImageFilterKernel:
    """Pre-computed convolution kernels for digital signal and image filtering."""

    @staticmethod
    def gaussian_1d(kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """Construct 1D normalized Gaussian kernel."""
        if kernel_size % 2 == 0:
            kernel_size += 1
        half = kernel_size // 2
        x = np.arange(-half, half + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (x / max(1e-6, sigma)) ** 2)
        return kernel / np.sum(kernel)

    @staticmethod
    def gaussian_2d(kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """Construct separable 2D normalized Gaussian kernel."""
        k1d = ImageFilterKernel.gaussian_1d(kernel_size, sigma)
        return np.outer(k1d, k1d)

    @staticmethod
    def sobel_x() -> np.ndarray:
        """Sobel 3x3 horizontal gradient operator."""
        return np.array([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]
        ], dtype=np.float64)

    @staticmethod
    def sobel_y() -> np.ndarray:
        """Sobel 3x3 vertical gradient operator."""
        return np.array([
            [-1.0, -2.0, -1.0],
            [ 0.0,  0.0,  0.0],
            [ 1.0,  2.0,  1.0]
        ], dtype=np.float64)

    @staticmethod
    def scharr_x() -> np.ndarray:
        """Scharr 3x3 horizontal operator with enhanced rotational symmetry."""
        return np.array([
            [-3.0, 0.0, 3.0],
            [-10.0, 0.0, 10.0],
            [-3.0, 0.0, 3.0]
        ], dtype=np.float64) / 16.0

    @staticmethod
    def scharr_y() -> np.ndarray:
        """Scharr 3x3 vertical operator with enhanced rotational symmetry."""
        return np.array([
            [-3.0, -10.0, -3.0],
            [ 0.0,   0.0,  0.0],
            [ 3.0,  10.0,  3.0]
        ], dtype=np.float64) / 16.0

    @staticmethod
    def laplacian_4() -> np.ndarray:
        """4-neighborhood discrete Laplacian kernel."""
        return np.array([
            [ 0.0, -1.0,  0.0],
            [-1.0,  4.0, -1.0],
            [ 0.0, -1.0,  0.0]
        ], dtype=np.float64)

    @staticmethod
    def laplacian_8() -> np.ndarray:
        """8-neighborhood discrete Laplacian kernel."""
        return np.array([
            [-1.0, -1.0, -1.0],
            [-1.0,  8.0, -1.0],
            [-1.0, -1.0, -1.0]
        ], dtype=np.float64)

    @staticmethod
    def log_kernel(kernel_size: int = 9, sigma: float = 1.4) -> np.ndarray:
        """Laplacian of Gaussian (LoG) Mexican hat kernel."""
        if kernel_size % 2 == 0:
            kernel_size += 1
        half = kernel_size // 2
        y, x = np.ogrid[-half:half + 1, -half:half + 1]
        sigma_sq = sigma * sigma
        radius_sq = x * x + y * y
        log_k = -(1.0 - radius_sq / (2.0 * sigma_sq)) * np.exp(-radius_sq / (2.0 * sigma_sq)) / (math.pi * (sigma_sq ** 2))
        return log_k - np.mean(log_k)

    @staticmethod
    def gabor_kernel(
        kernel_size: int = 21,
        sigma: float = 3.0,
        theta: float = 0.0,
        lambd: float = 10.0,
        gamma: float = 0.5,
        psi: float = 0.0,
    ) -> np.ndarray:
        """2D Gabor filter kernel for spatial frequency and orientation selectivity."""
        if kernel_size % 2 == 0:
            kernel_size += 1
        half = kernel_size // 2
        y, x = np.ogrid[-half:half + 1, -half:half + 1]
        x_prime = x * math.cos(theta) + y * math.sin(theta)
        y_prime = -x * math.sin(theta) + y * math.cos(theta)
        sigma_x = sigma
        sigma_y = sigma / max(1e-6, gamma)
        envelope = np.exp(-0.5 * ((x_prime / sigma_x) ** 2 + (y_prime / sigma_y) ** 2))
        carrier = np.cos(2.0 * math.pi * x_prime / max(1e-6, lambd) + psi)
        return envelope * carrier


class SpatialConvolutionEngine:
    """Performs spatial 2D image convolutions with boundary padding modes."""

    @staticmethod
    def convolve2d(
        image: np.ndarray,
        kernel: np.ndarray,
        mode: str = "reflect",
    ) -> np.ndarray:
        """Convolve 2D single-channel image with 2D kernel.

        Args:
            image: (H, W) float array
            kernel: (Kh, Kw) float array
            mode: padding mode ('reflect', 'constant', 'edge')
        Returns:
            convolved: (H, W) float array
        """
        img_h, img_w = image.shape[:2]
        kh, kw = kernel.shape[:2]
        pad_y = kh // 2
        pad_x = kw // 2

        if mode == "reflect":
            padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
        elif mode == "edge":
            padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
        else:
            padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant", constant_values=0.0)

        # Vectorized 2D convolution using sliding window views
        from numpy.lib.stride_tricks import as_strided
        shape = (img_h, img_w, kh, kw)
        strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
        windows = as_strided(padded, shape=shape, strides=strides)

        # Flipped kernel for true mathematical convolution
        flipped_k = np.flipud(np.fliplr(kernel))
        return np.einsum("ijkl,kl->ij", windows, flipped_k)


class BilateralFilter:
    """Edge-preserving bilateral smoothing combining spatial and photometric range kernels."""

    def __init__(self, diameter: int = 7, sigma_color: float = 25.0, sigma_space: float = 3.0) -> None:
        self.radius = max(1, diameter // 2)
        self.sigma_color = max(1e-3, sigma_color)
        self.sigma_space = max(1e-3, sigma_space)
        self._spatial_lut = self._precompute_spatial_weights()

    def _precompute_spatial_weights(self) -> np.ndarray:
        r = self.radius
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        dist_sq = x * x + y * y
        return np.exp(-0.5 * dist_sq / (self.sigma_space ** 2))

    def filter(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral filtering to 2D image."""
        h, w = image.shape[:2]
        r = self.radius
        padded = np.pad(image, ((r, r), (r, r)), mode="reflect")
        output = np.zeros_like(image, dtype=np.float64)

        for i in range(h):
            for j in range(w):
                center_val = padded[i + r, j + r]
                window = padded[i:i + 2 * r + 1, j:j + 2 * r + 1]
                color_diff = window - center_val
                range_weights = np.exp(-0.5 * (color_diff / self.sigma_color) ** 2)
                weights = self._spatial_lut * range_weights
                norm = np.sum(weights)
                output[i, j] = np.sum(weights * window) / max(1e-8, norm)

        return output


class CannyEdgeDetector:
    """Multi-stage Canny Edge Detection pipeline."""

    def __init__(
        self,
        low_threshold: float = 50.0,
        high_threshold: float = 150.0,
        gaussian_sigma: float = 1.2,
    ) -> None:
        self.low_thresh = low_threshold
        self.high_thresh = high_threshold
        self.sigma = gaussian_sigma

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Execute full Canny pipeline: Gaussian blur, gradient, non-max suppression, hysteresis."""
        # 1. Noise reduction via Gaussian convolution
        g_kernel = ImageFilterKernel.gaussian_2d(5, self.sigma)
        smoothed = SpatialConvolutionEngine.convolve2d(image.astype(np.float64), g_kernel)

        # 2. Compute image gradients
        gx = SpatialConvolutionEngine.convolve2d(smoothed, ImageFilterKernel.sobel_x())
        gy = SpatialConvolutionEngine.convolve2d(smoothed, ImageFilterKernel.sobel_y())
        magnitude = np.hypot(gx, gy)
        theta = np.arctan2(gy, gx) * (180.0 / math.pi)
        theta[theta < 0] += 180.0

        # 3. Non-maximum suppression along gradient direction
        nms = self._non_maximum_suppression(magnitude, theta)

        # 4. Double threshold and hysteresis edge tracking
        edges = self._hysteresis_threshold(nms)
        return edges

    def _non_maximum_suppression(self, magnitude: np.ndarray, theta: np.ndarray) -> np.ndarray:
        h, w = magnitude.shape
        suppressed = np.zeros((h, w), dtype=np.float64)
        angle = theta

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                a = angle[i, j]
                mag = magnitude[i, j]

                # 0 degrees (East-West)
                if (0 <= a < 22.5) or (157.5 <= a <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                # 45 degrees (North-East, South-West)
                elif 22.5 <= a < 67.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                # 90 degrees (North-South)
                elif 67.5 <= a < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                # 135 degrees (North-West, South-East)
                else:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                if mag >= q and mag >= r:
                    suppressed[i, j] = mag

        return suppressed

    def _hysteresis_threshold(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        res = np.zeros((h, w), dtype=np.uint8)

        strong_i, strong_j = np.where(img >= self.high_thresh)
        weak_i, weak_j = np.where((img >= self.low_thresh) & (img < self.high_thresh))

        res[strong_i, strong_j] = 255
        res[weak_i, weak_j] = 50

        # Propagate edge connectivity
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if res[i, j] == 50:
                    if np.any(res[i - 1:i + 2, j - 1:j + 2] == 255):
                        res[i, j] = 255
                    else:
                        res[i, j] = 0

        return res


class AnisotropicDiffusion:
    """Perona-Malik nonlinear anisotropic diffusion filter for scale-space edge preservation."""

    def __init__(self, n_iter: int = 15, kappa: float = 30.0, gamma: float = 0.15, option: int = 1) -> None:
        self.n_iter = n_iter
        self.kappa = max(1e-3, kappa)
        self.gamma = min(0.25, gamma)
        self.option = option

    def diffuse(self, image: np.ndarray) -> np.ndarray:
        """Iterate Perona-Malik partial differential equation."""
        diffused = image.astype(np.float64).copy()

        for _ in range(self.n_iter):
            # Compute 4-directional spatial differences
            delta_n = np.zeros_like(diffused)
            delta_s = np.zeros_like(diffused)
            delta_e = np.zeros_like(diffused)
            delta_w = np.zeros_like(diffused)

            delta_n[:-1, :] = np.diff(diffused, axis=0)
            delta_s[1:, :] = -np.diff(diffused, axis=0)
            delta_e[:, :-1] = np.diff(diffused, axis=1)
            delta_w[:, 1:] = -np.diff(diffused, axis=1)

            # Conduction coefficients
            if self.option == 1:
                # Exponential flux: favors high-contrast edges over low-contrast
                cn = np.exp(-(delta_n / self.kappa) ** 2)
                cs = np.exp(-(delta_s / self.kappa) ** 2)
                ce = np.exp(-(delta_e / self.kappa) ** 2)
                cw = np.exp(-(delta_w / self.kappa) ** 2)
            else:
                # Quadratic flux: favors wide regions over smaller regions
                cn = 1.0 / (1.0 + (delta_n / self.kappa) ** 2)
                cs = 1.0 / (1.0 + (delta_s / self.kappa) ** 2)
                ce = 1.0 / (1.0 + (delta_e / self.kappa) ** 2)
                cw = 1.0 / (1.0 + (delta_w / self.kappa) ** 2)

            diffused += self.gamma * (cn * delta_n + cs * delta_s + ce * delta_e + cw * delta_w)

        return diffused
