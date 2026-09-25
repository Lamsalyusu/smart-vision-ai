from __future__ import annotations

import numpy as np


class ImageSegmentationEngine:
    """Image segmentation via thresholding and region-based algorithms."""

    @staticmethod
    def otsu_threshold(image: np.ndarray) -> tuple[float, np.ndarray]:
        """Otsu thresholding: maximizes inter-class variance between background and foreground."""
        gray = image.astype(np.uint8) if image.ndim == 2 else np.mean(image, axis=-1).astype(np.uint8)
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        total_pixels = gray.size

        current_max = 0.0
        best_threshold = 0

        sum_total = np.dot(np.arange(256), hist)
        sum_b = 0.0
        weight_b = 0

        for t in range(256):
            weight_b += hist[t]
            if weight_b == 0:
                continue
            weight_f = total_pixels - weight_b
            if weight_f == 0:
                break

            sum_b += t * hist[t]
            mean_b = sum_b / weight_b
            mean_f = (sum_total - sum_b) / weight_f

            # Inter-class variance: sigma_b^2 = w_b * w_f * (mu_b - mu_f)^2
            var_between = float(weight_b) * float(weight_f) * ((mean_b - mean_f) ** 2)

            if var_between > current_max:
                current_max = var_between
                best_threshold = t

        binary = (gray >= best_threshold).astype(np.uint8) * 255
        return float(best_threshold), binary

    @staticmethod
    def adaptive_threshold(
        image: np.ndarray,
        block_size: int = 15,
        c_constant: float = 5.0,
        method: str = "gaussian",
    ) -> np.ndarray:
        """Locally adaptive thresholding for uneven illumination."""
        if block_size % 2 == 0:
            block_size += 1

        img = image.astype(np.float64)
        if method == "gaussian":
            from .filters import ImageFilterKernel, SpatialConvolutionEngine
            kernel = ImageFilterKernel.gaussian_2d(block_size, sigma=block_size / 6.0)
            local_mean = SpatialConvolutionEngine.convolve2d(img, kernel)
        else:
            from .filters import SpatialConvolutionEngine
            kernel = np.ones((block_size, block_size), dtype=np.float64) / (block_size ** 2)
            local_mean = SpatialConvolutionEngine.convolve2d(img, kernel)

        thresh = local_mean - c_constant
        return (img > thresh).astype(np.uint8) * 255

    @staticmethod
    def region_growing(image: np.ndarray, seed: tuple[int, int], threshold: float = 15.0) -> np.ndarray:
        """Connected component region growing starting from seed pixel coordinate."""
        h, w = image.shape
        visited = np.zeros((h, w), dtype=bool)
        segmented = np.zeros((h, w), dtype=np.uint8)

        sy, sx = seed
        seed_val = float(image[sy, sx])
        queue = [(sy, sx)]
        visited[sy, sx] = True

        while queue:
            cy, cx = queue.pop(0)
            segmented[cy, cx] = 255

            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    visited[ny, nx] = True
                    if abs(float(image[ny, nx]) - seed_val) <= threshold:
                        queue.append((ny, nx))

        return segmented
