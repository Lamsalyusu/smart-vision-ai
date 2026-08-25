from __future__ import annotations

import math
from typing import List, Tuple
import numpy as np


class StructuringElement:
    """Mathematical morphology structuring elements (kernels)."""

    @staticmethod
    def rectangle(height: int = 3, width: int = 3) -> np.ndarray:
        return np.ones((height, width), dtype=np.uint8)

    @staticmethod
    def cross(size: int = 3) -> np.ndarray:
        elem = np.zeros((size, size), dtype=np.uint8)
        half = size // 2
        elem[half, :] = 1
        elem[:, half] = 1
        return elem

    @staticmethod
    def ellipse(size: int = 5) -> np.ndarray:
        elem = np.zeros((size, size), dtype=np.uint8)
        half = size // 2
        y, x = np.ogrid[-half:half + 1, -half:half + 1]
        mask = (x * x + y * y) <= (half * half)
        elem[mask] = 1
        return elem


class MathematicalMorphology:
    """Non-linear mathematical morphology operations on binary and grayscale images."""

    @staticmethod
    def erode(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Min-filter erosion operation."""
        kh, kw = kernel.shape
        pad_y = kh // 2
        pad_x = kw // 2
        padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
        h, w = image.shape
        out = np.zeros((h, w), dtype=image.dtype)

        # Sliding window local minimum over active structuring element elements
        k_indices = np.where(kernel == 1)
        for i in range(h):
            for j in range(w):
                neighborhood = padded[i:i + kh, j:j + kw]
                out[i, j] = np.min(neighborhood[k_indices])
        return out

    @staticmethod
    def dilate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Max-filter dilation operation."""
        kh, kw = kernel.shape
        pad_y = kh // 2
        pad_x = kw // 2
        padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
        h, w = image.shape
        out = np.zeros((h, w), dtype=image.dtype)

        k_indices = np.where(kernel == 1)
        for i in range(h):
            for j in range(w):
                neighborhood = padded[i:i + kh, j:j + kw]
                out[i, j] = np.max(neighborhood[k_indices])
        return out

    @classmethod
    def opening(cls, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Erosion followed by dilation to eliminate foreground noise."""
        return cls.dilate(cls.erode(image, kernel), kernel)

    @classmethod
    def closing(cls, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Dilation followed by erosion to bridge gaps and fill holes."""
        return cls.erode(cls.dilate(image, kernel), kernel)

    @classmethod
    def morphological_gradient(cls, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Difference between dilation and erosion: outlines structural contours."""
        dil = cls.dilate(image, kernel)
        ero = cls.erode(image, kernel)
        return np.clip(dil.astype(np.int32) - ero.astype(np.int32), 0, 255).astype(image.dtype)

    @classmethod
    def top_hat(cls, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Difference between image and its opening: isolates elements brighter than surroundings."""
        opened = cls.opening(image, kernel)
        return np.clip(image.astype(np.int32) - opened.astype(np.int32), 0, 255).astype(image.dtype)

    @classmethod
    def black_hat(cls, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Difference between closing and image: isolates elements darker than surroundings."""
        closed = cls.closing(image, kernel)
        return np.clip(closed.astype(np.int32) - image.astype(np.int32), 0, 255).astype(image.dtype)

    @staticmethod
    def hit_or_miss(image: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
        """Hit-or-Miss transform for morphological pattern matching."""
        # Erode image with foreground kernel B1
        c1 = MathematicalMorphology.erode(image, b1)
        # Erode complement of image with background kernel B2
        inv_img = 255 - image if image.dtype == np.uint8 else 1 - image
        c2 = MathematicalMorphology.erode(inv_img, b2)
        return np.bitwise_and(c1, c2)

    @staticmethod
    def skeletonize_zhang_suen(binary_image: np.ndarray) -> np.ndarray:
        """Zhang-Suen 2-iteration parallel thinning algorithm for skeleton extraction."""
        img = (binary_image > 0).astype(np.uint8)
        changing = True
        h, w = img.shape

        while changing:
            changing = False
            # Step 1
            marked = np.zeros_like(img)
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    if img[i, j] != 1:
                        continue
                    p2, p3, p4 = img[i - 1, j], img[i - 1, j + 1], img[i, j + 1]
                    p5, p6, p7 = img[i + 1, j + 1], img[i + 1, j], img[i + 1, j - 1]
                    p8, p9 = img[i, j - 1], img[i - 1, j - 1]

                    neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                    b_p1 = sum(neighbors)
                    if not (2 <= b_p1 <= 6):
                        continue

                    # Number of 0 to 1 transitions in ordered sequence
                    transitions = 0
                    seq = neighbors + [p2]
                    for k in range(8):
                        if seq[k] == 0 and seq[k + 1] == 1:
                            transitions += 1
                    if transitions != 1:
                        continue

                    if p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0:
                        marked[i, j] = 1
                        changing = True
            img[marked == 1] = 0

            # Step 2
            marked = np.zeros_like(img)
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    if img[i, j] != 1:
                        continue
                    p2, p3, p4 = img[i - 1, j], img[i - 1, j + 1], img[i, j + 1]
                    p5, p6, p7 = img[i + 1, j + 1], img[i + 1, j], img[i + 1, j - 1]
                    p8, p9 = img[i, j - 1], img[i - 1, j - 1]

                    neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                    b_p1 = sum(neighbors)
                    if not (2 <= b_p1 <= 6):
                        continue

                    transitions = 0
                    seq = neighbors + [p2]
                    for k in range(8):
                        if seq[k] == 0 and seq[k + 1] == 1:
                            transitions += 1
                    if transitions != 1:
                        continue

                    if p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0:
                        marked[i, j] = 1
                        changing = True
            img[marked == 1] = 0

        return img * 255

    @staticmethod
    def euclidean_distance_transform_2d(binary_mask: np.ndarray) -> np.ndarray:
        """Compute exact 2D Euclidean Distance Transform (Meijster algorithm)."""
        h, w = binary_mask.shape
        inf = h + w
        dt = np.full((h, w), inf, dtype=np.float64)
        fg_y, fg_x = np.where(binary_mask > 0)
        if len(fg_x) == 0:
            return dt

        # Vectorized distance computation for small/medium masks
        for i in range(h):
            for j in range(w):
                if binary_mask[i, j] > 0:
                    dt[i, j] = 0.0
                else:
                    dists_sq = (fg_y - i) ** 2 + (fg_x - j) ** 2
                    dt[i, j] = math.sqrt(np.min(dists_sq))
        return dt
