from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .filters import ImageFilterKernel, SpatialConvolutionEngine


@dataclass(slots=True)
class Keypoint2D:
    x: float
    y: float
    response: float
    size: float = 1.0
    angle: float = 0.0
    octave: int = 0


class HarrisCornerDetector:
    """Harris & Stephens corner and interest point detector."""

    def __init__(self, k: float = 0.04, threshold: float = 1e4, gaussian_sigma: float = 1.5) -> None:
        self.k = k
        self.threshold = threshold
        self.sigma = gaussian_sigma

    def compute_response(self, image: np.ndarray) -> np.ndarray:
        """Compute the Harris corner response matrix R = det(M) - k * trace(M)^2."""
        img = image.astype(np.float64)
        gx = SpatialConvolutionEngine.convolve2d(img, ImageFilterKernel.sobel_x())
        gy = SpatialConvolutionEngine.convolve2d(img, ImageFilterKernel.sobel_y())

        # Second moment matrix products
        ix2 = gx * gx
        iy2 = gy * gy
        ixiy = gx * gy

        # Smooth with Gaussian window
        g_kernel = ImageFilterKernel.gaussian_2d(7, self.sigma)
        s_ix2 = SpatialConvolutionEngine.convolve2d(ix2, g_kernel)
        s_iy2 = SpatialConvolutionEngine.convolve2d(iy2, g_kernel)
        s_ixiy = SpatialConvolutionEngine.convolve2d(ixiy, g_kernel)

        det_m = (s_ix2 * s_iy2) - (s_ixiy ** 2)
        trace_m = s_ix2 + s_iy2

        # Harris response
        r = det_m - self.k * (trace_m ** 2)
        return r

    def detect(self, image: np.ndarray) -> List[Keypoint2D]:
        """Detect local maxima corner points passing threshold."""
        r = self.compute_response(image)
        h, w = r.shape
        keypoints = []

        # 3x3 local non-maximum suppression
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                val = r[i, j]
                if val > self.threshold:
                    patch = r[i - 1:i + 2, j - 1:j + 2]
                    if val == np.max(patch):
                        keypoints.append(Keypoint2D(x=float(j), y=float(i), response=float(val)))

        return keypoints


class ShiTomasiCornerDetector:
    """Shi-Tomasi Good Features to Track (GFTT) using min eigenvalue."""

    def __init__(self, threshold: float = 50.0, sigma: float = 1.5) -> None:
        self.threshold = threshold
        self.sigma = sigma

    def detect(self, image: np.ndarray) -> List[Keypoint2D]:
        img = image.astype(np.float64)
        gx = SpatialConvolutionEngine.convolve2d(img, ImageFilterKernel.sobel_x())
        gy = SpatialConvolutionEngine.convolve2d(img, ImageFilterKernel.sobel_y())

        g_kernel = ImageFilterKernel.gaussian_2d(7, self.sigma)
        ix2 = SpatialConvolutionEngine.convolve2d(gx * gx, g_kernel)
        iy2 = SpatialConvolutionEngine.convolve2d(gy * gy, g_kernel)
        ixiy = SpatialConvolutionEngine.convolve2d(gx * gy, g_kernel)

        # Analytical lambda_min for 2x2 symmetric matrix
        # eigenvalues = 0.5 * ((a + c) +/- sqrt((a - c)^2 + 4b^2))
        trace = ix2 + iy2
        diff = ix2 - iy2
        sqrt_term = np.sqrt(np.maximum(0.0, diff ** 2 + 4.0 * (ixiy ** 2)))
        lambda_min = 0.5 * (trace - sqrt_term)

        h, w = lambda_min.shape
        keypoints = []
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                val = lambda_min[i, j]
                if val > self.threshold:
                    if val == np.max(lambda_min[i - 1:i + 2, j - 1:j + 2]):
                        keypoints.append(Keypoint2D(x=float(j), y=float(i), response=float(val)))

        return keypoints


class FastFeatureDetector:
    """Features from Accelerated Segment Test (FAST-9/16) corner detector."""

    # 16-pixel Bresenham circle offsets around center (x, y)
    CIRCLE_OFFSETS = [
        (0, -3), (1, -3), (2, -2), (3, -1),
        (3, 0), (3, 1), (2, 2), (1, 3),
        (0, 3), (-1, 3), (-2, 2), (-3, 1),
        (-3, 0), (-3, -1), (-2, -2), (-1, -3)
    ]

    def __init__(self, threshold: int = 20, n_contiguous: int = 9) -> None:
        self.threshold = threshold
        self.n_contiguous = n_contiguous

    def detect(self, image: np.ndarray) -> List[Keypoint2D]:
        """Detect fast corners."""
        h, w = image.shape
        img = image.astype(np.int32)
        keypoints = []
        t = self.threshold
        n_req = self.n_contiguous

        for y in range(3, h - 3):
            for x in range(3, w - 3):
                p = img[y, x]
                # High-speed test on cardinal points 1, 5, 9, 13
                p1 = img[y - 3, x]
                p5 = img[y, x + 3]
                p9 = img[y + 3, x]
                p13 = img[y, x - 3]

                brighter_cardinal = (p1 > p + t) + (p5 > p + t) + (p9 > p + t) + (p13 > p + t)
                darker_cardinal = (p1 < p - t) + (p5 < p - t) + (p9 < p - t) + (p13 < p - t)

                if brighter_cardinal < 3 and darker_cardinal < 3:
                    continue

                # Full 16-pixel ring test
                ring = [img[y + dy, x + dx] for dx, dy in self.CIRCLE_OFFSETS]
                # Double ring for contiguous wrap-around check
                double_ring = ring + ring

                is_corner = False
                response = 0.0

                # Check contiguous brighter
                streak = 0
                for val in double_ring:
                    if val > p + t:
                        streak += 1
                        if streak >= n_req:
                            is_corner = True
                            response = float(val - p)
                            break
                    else:
                        streak = 0

                if not is_corner:
                    streak = 0
                    for val in double_ring:
                        if val < p - t:
                            streak += 1
                            if streak >= n_req:
                                is_corner = True
                                response = float(p - val)
                                break
                        else:
                            streak = 0

                if is_corner:
                    keypoints.append(Keypoint2D(x=float(x), y=float(y), response=response))

        return keypoints


class OrbDescriptorExtractor:
    """Oriented FAST and Rotated BRIEF (ORB) feature descriptor."""

    def __init__(self, n_pairs: int = 256, seed: int = 42) -> None:
        self.n_pairs = n_pairs
        # Precompute random Gaussian sampling point pairs (x1, y1) and (x2, y2)
        rng = np.random.RandomState(seed)
        self.test_pairs = rng.normal(loc=0.0, scale=4.0, size=(n_pairs, 4)).round().astype(np.int32)
        self.test_pairs = np.clip(self.test_pairs, -15, 15)

    @staticmethod
    def compute_orientation(image: np.ndarray, x: int, y: int, radius: int = 15) -> float:
        """Compute intensity centroid angle: theta = atan2(m01, m10)."""
        h, w = image.shape
        y_min = max(0, y - radius)
        y_max = min(h, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(w, x + radius + 1)

        patch = image[y_min:y_max, x_min:x_max].astype(np.float64)
        yy, xx = np.mgrid[y_min - y:y_max - y, x_min - x:x_max - x]

        # Circular mask
        mask = (xx * xx + yy * yy) <= (radius * radius)
        m10 = np.sum(xx[mask] * patch[mask])
        m01 = np.sum(yy[mask] * patch[mask])

        return math.atan2(m01, m10)

    def extract(self, image: np.ndarray, keypoints: List[Keypoint2D]) -> np.ndarray:
        """Extract 256-bit binary descriptors for given keypoints."""
        h, w = image.shape
        descriptors = np.zeros((len(keypoints), self.n_pairs // 8), dtype=np.uint8)

        for kp_idx, kp in enumerate(keypoints):
            ix, iy = int(round(kp.x)), int(round(kp.y))
            if not (16 <= ix < w - 16 and 16 <= iy < h - 16):
                continue

            angle = self.compute_orientation(image, ix, iy)
            kp.angle = angle
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            bit_idx = 0
            byte_val = 0
            byte_pos = 0

            for p_idx in range(self.n_pairs):
                x1, y1, x2, y2 = self.test_pairs[p_idx]
                # Steer sampling points by keypoint orientation angle
                rx1 = int(round(x1 * cos_a - y1 * sin_a)) + ix
                ry1 = int(round(x1 * sin_a + y1 * cos_a)) + iy
                rx2 = int(round(x2 * cos_a - y2 * sin_a)) + ix
                ry2 = int(round(x2 * sin_a + y2 * cos_a)) + iy

                val1 = image[ry1, rx1]
                val2 = image[ry2, rx2]

                bit = 1 if val1 < val2 else 0
                byte_val |= (bit << (bit_idx % 8))
                bit_idx += 1

                if bit_idx % 8 == 0:
                    descriptors[kp_idx, byte_pos] = byte_val
                    byte_pos += 1
                    byte_val = 0

        return descriptors
