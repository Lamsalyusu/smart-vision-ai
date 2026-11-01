from __future__ import annotations

import math
from typing import Tuple
import numpy as np


class ColorSpaceConverter:
    """Colorimetric transformations between photometric color spaces."""

    @staticmethod
    def rgb_to_grayscale(rgb: np.ndarray, standard: str = "rec709") -> np.ndarray:
        """Converts RGB image to grayscale luminance."""
        if standard == "rec709":
            # ITU-R BT.709 (sRGB HDTV)
            weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
        elif standard == "rec601":
            # ITU-R BT.601 (SDTV / NTSC)
            weights = np.array([0.2990, 0.5870, 0.1140], dtype=np.float64)
        else:
            weights = np.array([1/3, 1/3, 1/3], dtype=np.float64)
        return np.dot(rgb[..., :3].astype(np.float64), weights)

    @staticmethod
    def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
        """Converts RGB [0, 255] to HSV [H: 0-360, S: 0-1, V: 0-1]."""
        arr = rgb[..., :3].astype(np.float64) / 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        cmax = np.max(arr, axis=-1)
        cmin = np.min(arr, axis=-1)
        delta = cmax - cmin

        h = np.zeros_like(cmax)
        # Red is max
        mask_r = (cmax == r) & (delta != 0)
        h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0)
        # Green is max
        mask_g = (cmax == g) & (delta != 0)
        h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0)
        # Blue is max
        mask_b = (cmax == b) & (delta != 0)
        h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0)

        s = np.zeros_like(cmax)
        nonzero_cmax = cmax != 0
        s[nonzero_cmax] = delta[nonzero_cmax] / cmax[nonzero_cmax]
        v = cmax

        return np.stack([h, s, v], axis=-1)

    @staticmethod
    def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
        """Converts HSV to RGB [0, 255]."""
        h = hsv[..., 0] % 360.0
        s = np.clip(hsv[..., 1], 0.0, 1.0)
        v = np.clip(hsv[..., 2], 0.0, 1.0)

        c = v * s
        x = c * (1.0 - np.abs((h / 60.0) % 2.0 - 1.0))
        m = v - c

        r_prime = np.zeros_like(h)
        g_prime = np.zeros_like(h)
        b_prime = np.zeros_like(h)

        cond0 = (0.0 <= h) & (h < 60.0)
        cond1 = (60.0 <= h) & (h < 120.0)
        cond2 = (120.0 <= h) & (h < 180.0)
        cond3 = (180.0 <= h) & (h < 240.0)
        cond4 = (240.0 <= h) & (h < 300.0)
        cond5 = (300.0 <= h) & (h < 360.0)

        r_prime[cond0], g_prime[cond0], b_prime[cond0] = c[cond0], x[cond0], 0
        r_prime[cond1], g_prime[cond1], b_prime[cond1] = x[cond1], c[cond1], 0
        r_prime[cond2], g_prime[cond2], b_prime[cond2] = 0, c[cond2], x[cond2]
        r_prime[cond3], g_prime[cond3], b_prime[cond3] = 0, x[cond3], c[cond3]
        r_prime[cond4], g_prime[cond4], b_prime[cond4] = x[cond4], 0, c[cond4]
        r_prime[cond5], g_prime[cond5], b_prime[cond5] = c[cond5], 0, x[cond5]

        rgb = np.stack([(r_prime + m) * 255.0, (g_prime + m) * 255.0, (b_prime + m) * 255.0], axis=-1)
        return np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def rgb_to_cie_xyz(rgb: np.ndarray) -> np.ndarray:
        """Converts sRGB to CIE 1931 XYZ tristimulus values with D65 illuminant."""
        srgb = rgb[..., :3].astype(np.float64) / 255.0
        # Gamma linearization
        linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
        m = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], dtype=np.float64)
        return np.einsum("...j,ij->...i", linear, m)

    @staticmethod
    def cie_xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
        """Converts CIE XYZ to CIE L*a*b* (CIELAB) with D65 reference white [Xn, Yn, Zn]."""
        d65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
        scaled = xyz / d65
        delta = 6.0 / 29.0

        def f(t):
            return np.where(t > (delta ** 3), t ** (1.0 / 3.0), t / (3.0 * delta ** 2) + 4.0 / 29.0)

        fx = f(scaled[..., 0])
        fy = f(scaled[..., 1])
        fz = f(scaled[..., 2])

        l_star = 116.0 * fy - 16.0
        a_star = 500.0 * (fx - fy)
        b_star = 200.0 * (fy - fz)

        return np.stack([l_star, a_star, b_star], axis=-1)

    @staticmethod
    def rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
        """Converts RGB to YUV (PAL system)."""
        arr = rgb[..., :3].astype(np.float64)
        y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        u = -0.14713 * arr[..., 0] - 0.28886 * arr[..., 1] + 0.436 * arr[..., 2]
        v = 0.615 * arr[..., 0] - 0.51499 * arr[..., 1] - 0.10001 * arr[..., 2]
        return np.stack([y, u, v], axis=-1)

    @staticmethod
    def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
        """Converts RGB to YCbCr (ITU-R BT.601 for JPEG / video codecs)."""
        arr = rgb[..., :3].astype(np.float64)
        y = 16.0 + 65.481 * arr[..., 0] / 255.0 + 128.553 * arr[..., 1] / 255.0 + 24.966 * arr[..., 2] / 255.0
        cb = 128.0 - 37.797 * arr[..., 0] / 255.0 - 74.203 * arr[..., 1] / 255.0 + 112.0 * arr[..., 2] / 255.0
        cr = 128.0 + 112.0 * arr[..., 0] / 255.0 - 93.786 * arr[..., 1] / 255.0 - 18.214 * arr[..., 2] / 255.0
        return np.stack([y, cb, cr], axis=-1)


class AutoWhiteBalance:
    """Chromatic adaptation and white balance algorithms."""

    @staticmethod
    def gray_world(image: np.ndarray) -> np.ndarray:
        """Gray World assumption: average color in world is achromatic gray."""
        arr = image.astype(np.float64)
        mean_r = np.mean(arr[..., 0])
        mean_g = np.mean(arr[..., 1])
        mean_b = np.mean(arr[..., 2])
        mean_gray = (mean_r + mean_g + mean_b) / 3.0

        scale_r = mean_gray / max(1e-4, mean_r)
        scale_g = mean_gray / max(1e-4, mean_g)
        scale_b = mean_gray / max(1e-4, mean_b)

        balanced = np.stack([
            arr[..., 0] * scale_r,
            arr[..., 1] * scale_g,
            arr[..., 2] * scale_b
        ], axis=-1)
        return np.clip(balanced, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def retinex_white_patch(image: np.ndarray, percentile: float = 99.0) -> np.ndarray:
        """Retinex White Patch algorithm: normalizes top specular reflectance percentile to white."""
        arr = image.astype(np.float64)
        max_r = np.percentile(arr[..., 0], percentile)
        max_g = np.percentile(arr[..., 1], percentile)
        max_b = np.percentile(arr[..., 2], percentile)

        scale_r = 255.0 / max(1.0, max_r)
        scale_g = 255.0 / max(1.0, max_g)
        scale_b = 255.0 / max(1.0, max_b)

        balanced = np.stack([
            arr[..., 0] * scale_r,
            arr[..., 1] * scale_g,
            arr[..., 2] * scale_b
        ], axis=-1)
        return np.clip(balanced, 0.0, 255.0).astype(np.uint8)
