from __future__ import annotations

import math
from typing import List, Optional, Tuple
import numpy as np


class PlanarHomographySolver:
    """Projective 2D planar homography estimation and decomposition."""

    @staticmethod
    def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Hartley normalization: centroid at origin, mean distance = sqrt(2)."""
        centroid = np.mean(points, axis=0)
        shifted = points - centroid
        mean_dist = np.mean(np.hypot(shifted[:, 0], shifted[:, 1]))
        scale = math.sqrt(2.0) / max(1e-8, mean_dist)

        t = np.array([
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        homog = np.column_stack([points, np.ones(len(points))])
        norm_pts = (t @ homog.T).T[:, :2]
        return norm_pts, t

    @classmethod
    def estimate_dlt(cls, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
        """Normalized Direct Linear Transformation (DLT) for 3x3 Homography H."""
        if len(src_pts) < 4 or len(dst_pts) < 4:
            raise ValueError("DLT requires at least 4 corresponding point pairs")

        norm_src, t_src = cls.normalize_points(src_pts)
        norm_dst, t_dst = cls.normalize_points(dst_pts)

        n = len(src_pts)
        a = np.zeros((2 * n, 9), dtype=np.float64)

        for i in range(n):
            xs, ys = norm_src[i]
            xd, yd = norm_dst[i]
            a[2 * i] = [-xs, -ys, -1.0, 0.0, 0.0, 0.0, xs * xd, ys * xd, xd]
            a[2 * i + 1] = [0.0, 0.0, 0.0, -xs, -ys, -1.0, xs * yd, ys * yd, yd]

        # SVD solution
        _, _, vh = np.linalg.svd(a)
        h_norm = vh[-1].reshape((3, 3))

        # Denormalize: H = T_dst^(-1) * H_norm * T_src
        h = np.linalg.inv(t_dst) @ h_norm @ t_src
        return h / h[2, 2]

    @classmethod
    def ransac_homography(
        cls,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        threshold_px: float = 3.0,
        max_iters: int = 1000,
        inlier_ratio_target: float = 0.85,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Robust RANSAC homography estimation minimizing symmetric transfer error."""
        n = len(src_pts)
        if n < 4:
            raise ValueError("RANSAC homography requires at least 4 points")

        best_h = np.eye(3)
        best_inliers = np.zeros(n, dtype=bool)
        max_inlier_count = 0

        src_h = np.column_stack([src_pts, np.ones(n)])
        dst_h = np.column_stack([dst_pts, np.ones(n)])

        for _ in range(max_iters):
            indices = np.random.choice(n, 4, replace=False)
            try:
                candidate_h = cls.estimate_dlt(src_pts[indices], dst_pts[indices])
                candidate_h_inv = np.linalg.inv(candidate_h)
            except np.linalg.LinAlgError:
                continue

            # Forward projection error: dist(H * x, x')
            proj_forward = (candidate_h @ src_h.T).T
            proj_forward /= np.maximum(1e-8, proj_forward[:, 2:3])
            err_fwd_sq = np.sum((proj_forward[:, :2] - dst_pts) ** 2, axis=-1)

            # Backward projection error: dist(H^-1 * x', x)
            proj_backward = (candidate_h_inv @ dst_h.T).T
            proj_backward /= np.maximum(1e-8, proj_backward[:, 2:3])
            err_bwd_sq = np.sum((proj_backward[:, :2] - src_pts) ** 2, axis=-1)

            # Symmetric transfer error
            sym_err = np.sqrt(err_fwd_sq + err_bwd_sq)
            inliers = sym_err < threshold_px
            inlier_count = int(np.sum(inliers))

            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                best_inliers = inliers
                best_h = candidate_h

                if max_inlier_count >= int(n * inlier_ratio_target):
                    break

        # Final refit over all inliers
        if max_inlier_count >= 4:
            best_h = cls.estimate_dlt(src_pts[best_inliers], dst_pts[best_inliers])

        return best_h, best_inliers

    @staticmethod
    def warp_perspective_bilinear(image: np.ndarray, h_matrix: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
        """Backward mapping perspective warp with bilinear interpolation."""
        out_h, out_w = out_shape
        h_inv = np.linalg.inv(h_matrix)

        # Output coordinate grid
        y, x = np.indices((out_h, out_w))
        coords = np.stack([x.flatten(), y.flatten(), np.ones(out_h * out_w)], axis=0)

        # Map to source coordinates
        src_coords = h_inv @ coords
        src_x = src_coords[0] / np.maximum(1e-8, src_coords[2])
        src_y = src_coords[1] / np.maximum(1e-8, src_coords[2])

        # Bilinear interpolation
        img_h, img_w = image.shape[:2]
        x0 = np.floor(src_x).astype(np.int32)
        x1 = x0 + 1
        y0 = np.floor(src_y).astype(np.int32)
        y1 = y0 + 1

        # Valid source bounds mask
        valid = (x0 >= 0) & (x1 < img_w) & (y0 >= 0) & (y1 < img_h)

        wa = (x1 - src_x) * (y1 - src_y)
        wb = (x1 - src_x) * (src_y - y0)
        wc = (src_x - x0) * (y1 - src_y)
        wd = (src_x - x0) * (src_y - y0)

        out = np.zeros((out_h, out_w, *image.shape[2:]), dtype=image.dtype)

        if image.ndim == 2:
            out_flat = out.reshape(-1)
            out_flat[valid] = (
                wa[valid] * image[y0[valid], x0[valid]] +
                wb[valid] * image[y1[valid], x0[valid]] +
                wc[valid] * image[y0[valid], x1[valid]] +
                wd[valid] * image[y1[valid], x1[valid]]
            ).astype(image.dtype)
        else:
            for c in range(image.shape[2]):
                ch_out = out[..., c].reshape(-1)
                ch_in = image[..., c]
                ch_out[valid] = (
                    wa[valid] * ch_in[y0[valid], x0[valid]] +
                    wb[valid] * ch_in[y1[valid], x0[valid]] +
                    wc[valid] * ch_in[y0[valid], x1[valid]] +
                    wd[valid] * ch_in[y1[valid], x1[valid]]
                ).astype(image.dtype)

        return out
