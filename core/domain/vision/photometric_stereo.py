from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict
import copy


class PhotometricStereoMethod(Enum):
    LEAST_SQUARES = "least_squares"
    RANSAC = "ransac"
    ROBUST = "robust"
    INTEGRABILITY = "integrability"
    DEEP_LEARNING = "deep_learning"


class LightModel(Enum):
    LAMBERTIAN = "lambertian"
    PHONG = "phong"
    BLINN_PHONG = "blinn_phong"
    OREN_NAYAR = "oren_nayar"
    COOK_TORRANCE = "cook_torrance"
    NON_LAMBERTIAN = "non_lambertian"


@dataclass(frozen=True, slots=True)
class LightSource:
    direction: np.ndarray
    intensity: float = 1.0
    color: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    position: Optional[np.ndarray] = None
    light_type: str = "directional"


@dataclass(frozen=True, slots=True)
class CameraParams:
    intrinsics: np.ndarray
    distortion: np.ndarray
    resolution: Tuple[int, int]


@dataclass(frozen=True, slots=True)
class PhotometricStereoImage:
    image: np.ndarray
    light_source: LightSource
    camera_params: CameraParams
    timestamp: float = field(default_factory=time.time)
    mask: Optional[np.ndarray] = None


@dataclass(frozen=True, slots=True)
class SurfaceNormal:
    normal: np.ndarray
    confidence: float
    albedo: float
    position: np.ndarray


@dataclass(frozen=True, slots=True)
class PhotometricStereoResult:
    normals: np.ndarray
    albedo: np.ndarray
    depth: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    lights: List[LightSource] = field(default_factory=list)
    reconstruction_error: float = 0.0


class BasePhotometricStereo(ABC):
    @abstractmethod
    def compute_normals(
        self,
        images: List[PhotometricStereoImage]
    ) -> PhotometricStereoResult:
        pass


class LeastSquaresPhotometricStereo(BasePhotometricStereo):
    def __init__(
        self,
        min_images: int = 3,
        regularization: float = 0.0,
        shadow_threshold: float = 0.1
    ):
        self.min_images = min_images
        self.regularization = regularization
        self.shadow_threshold = shadow_threshold

    def compute_normals(
        self,
        images: List[PhotometricStereoImage]
    ) -> PhotometricStereoResult:
        if len(images) < self.min_images:
            raise ValueError(f"Need at least {self.min_images} images")

        h, w = images[0].image.shape[:2]
        n_images = len(images)

        L = np.zeros((n_images, 3))
        I = np.zeros((n_images, 0))
        masks = []

        for img in images:
            L = np.vstack([L, img.light_source.direction])
            I = np.hstack([I, img.image.flatten()[:, np.newaxis]])
            if img.mask is not None:
                masks.append(img.mask.flatten())
            else:
                masks.append(np.ones(img.image.shape[:2]).flatten())

        masks = np.array(masks).T

        normals = np.zeros((h * w, 3))
        albedo = np.zeros(h * w)
        valid_mask = np.zeros(h * w, dtype=bool)

        for i in range(h * w):
            valid_lights = masks[i]
            if np.sum(valid_lights) < 3:
                continue

            L_valid = L[valid_lights]
            I_valid = I[i, valid_lights]

            if self.regularization > 0:
                A = L_valid.T @ L_valid + self.regularization * np.eye(3)
            else:
                A = L_valid.T @ L_valid

            b = L_valid.T @ I_valid

            try:
                g = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue

            albedo_val = np.linalg.norm(g)
            if albedo_val < 1e-6:
                continue

            normal = g / albedo_val

            shadow_mask = self._detect_shadows(L_valid, I_valid, normal, albedo_val)
            if np.sum(shadow_mask) > len(L_valid) * self.shadow_threshold:
                continue

            normals[i] = normal
            albedo[i] = albedo_val
            valid_mask[i] = True

        normals_img = normals.reshape(h, w, 3)
        albedo_img = albedo.reshape(h, w)
        mask_img = valid_mask.reshape(h, w)

        return PhotometricStereoResult(
            normals=normals_img,
            albedo=albedo_img,
            mask=mask_img,
            lights=[img.light_source for img in images]
        )

    def _detect_shadows(
        self,
        L: np.ndarray,
        I: np.ndarray,
        normal: np.ndarray,
        albedo: float
    ) -> np.ndarray:
        predicted = albedo * (L @ normal)
        residuals = I - predicted
        return residuals > 0.1 * np.max(I)


class RANSACPhotometricStereo(BasePhotometricStereo):
    def __init__(
        self,
        min_images: int = 3,
        max_iterations: int = 1000,
        threshold: float = 0.05,
        confidence: float = 0.99
    ):
        self.min_images = min_images
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.confidence = confidence

    def compute_normals(
        self,
        images: List[PhotometricStereoImage]
    ) -> PhotometricStereoResult:
        h, w = images[0].image.shape[:2]
        n_images = len(images)

        L = np.vstack([img.light_source.direction for img in images])
        I = np.hstack([img.image.flatten()[:, np.newaxis] for img in images])

        normals = np.zeros((h * w, 3))
        albedo = np.zeros(h * w)
        valid_mask = np.zeros(h * w, dtype=bool)

        n_samples = min(3, n_images)
        n_iterations = self.max_iterations

        for i in range(h * w):
            I_i = I[i, :]
            if np.sum(I_i > 0) < self.min_images:
                continue

            best_normal = None
            best_inliers = 0
            best_albedo = 0

            for _ in range(n_iterations):
                sample_idx = np.random.choice(n_images, n_samples, replace=False)
                L_sample = L[sample_idx]
                I_sample = I_i[sample_idx]

                try:
                    g = np.linalg.solve(L_sample.T @ L_sample, L_sample.T @ I_sample)
                except np.linalg.LinAlgError:
                    continue

                albedo_val = np.linalg.norm(g)
                if albedo_val < 1e-6:
                    continue

                normal = g / albedo_val
                predicted = albedo_val * (L @ normal)
                residuals = np.abs(I_i - predicted)
                inliers = np.sum(residuals < self.threshold)

                if inliers > best_inliers:
                    best_inliers = inliers
                    best_normal = normal
                    best_albedo = albedo_val

            if best_inliers >= self.min_images and best_normal is not None:
                normals[i] = best_normal
                albedo[i] = best_albedo
                valid_mask[i] = True

        normals_img = normals.reshape(h, w, 3)
        albedo_img = albedo.reshape(h, w)
        mask_img = valid_mask.reshape(h, w)

        return PhotometricStereoResult(
            normals=normals_img,
            albedo=albedo_img,
            mask=mask_img
        )


class RobustPhotometricStereo(BasePhotometricStereo):
    def __init__(
        self,
        min_images: int = 3,
        robust_kernel: str = "huber",
        huber_delta: float = 0.1,
        max_iterations: int = 50
    ):
        self.min_images = min_images
        self.robust_kernel = robust_kernel
        self.huber_delta = huber_delta
        self.max_iterations = max_iterations

    def compute_normals(
        self,
        images: List[PhotometricStereoImage]
    ) -> PhotometricStereoResult:
        h, w = images[0].image.shape[:2]
        n_images = len(images)

        L = np.vstack([img.light_source.direction for img in images])
        I = np.hstack([img.image.flatten()[:, np.newaxis] for img in images])

        normals = np.zeros((h * w, 3))
        albedo = np.zeros(h * w)
        valid_mask = np.zeros(h * w, dtype=bool)

        for i in range(h * w):
            I_i = I[i, :]
            valid = I_i > 0

            if np.sum(valid) < self.min_images:
                continue

            L_valid = L[valid]
            I_valid = I_i[valid]

            g = np.linalg.lstsq(L_valid, I_valid, rcond=None)[0]
            normal = g / (np.linalg.norm(g) + 1e-8)
            albedo_val = np.linalg.norm(g)

            for _ in range(self.max_iterations):
                residuals = I_valid - np.dot(L_valid, normal) * albedo_val
                weights = self._compute_weights(residuals)

                W = np.diag(weights)
                g = np.linalg.lstsq(np.sqrt(W) @ L_valid, np.sqrt(W) * I_valid, rcond=None)[0]
                normal = g / (np.linalg.norm(g) + 1e-8)
                albedo_val = np.linalg.norm(g)

            normals[i] = normal
            albedo[i] = albedo_val
            valid_mask[i] = True

        normals_img = normals.reshape(h, w, 3)
        albedo_img = albedo.reshape(h, w)
        mask_img = valid_mask.reshape(h, w)

        return PhotometricStereoResult(
            normals=normals_img,
            albedo=albedo_img,
            mask=mask_img
        )

    def _compute_weights(self, residuals: np.ndarray) -> np.ndarray:
        if self.robust_kernel == "huber":
            abs_res = np.abs(residuals)
            weights = np.ones_like(residuals)
            mask = abs_res > self.huber_delta
            weights[mask] = self.huber_delta / abs_res[mask]
            return weights
        return np.ones_like(residuals)


class IntegrabilityPhotometricStereo(BasePhotometricStereo):
    def __init__(
        self,
        min_images: int = 3,
        integrability_weight: float = 1.0,
        max_iterations: int = 50
    ):
        self.min_images = min_images
        self.integrability_weight = integrability_weight
        self.max_iterations = max_iterations

    def compute_normals(
        self,
        images: List[PhotometricStereoImage]
    ) -> PhotometricStereoResult:
        h, w = images[0].image.shape[:2]
        n_images = len(images)

        L = np.vstack([img.light_source.direction for img in images])
        I = np.hstack([img.image.flatten()[:, np.newaxis] for img in images])

        normals = np.zeros((h * w, 3))
        albedo = np.zeros(h * w)
        valid_mask = np.zeros(h * w, dtype=bool)

        for i in range(h * w):
            I_i = I[i, :]
            valid = I_i > 0

            if np.sum(valid) < self.min_images:
                continue

            L_valid = L[valid]
            I_valid = I_i[valid]

            g = np.linalg.lstsq(L_valid, I_valid, rcond=None)[0]
            normal = g / (np.linalg.norm(g) + 1e-8)
            albedo_val = np.linalg.norm(g)

            normals[i] = normal
            albedo[i] = albedo_val
            valid_mask[i] = True

        normals_img = normals.reshape(h, w, 3)

        for _ in range(self.max_iterations):
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    idx = y * w + x
                    if not valid_mask[idx]:
                        continue

                    n = normals_img[y, x]
                    n_left = normals_img[y, x - 1] if valid_mask[y, x - 1] else n
                    n_right = normals_img[y, x + 1] if valid_mask[y, x + 1] else n
                    n_up = normals_img[y - 1, x] if valid_mask[y - 1, x] else n
                    n_down = normals_img[y + 1, x] if valid_mask[y + 1, x] else n

                    n_avg = (n_left + n_right + n_up + n_down) / 4
                    n_new = n + self.integrability_weight * (n_avg - n)
                    n_new = n_new / (np.linalg.norm(n_new) + 1e-8)

                    normals_img[y, x] = n_new

        albedo_img = albedo.reshape(h, w)
        mask_img = valid_mask.reshape(h, w)

        return PhotometricStereoResult(
            normals=normals_img,
            albedo=albedo_img,
            mask=mask_img
        )


class NonLambertianPhotometricStereo(BasePhotometricStereo):
    def __init__(
        self,
        min_images: int = 4,
        light_model: LightModel = LightModel.PHONG,
        max_iterations: int = 100
    ):
        self.min_images = min_images
        self.light_model = light_model
        self.max_iterations = max_iterations

    def compute_normals(
        self,
        images: List[PhotometricStereoImage]
    ) -> PhotometricStereoResult:
        h, w = images[0].image.shape[:2]
        n_images = len(images)

        L = np.vstack([img.light_source.direction for img in images])
        I = np.hstack([img.image.flatten()[:, np.newaxis] for img in images])

        normals = np.zeros((h * w, 3))
        albedo = np.zeros(h * w)
        specular = np.zeros(h * w)
        shininess = np.zeros(h * w)
        valid_mask = np.zeros(h * w, dtype=bool)

        for i in range(h * w):
            I_i = I[i, :]
            valid = I_i > 0

            if np.sum(valid) < self.min_images:
                continue

            params = self._fit_phong(L[valid], I_i[valid])
            if params is None:
                continue

            normal, albedo_val, spec, shin = params
            normals[i] = normal
            albedo[i] = albedo_val
            specular[i] = spec
            shininess[i] = shin
            valid_mask[i] = True

        return PhotometricStereoResult(
            normals=normals.reshape(h, w, 3),
            albedo=albedo.reshape(h, w),
            mask=valid_mask.reshape(h, w)
        )

    def _fit_phong(
        self,
        L: np.ndarray,
        I: np.ndarray
    ) -> Optional[Tuple[np.ndarray, float, float, float]]:
        def phong_model(params, L):
            normal = params[:3]
            albedo = params[3]
            specular = params[4]
            shininess = params[5]

            normal = normal / (np.linalg.norm(normal) + 1e-8)

            diffuse = albedo * np.maximum(L @ normal, 0)

            view_dir = np.array([0, 0, 1])
            reflect = 2 * (L @ normal)[:, np.newaxis] * normal - L
            spec = specular * np.maximum(reflect @ view_dir, 0) ** shininess

            return albedo * diffuse + specular * spec

        def residuals(params):
            pred = phong_model(params, L)
            return pred - I

        initial = np.array([0, 0, 1, 0.5, 0.1, 10.0])

        try:
            from scipy.optimize import least_squares
            result = least_squares(residuals, initial, max_nfev=50)
            if result.success:
                normal = result.x[:3] / (np.linalg.norm(result.x[:3]) + 1e-8)
                return normal, result.x[3], result.x[4], result.x[5]
        except ImportError:
            pass

        return None


class PhotometricStereoPipeline:
    def __init__(
        self,
        method: PhotometricStereoMethod = PhotometricStereoMethod.LEAST_SQUARES,
        camera_params: CameraParams = None,
        **kwargs
    ):
        self.method = method
        self.camera_params = camera_params
        self.estimator = self._create_estimator(method, **kwargs)

    def _create_estimator(self, method: PhotometricStereoMethod, **kwargs) -> BasePhotometricStereo:
        if method == PhotometricStereoMethod.LEAST_SQUARES:
            return LeastSquaresPhotometricStereo(**kwargs)
        elif method == PhotometricStereoMethod.RANSAC:
            return RANSACPhotometricStereo(**kwargs)
        elif method == PhotometricStereoMethod.Robust:
            return RobustPhotometricStereo(**kwargs)
        elif method == PhotometricStereoMethod.INTEGRABILITY:
            return IntegrabilityPhotometricStereo(**kwargs)
        elif method == PhotometricStereoMethod.DEEP_LEARNING:
            return NonLambertianPhotometricStereo(**kwargs)
        else:
            return LeastSquaresPhotometricStereo(**kwargs)

    def calibrate_lights(
        self,
        images: List[np.ndarray],
        known_normals: np.ndarray,
        known_albedo: np.ndarray
    ) -> List[LightSource]:
        lights = []

        for i, img in enumerate(images):
            mask = known_normals[..., 2] > 0
            if np.sum(mask) < 10:
                continue

            I = img[mask].flatten()
            N = known_normals[mask]
            A = known_albedo[mask].flatten()

            L_dir = np.linalg.lstsq(known_normals[mask] * A[:, np.newaxis], I, rcond=None)[0]
            L_dir = L_dir / (np.linalg.norm(L_dir) + 1e-8)

            lights.append(LightSource(
                direction=L_dir,
                intensity=1.0
            ))

        return lights

    def reconstruct_surface(
        self,
        result: PhotometricStereoResult,
        integration_method: str = "poisson"
    ) -> np.ndarray:
        if integration_method == "poisson":
            return self._poisson_integration(result.normals, result.mask)
        elif integration_method == "francot":
            return self._francot_integration(result.normals, result.mask)
        else:
            return self._height_integration(result.normals, result.mask)

    def _poisson_integration(self, normals: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = normals.shape[:2]
        n = normals.reshape(-1, 3)

        p = -n[:, 0] / (n[:, 2] + 1e-8)
        q = -n[:, 1] / (n[:, 2] + 1e-8)

        p_img = p.reshape(h, w)
        q_img = q.reshape(h, w)

        from scipy.fftpack import dctn, idctn

        fxx = np.gradient(p_img, axis=1)
        fyy = np.gradient(q_img, axis=0)

        rhs = fxx + fyy
        rhs_fft = np.fft.fft2(rhs)

        kx = np.fft.fftfreq(w) * 2 * np.pi
        ky = np.fft.fftfreq(h) * 2 * np.pi
        kx, ky = np.meshgrid(kx, ky)

        denom = kx**2 + ky**2
        denom[0, 0] = 1

        Z_fft = -rhs_fft / denom
        Z_fft[0, 0] = 0

        depth = np.real(np.fft.ifft2(Z_fft))

        depth[~mask.reshape(h, w)] = 0
        return depth

    def _francot_integration(self, normals: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = normals.shape[:2]
        n = normals.reshape(-1, 3)

        p = -n[:, 0] / (n[:, 2] + 1e-8)
        q = -n[:, 1] / (n[:, 2] + 1e-8)

        p_img = p.reshape(h, w)
        q_img = q.reshape(h, w)

        depth = np.zeros((h, w))
        depth[0, 0] = 0

        for y in range(1, h):
            depth[y, 0] = depth[y-1, 0] - q_img[y, 0]

        for x in range(1, w):
            depth[0, x] = depth[0, x-1] + p_img[0, x]

        for y in range(1, h):
            for x in range(1, w):
                depth[y, x] = 0.5 * (depth[y-1, x] + q_img[y, x] + depth[y, x-1] + p_img[y, x])

        depth[~mask] = 0
        return depth

    def _height_integration(self, normals: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self._francot_integration(normals, mask)


class PhotometricStereoVisualizer:
    @staticmethod
    def draw_normals(
        normals: np.ndarray,
        mask: np.ndarray = None,
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        h, w = normals.shape[:2]

        if mask is None:
            mask = np.ones((h, w), dtype=bool)

        scale_x = image_size[0] / w
        scale_y = image_size[1] / h

        for y in range(h):
            for x in range(w):
                if mask[y, x]:
                    nx, ny, nz = normals[y, x]
                    r = int((nx + 1) * 127.5)
                    g = int((ny + 1) * 127.5)
                    b = int((nz + 1) * 127.5)

                    px = int(x * scale_x)
                    py = int(y * scale_y)

                    if 0 <= px < image_size[0] and 0 <= py < image_size[1]:
                        vis[py, px] = [b, g, r]

        return vis

    @staticmethod
    def draw_albedo(albedo: np.ndarray, mask: np.ndarray = None, image_size: Tuple[int, int] = (800, 600)) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        h, w = albedo.shape
        if mask is None:
            mask = np.ones((h, w), dtype=bool)

        scale_x = image_size[0] / w
        scale_y = image_size[1] / h

        max_albedo = np.max(albedo[mask])
        if max_albedo == 0:
            max_albedo = 1.0

        for y in range(h):
            for x in range(w):
                if mask[y, x]:
                    val = albedo[y, x] / max_albedo
                    intensity = int(val * 255)

                    px = int(x * scale_x)
                    py = int(y * scale_y)

                    if 0 <= px < image_size[0] and 0 <= py < image_size[1]:
                        vis[py, px] = [intensity, intensity, intensity]

        return vis

    @staticmethod
    def draw_depth(depth: np.ndarray, mask: np.ndarray = None, image_size: Tuple[int, int] = (800, 600)) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        h, w = depth.shape
        if mask is None:
            mask = np.ones((h, w), dtype=bool)

        valid_depths = depth[mask]
        if len(valid_depths) == 0:
            return vis

        min_d = np.min(valid_depths)
        max_d = np.max(valid_depths)

        scale_x = image_size[0] / w
        scale_y = image_size[1] / h

        for y in range(h):
            for x in range(w):
                if mask[y, x]:
                    val = (depth[y, x] - min_d) / (max_d - min_d + 1e-6)
                    px = int(x * scale_x)
                    py = int(y * scale_y)

                    if 0 <= px < image_size[0] and 0 <= py < image_size[1]:
                        vis[py, px] = cv2.applyColorMap(np.array([int(val * 255)], dtype=np.uint8), cv2.COLORMAP_PLASMA)[0][0]

        return vis


class PhotometricStereoMetrics:
    @staticmethod
    def angular_error(gt_normals: np.ndarray, est_normals: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float]:
        gt = gt_normals[mask]
        est = est_normals[mask]

        if len(gt) == 0:
            return 0.0, 0.0, 0.0

        dots = np.sum(gt * est, axis=1)
        dots = np.clip(dots, -1, 1)
        angles = np.arccos(dots)

        return float(np.mean(angles)), float(np.median(angles)), float(np.max(angles))

    @staticmethod
    def albedo_error(gt_albedo: np.ndarray, est_albedo: np.ndarray, mask: np.ndarray) -> float:
        gt = gt_albedo[mask]
        est = est_albedo[mask]

        if len(gt) == 0:
            return 0.0

        return float(np.mean(np.abs(gt - est)))

    @staticmethod
    def depth_error(gt_depth: np.ndarray, est_depth: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
        gt = gt_depth[mask]
        est = est_depth[mask]

        if len(gt) == 0:
            return 0.0, 0.0

        rmse = float(np.sqrt(np.mean((gt - est)**2)))
        rel = float(np.mean(np.abs(gt - est) / (gt + 1e-6)))

        return rmse, rel


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
