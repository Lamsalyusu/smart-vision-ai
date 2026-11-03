from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Sequence, Callable
from enum import Enum
from abc import ABC, abstractmethod
import math


class DepthSource(Enum):
    STEREO = "stereo"
    RGBD = "rgbd"
    TOF = "tof"
    STRUCTURED_LIGHT = "structured_light"
    MONOCULAR = "monocular"
    LIDAR = "lidar"


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: Optional[np.ndarray] = None

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)

    @property
    def inv_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.matrix)

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        points_3d = points_3d.reshape(-1, 3)
        points_3d = points_3d[points_3d[:, 2] > 0]
        proj = (self.matrix @ points_3d.T).T
        proj[:, 0] /= proj[:, 2]
        proj[:, 1] /= proj[:, 2]
        return proj[:, :2]

    def unproject(self, points_2d: np.ndarray, depths: np.ndarray) -> np.ndarray:
        points_2d = points_2d.reshape(-1, 2)
        depths = depths.reshape(-1)
        x = (points_2d[:, 0] - self.cx) * depths / self.fx
        y = (points_2d[:, 1] - self.cy) * depths / self.fy
        return np.column_stack([x, y, depths])


@dataclass(frozen=True, slots=True)
class CameraExtrinsics:
    rotation: np.ndarray
    translation: np.ndarray

    @property
    def matrix(self) -> np.ndarray:
        mat = np.eye(4)
        mat[:3, :3] = self.rotation
        mat[:3, 3] = self.translation
        return mat

    @property
    def inv_matrix(self) -> np.ndarray:
        mat = np.eye(4)
        mat[:3, :3] = self.rotation.T
        mat[:3, 3] = -self.rotation.T @ self.translation
        return mat


@dataclass(frozen=True, slots=True)
class DepthFrame:
    depth: np.ndarray
    confidence: Optional[np.ndarray] = None
    intrinsics: Optional[CameraIntrinsics] = None
    extrinsics: Optional[CameraExtrinsics] = None
    timestamp: float = 0.0
    source: DepthSource = DepthSource.MONOCULAR

    def get_point_cloud(self, max_points: int = 100000) -> np.ndarray:
        if self.intrinsics is None:
            raise ValueError("Intrinsics required for point cloud generation")

        h, w = self.depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u = u.flatten()
        v = v.flatten()
        z = self.depth.flatten()

        valid = (z > 0) & (z < 100) & np.isfinite(z)
        if self.confidence is not None:
            valid = valid & (self.confidence.flatten() > 0.5)

        u, v, z = u[valid], v[valid], z[valid]

        if len(u) > max_points:
            idx = np.random.choice(len(u), max_points, replace=False)
            u, v, z = u[idx], v[idx], z[idx]

        x = (u - self.intrinsics.cx) * z / self.intrinsics.fx
        y = (v - self.intrinsics.cy) * z / self.intrinsics.fy
        return np.column_stack([x, y, z])


class BaseDepthEstimator(ABC):
    @abstractmethod
    def estimate(self, image: np.ndarray) -> DepthFrame:
        pass

    @abstractmethod
    def estimate_batch(self, images: Sequence[np.ndarray]) -> Sequence[DepthFrame]:
        pass


class MiDaSEstimator(BaseDepthEstimator):
    def __init__(
        self,
        model_type: str = "DPT_Large",
        device: str = "cpu",
    ):
        self.model_type = model_type
        self.device = device
        self._model = None
        self._transform = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            from torchvision.transforms import Compose, Resize, ToTensor, Normalize

            model_map = {
                "DPT_Large": "intel-isl/MiDaS",
                "DPT_Hybrid": "intel-isl/MiDaS",
                "MiDaS_small": "intel-isl/MiDaS",
            }

            self._model = torch.hub.load("intel-isl/MiDaS", self.model_type)
            self._model.eval()

            if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
                self._transform = Compose([
                    Resize(384),
                    ToTensor(),
                    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
            else:
                self._transform = Compose([
                    Resize(256),
                    ToTensor(),
                    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])

            self._initialized = True
            self._torch = torch

        except ImportError:
            self._initialized = False

    def estimate(self, image: np.ndarray) -> DepthFrame:
        if not self._initialized:
            self._initialize()

        if self._initialized:
            return self._estimate_torch(image)
        else:
            return self._estimate_classical(image)

    def _estimate_torch(self, image: np.ndarray) -> DepthFrame:
        torch = self._torch

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        h, w = image.shape[:2]
        input_tensor = self._transform(image).unsqueeze(0)

        if self.device != "cpu" and torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            self._model = self._model.cuda()

        with torch.no_grad():
            prediction = self._model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()
        depth = self._normalize_depth(depth)

        return DepthFrame(
            depth=depth,
            source=DepthSource.MONOCULAR,
            intrinsics=CameraIntrinsics(
                fx=w, fy=w, cx=w/2, cy=h/2, width=w, height=h
            ),
        )

    def _estimate_classical(self, image: np.ndarray) -> DepthFrame:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        depth = cv2.Laplacian(gray, cv2.CV_64F)
        depth = np.abs(depth)
        depth = self._normalize_depth(depth)

        h, w = image.shape[:2]
        return DepthFrame(
            depth=depth,
            source=DepthSource.MONOCULAR,
            intrinsics=CameraIntrinsics(fx=w, fy=w, cx=w/2, cy=h/2, width=w, height=h),
        )

    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        valid = depth > 0
        if valid.any():
            d_min, d_max = depth[valid].min(), depth[valid].max()
            if d_max > d_min:
                depth = (depth - d_min) / (d_max - d_min) * 10.0
        return depth.astype(np.float32)

    def estimate_batch(self, images: Sequence[np.ndarray]) -> Sequence[DepthFrame]:
        return [self.estimate(img) for img in images]


class ZoeDepthEstimator(BaseDepthEstimator):
    def __init__(
        self,
        model_name: str = "zoedepth-nyu-kitti",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch

            self._model = torch.hub.load(".", "zoedepth", model_name=self.model_name, source="local")
            self._model.eval()
            self._initialized = True
            self._torch = torch

        except ImportError:
            self._initialized = False

    def estimate(self, image: np.ndarray) -> DepthFrame:
        if not self._initialized:
            self._initialize()

        if self._initialized:
            return self._estimate_torch(image)
        else:
            return self._estimate_classical(image)

    def _estimate_torch(self, image: np.ndarray) -> DepthFrame:
        torch = self._torch

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        h, w = image.shape[:2]
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

        if self.device != "cpu" and torch.cuda.is_available():
            tensor = tensor.cuda()
            self._model = self._model.cuda()

        with torch.no_grad():
            depth = self._model.infer(tensor)

        depth = depth.cpu().numpy()[0, 0]
        depth = np.clip(depth, 0, 80)

        return DepthFrame(
            depth=depth.astype(np.float32),
            source=DepthSource.MONOCULAR,
            intrinsics=CameraIntrinsics(fx=w, fy=w, cx=w/2, cy=h/2, width=w, height=h),
        )

    def _estimate_classical(self, image: np.ndarray) -> DepthFrame:
        estimator = MiDaSEstimator()
        return estimator._estimate_classical(image)

    def estimate_batch(self, images: Sequence[np.ndarray]) -> Sequence[DepthFrame]:
        return [self.estimate(img) for img in images]


class StereoDepthEstimator(BaseDepthEstimator):
    def __init__(
        self,
        baseline: float = 0.12,
        focal_length: float = 718.856,
        min_disp: int = 0,
        num_disp: int = 128,
        block_size: int = 9,
        uniqueness_ratio: int = 10,
        speckle_window: int = 100,
        speckle_range: int = 32,
        disp12_max_diff: int = 1,
    ):
        self.baseline = baseline
        self.focal_length = focal_length
        self.min_disp = min_disp
        self.num_disp = num_disp
        self.block_size = block_size
        self.uniqueness_ratio = uniqueness_ratio
        self.speckle_window = speckle_window
        self.speckle_range = speckle_range
        self.disp12_max_diff = disp12_max_diff

        self._stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            uniquenessRatio=uniqueness_ratio,
            speckleWindowSize=speckle_window,
            speckleRange=speckle_range,
            disp12MaxDiff=disp12_max_diff,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
        )
        self._stereo_right = cv2.ximgproc.createRightMatcher(self._stereo)
        self._wls_filter = cv2.ximgproc.createDisparityWLSFilter(self._stereo)
        self._wls_filter.setLambda(8000)
        self._wls_filter.setSigmaColor(1.5)

    def estimate(self, image: np.ndarray) -> DepthFrame:
        raise NotImplementedError("Stereo estimator requires left and right images")

    def estimate_stereo(self, left: np.ndarray, right: np.ndarray) -> DepthFrame:
        if left.dtype != np.uint8:
            left = (left * 255).astype(np.uint8)
        if right.dtype != np.uint8:
            right = (right * 255).astype(np.uint8)

        gray_left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

        disp_left = self._stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        disp_right = self._stereo_right.compute(gray_right, gray_left).astype(np.float32) / 16.0

        filtered_disp = self._wls_filter.filter(disp_left, gray_left, None, disp_right)

        depth = np.zeros_like(filtered_disp)
        valid = filtered_disp > 0
        depth[valid] = self.focal_length * self.baseline / filtered_disp[valid]

        h, w = left.shape[:2]
        return DepthFrame(
            depth=depth.astype(np.float32),
            confidence=(filtered_disp > 0).astype(np.float32),
            source=DepthSource.STEREO,
            intrinsics=CameraIntrinsics(
                fx=self.focal_length, fy=self.focal_length, cx=w/2, cy=h/2, width=w, height=h
            ),
        )

    def estimate_batch(self, images: Sequence[np.ndarray]) -> Sequence[DepthFrame]:
        return [self.estimate(img) for img in images]


class DepthRefiner:
    def __init__(
        self,
        spatial_sigma: float = 15.0,
        color_sigma: float = 25.0,
        iterations: int = 4,
    ):
        self.spatial_sigma = spatial_sigma
        self.color_sigma = color_sigma
        self.iterations = iterations

    def refine(self, depth: np.ndarray, image: np.ndarray) -> np.ndarray:
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        refined = cv2.ximgproc.jointBilateralFilter(
            image, depth, d=0, sigmaColor=self.color_sigma, sigmaSpace=self.spatial_sigma
        )

        for _ in range(self.iterations - 1):
            refined = cv2.ximgproc.jointBilateralFilter(
                image, refined, d=0, sigmaColor=self.color_sigma, sigmaSpace=self.spatial_sigma
            )

        return refined


class DepthFusion:
    def __init__(
        self,
        method: str = "weighted_average",
        confidence_threshold: float = 0.5,
    ):
        self.method = method
        self.confidence_threshold = confidence_threshold

    def fuse(self, depth_frames: Sequence[DepthFrame], reference_intrinsics: CameraIntrinsics) -> DepthFrame:
        if not depth_frames:
            raise ValueError("No depth frames to fuse")

        if self.method == "weighted_average":
            return self._fuse_weighted(depth_frames, reference_intrinsics)
        elif self.method == "median":
            return self._fuse_median(depth_frames, reference_intrinsics)
        elif self.method == "max_confidence":
            return self._fuse_max_confidence(depth_frames, reference_intrinsics)
        else:
            return self._fuse_weighted(depth_frames, reference_intrinsics)

    def _fuse_weighted(self, depth_frames: Sequence[DepthFrame], reference_intrinsics: CameraIntrinsics) -> DepthFrame:
        h, w = reference_intrinsics.height, reference_intrinsics.width
        fused_depth = np.zeros((h, w), dtype=np.float32)
        fused_confidence = np.zeros((h, w), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)

        for frame in depth_frames:
            if frame.intrinsics is None:
                continue

            depth_aligned = self._align_depth(frame.depth, frame.intrinsics, reference_intrinsics)
            conf = frame.confidence if frame.confidence is not None else np.ones_like(depth_aligned)
            conf = (conf > self.confidence_threshold).astype(np.float32)

            weight_sum += conf
            fused_depth += depth_aligned * conf
            fused_confidence += conf

        valid = weight_sum > 0
        fused_depth[valid] /= weight_sum[valid]
        fused_confidence[valid] /= weight_sum[valid]

        return DepthFrame(
            depth=fused_depth,
            confidence=fused_confidence,
            intrinsics=reference_intrinsics,
            source=DepthSource.MONOCULAR,
        )

    def _fuse_median(self, depth_frames: Sequence[DepthFrame], reference_intrinsics: CameraIntrinsics) -> DepthFrame:
        h, w = reference_intrinsics.height, reference_intrinsics.width
        depths = []

        for frame in depth_frames:
            if frame.intrinsics is None:
                continue
            depth_aligned = self._align_depth(frame.depth, frame.intrinsics, reference_intrinsics)
            depths.append(depth_aligned)

        if not depths:
            return DepthFrame(depth=np.zeros((h, w)), intrinsics=reference_intrinsics)

        stacked = np.stack(depths, axis=-1)
        fused_depth = np.median(stacked, axis=-1).astype(np.float32)
        fused_confidence = (np.sum(stacked > 0, axis=-1) / len(depths)).astype(np.float32)

        return DepthFrame(
            depth=fused_depth,
            confidence=fused_confidence,
            intrinsics=reference_intrinsics,
            source=DepthSource.MONOCULAR,
        )

    def _fuse_max_confidence(self, depth_frames: Sequence[DepthFrame], reference_intrinsics: CameraIntrinsics) -> DepthFrame:
        h, w = reference_intrinsics.height, reference_intrinsics.width
        fused_depth = np.zeros((h, w), dtype=np.float32)
        fused_confidence = np.zeros((h, w), dtype=np.float32)

        for frame in depth_frames:
            if frame.intrinsics is None:
                continue

            depth_aligned = self._align_depth(frame.depth, frame.intrinsics, reference_intrinsics)
            conf = frame.confidence if frame.confidence is not None else np.ones_like(depth_aligned)

            update = conf > fused_confidence
            fused_depth[update] = depth_aligned[update]
            fused_confidence[update] = conf[update]

        return DepthFrame(
            depth=fused_depth,
            confidence=fused_confidence,
            intrinsics=reference_intrinsics,
            source=DepthSource.MONOCULAR,
        )

    def _align_depth(self, depth: np.ndarray, src_intrinsics: CameraIntrinsics, dst_intrinsics: CameraIntrinsics) -> np.ndarray:
        h, w = depth.shape
        if (src_intrinsics.width == dst_intrinsics.width and
            src_intrinsics.height == dst_intrinsics.height and
            np.allclose(src_intrinsics.matrix, dst_intrinsics.matrix)):
            return depth

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        points_3d = src_intrinsics.unproject(np.stack([u, v], axis=-1), depth)
        projected = dst_intrinsics.project(points_3d)

        x = projected[:, 0].reshape(h, w)
        y = projected[:, 1].reshape(h, w)

        map_x = x.astype(np.float32)
        map_y = y.astype(np.float32)

        aligned = cv2.remap(depth, map_x, map_y, cv2.INTER_LINEAR)
        return aligned


class NormalEstimator:
    def __init__(
        self,
        method: str = "pca",
        neighborhood_size: int = 5,
    ):
        self.method = method
        self.neighborhood_size = neighborhood_size

    def estimate(self, depth_frame: DepthFrame) -> np.ndarray:
        if depth_frame.intrinsics is None:
            raise ValueError("Intrinsics required for normal estimation")

        points = depth_frame.get_point_cloud(max_points=depth_frame.depth.size)
        h, w = depth_frame.depth.shape

        normal_map = np.zeros((h, w, 3), dtype=np.float32)

        if self.method == "pca":
            normal_map = self._estimate_pca_normals(points, depth_frame.depth.shape)
        elif self.method == "cross_product":
            normal_map = self._estimate_cross_normals(depth_frame)
        elif self.method == "gradient":
            normal_map = self._estimate_gradient_normals(depth_frame)

        return normal_map

    def _estimate_pca_normals(self, points: np.ndarray, shape: tuple) -> np.ndarray:
        h, w = shape
        normal_map = np.zeros((h, w, 3), dtype=np.float32)
        k = self.neighborhood_size

        for v in range(k, h - k):
            for u in range(k, w - k):
                patch = points[v-k:v+k+1, u-k:u+k+1]
                valid = patch[:, :, 2] > 0
                if valid.sum() < 9:
                    continue

                valid_points = patch[valid]
                centroid = np.mean(valid_points, axis=0)
                centered = valid_points - centroid
                cov = centered.T @ centered / len(valid_points)
                eigvals, eigvecs = np.linalg.eigh(cov)
                normal = eigvecs[:, 0]
                if normal[2] < 0:
                    normal = -normal
                normal_map[v, u] = normal

        return normal_map

    def _estimate_cross_normals(self, depth_frame: DepthFrame) -> np.ndarray:
        h, w = depth_frame.depth.shape
        normal_map = np.zeros((h, w, 3), dtype=np.float32)
        intrinsics = depth_frame.intrinsics

        for v in range(1, h - 1):
            for u in range(1, w - 1):
                z_c = depth_frame.depth[v, u]
                if z_c <= 0:
                    continue

                z_u = depth_frame.depth[v, u + 1]
                z_v = depth_frame.depth[v + 1, u]
                if z_u <= 0 or z_v <= 0:
                    continue

                fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy

                x_c = (u - cx) * z_c / fx
                y_c = (v - cy) * z_c / fy
                x_u = (u + 1 - cx) * z_u / fx
                y_v = (v + 1 - cy) * z_v / fy

                du = np.array([x_u - x_c, 0, z_u - z_c])
                dv = np.array([0, y_v - y_c, z_v - z_c])

                normal = np.cross(du, dv)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm
                    if normal[2] < 0:
                        normal = -normal
                    normal_map[v, u] = normal

        return normal_map

    def _estimate_gradient_normals(self, depth_frame: DepthFrame) -> np.ndarray:
        depth = depth_frame.depth
        dzdx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        dzdy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)

        normal_map = np.zeros((*depth.shape, 3), dtype=np.float32)
        normal_map[:, :, 0] = -dzdx
        normal_map[:, :, 1] = -dzdy
        normal_map[:, :, 2] = 1.0

        norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
        norm[norm == 0] = 1.0
        normal_map /= norm

        return normal_map


class PlaneDetector:
    def __init__(
        self,
        distance_threshold: float = 0.01,
        min_points: int = 100,
        max_iterations: int = 1000,
        normal_threshold: float = 0.1,
    ):
        self.distance_threshold = distance_threshold
        self.min_points = min_points
        self.max_iterations = max_iterations
        self.normal_threshold = normal_threshold

    def detect(self, depth_frame: DepthFrame) -> list[dict]:
        points = depth_frame.get_point_cloud()
        if len(points) < self.min_points:
            return []

        planes = []
        remaining = points.copy()

        for _ in range(10):
            if len(remaining) < self.min_points:
                break

            plane, inliers = self._ransac_plane(remaining)
            if plane is None or len(inliers) < self.min_points:
                break

            normal = plane[:3]
            d = plane[3]

            plane_info = {
                'normal': normal,
                'distance': d,
                'inliers': inliers,
                'num_points': len(inliers),
                'center': np.mean(inliers, axis=0),
            }
            planes.append(plane_info)

            mask = np.ones(len(remaining), dtype=bool)
            mask[inliers] = False
            remaining = remaining[mask]

        return planes

    def _ransac_plane(self, points: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        best_plane = None
        best_inliers = None
        best_count = 0

        for _ in range(self.max_iterations):
            idx = np.random.choice(len(points), 3, replace=False)
            sample = points[idx]

            v1 = sample[1] - sample[0]
            v2 = sample[2] - sample[0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm

            d = -np.dot(normal, sample[0])

            distances = np.abs(points @ normal + d)
            inliers = np.where(distances < self.distance_threshold)[0]

            if len(inliers) > best_count:
                best_count = len(inliers)
                best_plane = np.array([normal[0], normal[1], normal[2], d])
                best_inliers = inliers

        if best_inliers is not None and len(best_inliers) >= 3:
            inlier_points = points[best_inliers]
            centroid = np.mean(inlier_points, axis=0)
            centered = inlier_points - centroid
            cov = centered.T @ centered / len(centered)
            eigvals, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, 0]
            if normal[2] < 0:
                normal = -normal
            d = -np.dot(normal, centroid)
            best_plane = np.array([normal[0], normal[1], normal[2], d])
            distances = np.abs(points @ normal + d)
            best_inliers = np.where(distances < self.distance_threshold)[0]

        return best_plane, best_inliers


class DepthCompletion:
    def __init__(
        self,
        method: str = "inpaint",
        max_depth: float = 80.0,
    ):
        self.method = method
        self.max_depth = max_depth

    def complete(self, depth: np.ndarray, image: Optional[np.ndarray] = None) -> np.ndarray:
        mask = (depth <= 0) | (depth > self.max_depth) | ~np.isfinite(depth)
        if not mask.any():
            return depth

        if self.method == "inpaint":
            return self._inpaint_depth(depth, mask, image)
        elif self.method == "interpolate":
            return self._interpolate_depth(depth, mask)
        elif self.method == "diffusion":
            return self._diffusion_depth(depth, mask)
        else:
            return self._inpaint_depth(depth, mask, image)

    def _inpaint_depth(self, depth: np.ndarray, mask: np.ndarray, image: Optional[np.ndarray]) -> np.ndarray:
        depth_norm = np.zeros_like(depth)
        valid = depth > 0
        if valid.any():
            d_min, d_max = depth[valid].min(), depth[valid].max()
            if d_max > d_min:
                depth_norm[valid] = (depth[valid] - d_min) / (d_max - d_min) * 255

        depth_uint8 = depth_norm.astype(np.uint8)
        mask_uint8 = mask.astype(np.uint8) * 255

        inpainted = cv2.inpaint(depth_uint8, mask_uint8, 3, cv2.INPAINT_TELEA)

        if valid.any():
            inpainted = inpainted.astype(np.float32) / 255.0 * (d_max - d_min) + d_min
        return inpainted

    def _interpolate_depth(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        from scipy.interpolate import griddata

        h, w = depth.shape
        y, x = np.mgrid[0:h, 0:w]

        valid = ~mask
        points = np.column_stack([x[valid], y[valid]])
        values = depth[valid]

        if len(points) < 4:
            return depth

        completed = griddata(points, values, (x, y), method='linear', fill_value=0)
        return completed.astype(np.float32)

    def _diffusion_depth(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        completed = depth.copy()
        for _ in range(50):
            laplacian = cv2.Laplacian(completed, cv2.CV_32F, ksize=3)
            completed[mask] += 0.25 * laplacian[mask]
            completed = np.maximum(completed, 0)
        return completed
