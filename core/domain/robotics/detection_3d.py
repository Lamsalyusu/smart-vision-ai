from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math
import time


@dataclass(frozen=True, slots=True)
class Detection3D:
    class_id: int
    class_name: str
    confidence: float
    bbox_2d: Tuple[float, float, float, float]
    bbox_3d: Tuple[float, float, float, float, float, float]
    center_3d: np.ndarray
    dimensions: np.ndarray
    orientation: np.ndarray
    depth: float
    keypoints_2d: Optional[np.ndarray] = None
    keypoints_3d: Optional[np.ndarray] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Point3D:
    x: float
    y: float
    z: float
    intensity: float = 1.0
    rgb: Optional[Tuple[int, int, int]] = None


@dataclass(frozen=True, slots=True)
class BoundingBox3D:
    center: np.ndarray
    dimensions: np.ndarray
    orientation: np.ndarray
    class_id: int
    class_name: str
    confidence: float
    corners_3d: Optional[np.ndarray] = None

    @property
    def volume(self) -> float:
        return np.prod(self.dimensions)

    def get_corners(self) -> np.ndarray:
        if self.corners_3d is not None:
            return self.corners_3d

        dx, dy, dz = self.dimensions / 2
        corners_local = np.array([
            [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
            [-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz], [-dx, dy, dz],
        ])

        R = self._rotation_matrix(self.orientation)
        corners_world = (R @ corners_local.T).T + self.center
        return corners_world

    def _rotation_matrix(self, orientation: np.ndarray) -> np.ndarray:
        if len(orientation) == 4:
            x, y, z, w = orientation
            return np.array([
                [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y],
            ])
        else:
            return orientation


class Base3DDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray, point_cloud: np.ndarray = None) -> List[Detection3D]:
        pass

    @abstractmethod
    def detect_batch(self, images: List[np.ndarray], point_clouds: List[np.ndarray] = None) -> List[List[Detection3D]]:
        pass


class Monocular3DDetector:
    def __init__(
        self,
        model_path: str = "mono3d.pth",
        camera_matrix: np.ndarray = None,
        confidence_threshold: float = 0.3,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.camera_matrix = camera_matrix if camera_matrix is not None else np.eye(3)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            self._initialized = True
        except ImportError:
            self._initialized = False

    def detect(self, image: np.ndarray, point_cloud: np.ndarray = None) -> List[Detection3D]:
        if not self._initialized:
            self._initialize()

        if not self._initialized:
            return self._detect_classical(image)

        return self._detect_deep(image)

    def _detect_classical(self, image: np.ndarray) -> List[Detection3D]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detections.append(Detection3D(
                class_id=0,
                class_name="object",
                confidence=0.5,
                bbox_2d=(x, y, x + w, y + h),
                bbox_3d=(0, 0, 0, 1, 1, 1),
                center_3d=np.array([0.0, 0.0, 1.0]),
                dimensions=np.array([1.0, 1.0, 1.0]),
                orientation=np.eye(3).flatten(),
                depth=1.0,
            ))

        return detections

    def _detect_deep(self, image: np.ndarray) -> List[Detection3D]:
        return []

    def detect_batch(self, images: List[np.ndarray], point_clouds: List[np.ndarray] = None) -> List[List[Detection3D]]:
        return [self.detect(img) for img in images]


class Stereo3DDetector:
    def __init__(
        self,
        left_camera_matrix: np.ndarray,
        right_camera_matrix: np.ndarray,
        baseline: float,
        max_disparity: int = 128,
        block_size: int = 9,
    ):
        self.left_camera_matrix = left_camera_matrix
        self.right_camera_matrix = right_camera_matrix
        self.baseline = baseline
        self.max_disparity = max_disparity
        self.block_size = block_size

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=max_disparity,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
        )

    def detect(self, left_image: np.ndarray, right_image: np.ndarray) -> List[Detection3D]:
        if left_image.dtype != np.uint8:
            left_image = (left_image * 255).astype(np.uint8)
        if right_image.dtype != np.uint8:
            right_image = (right_image * 255).astype(np.uint8)

        gray_left = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY) if len(left_image.shape) == 3 else left_image
        gray_right = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY) if len(right_image.shape) == 3 else right_image

        disparity = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

        depth = np.zeros_like(disparity)
        valid = disparity > 0
        depth[valid] = self.left_camera_matrix[0, 0] * self.baseline / disparity[valid]

        detections = []
        h, w = disparity.shape

        grid_size = 32
        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                patch = disparity[y:y+grid_size, x:x+grid_size]
                if patch.size == 0:
                    continue
                mean_disp = np.mean(patch)
                if mean_disp > 0:
                    depth_val = self.left_camera_matrix[0, 0] * self.baseline / mean_disp
                    if depth_val < 50:
                        cx, cy = self.left_camera_matrix[0, 2], self.left_camera_matrix[1, 2]
                        fx, fy = self.left_camera_matrix[0, 0], self.left_camera_matrix[1, 1]
                        X = (x - self.left_camera_matrix[0, 2]) * depth_val / self.left_camera_matrix[0, 0]
                        Y = (y - self.left_camera_matrix[1, 2]) * depth_val / self.left_camera_matrix[1, 1]
                        Z = depth_val

                        detections.append(Detection3D(
                            class_id=0,
                            class_name="object",
                            confidence=0.7,
                            bbox_2d=(x*2, y*2, (x+1)*2, (y+1)*2),
                            bbox_3d=(X, Y, Z, 1, 1, 1),
                            center_3d=np.array([X, Y, Z]),
                            dimensions=np.array([1.0, 1.0, 1.0]),
                            orientation=np.eye(3).flatten(),
                            depth=depth_val,
                        ))

        return detections


class LiDARDetector:
    def __init__(
        self,
        voxel_size: float = 0.1,
        max_range: float = 100.0,
        min_points: int = 10,
    ):
        self.voxel_size = voxel_size
        self.max_range = max_range
        self.min_points = min_points

    def detect(self, point_cloud: np.ndarray) -> List[Detection3D]:
        if point_cloud.shape[1] < 3:
            return []

        points = point_cloud[:, :3]
        points = points[np.linalg.norm(points, axis=1) < self.max_range]

        downsampled = self._voxel_downsample(points)
        clusters = self._euclidean_cluster(downsampled)

        detections = []
        for cluster in clusters:
            if len(cluster) < self.min_points:
                continue

            centroid = np.mean(cluster, axis=0)
            cov = np.cov(cluster.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            dimensions = 2 * np.sqrt(np.maximum(eigvals, 0))
            orientation = eigvecs

            detections.append(Detection3D(
                class_id=0,
                class_name="object",
                confidence=0.8,
                bbox_2d=(0, 0, 0, 0),
                bbox_3d=(centroid[0], centroid[1], centroid[2], dimensions[0], dimensions[1], dimensions[2]),
                center_3d=centroid,
                dimensions=dimensions,
                orientation=eigvecs.flatten(),
                depth=np.linalg.norm(centroid),
            ))

        return detections

    def _voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        coords = np.floor(points / self.voxel_size).astype(np.int32)
        _, indices = np.unique(coords, axis=0, return_index=True)
        return points[indices]

    def _euclidean_cluster(self, points: np.ndarray, tolerance: float = 0.5, min_size: int = 10) -> List[np.ndarray]:
        if len(points) < 2:
            return [points] if len(points) > 0 else []

        clusters = []
        visited = np.zeros(len(points), dtype=bool)

        for i in range(len(points)):
            if visited[i]:
                continue

            cluster = []
            stack = [i]
            visited[i] = True

            while stack:
                idx = stack.pop()
                cluster.append(idx)

                for j in range(len(points)):
                    if not visited[j] and np.linalg.norm(points[j] - points[idx]) < 0.5:
                        visited[j] = True
                        stack.append(j)

            if len(cluster) >= self.min_points:
                clusters.append(points[cluster])

        return clusters


class PointPillarsDetector:
    def __init__(
        self,
        voxel_size: Tuple[float, float, float] = (0.16, 0.16, 4),
        point_cloud_range: Tuple[float, float, float, float, float, float] = (0, -40, -3, 70.4, 40, 1),
        max_voxels: int = 20000,
        num_classes: int = 3,
    ):
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_voxels = max_voxels
        self.num_classes = num_classes

    def detect(self, point_cloud: np.ndarray) -> List[Detection3D]:
        voxels, coords, num_points = self._points_to_voxels(point_cloud)
        features = self._extract_features(voxels, num_points)

        detections = self._decode(features)

        return detections

    def _points_to_voxels(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.array([]), np.array([]), np.array([])

    def _extract_features(self, voxels: np.ndarray, num_points: np.ndarray) -> np.ndarray:
        return np.array([])

    def _decode(self, features: np.ndarray) -> List[Detection3D]:
        return []


class CenterPointDetector:
    def __init__(
        self,
        model_path: str = "centerpoint.pth",
        voxel_size: float = 0.075,
        point_cloud_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.device = device
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            self._initialized = True
        except ImportError:
            self._initialized = False

    def detect(self, point_cloud: np.ndarray) -> List[Detection3D]:
        if not self._initialized:
            self._initialize()

        if self._initialized:
            return self._detect_deep(point_cloud)

        return self._detect_classical(point_cloud)

    def _detect_classical(self, point_cloud: np.ndarray) -> List[Detection3D]:
        detector = LiDARDetector()
        return detector.detect(point_cloud)

    def _detect_deep(self, point_cloud: np.ndarray) -> List[Detection3D]:
        return []


class VoteNetDetector:
    def __init__(
        self,
        model_path: str = "votenet.pth",
        num_points: int = 20000,
        num_classes: int = 10,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.num_points = num_points
        self.num_classes = num_classes
        self.device = device
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            self._initialized = True
        except ImportError:
            self._initialized = False

    def detect(self, point_cloud: np.ndarray) -> List[Detection3D]:
        if not self._initialized:
            self._initialize()

        if not self._initialized:
            return self._detect_classical(point_cloud)

        return self._detect_deep(point_cloud)

    def _detect_classical(self, point_cloud: np.ndarray) -> List[Detection3D]:
        detector = LiDARDetector()
        return detector.detect(point_cloud)

    def _detect_deep(self, point_cloud: np.ndarray) -> List[Detection3D]:
        return []


class PVRCNNDetector:
    def __init__(
        self,
        model_path: str = "pvrcnn.pth",
        voxel_size: float = 0.05,
        point_cloud_range: List[float] = [0, -40, -3, 70.4, 40, 1],
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.device = device
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            self._initialized = True
        except ImportError:
            self._initialized = False

    def detect(self, point_cloud: np.ndarray) -> List[Detection3D]:
        if not self._initialized:
            self._initialize()

        if not self._initialized:
            return self._detect_classical(point_cloud)

        return self._detect_deep(point_cloud)

    def _detect_classical(self, point_cloud: np.ndarray) -> List[Detection3D]:
        detector = LiDARDetector()
        return detector.detect(point_cloud)

    def _detect_deep(self, point_cloud: np.ndarray) -> List[Detection3D]:
        return []


class FrustumPointNetDetector:
    def __init__(
        self,
        model_path: str = "frustum_pointnet.pth",
        num_classes: int = 10,
        num_points: int = 1024,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.num_classes = num_classes
        self.num_points = num_points
        self.device = device
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            self._initialized = True
        except ImportError:
            self._initialized = False

    def detect(self, image: np.ndarray, point_cloud: np.ndarray, frustum: Tuple) -> List[Detection3D]:
        if not self._initialized:
            self._initialize()

        if not self._initialized:
            return self._detect_classical(image, point_cloud)

        return self._detect_deep(image, point_cloud, frustum)

    def _detect_classical(self, image: np.ndarray, point_cloud: np.ndarray) -> List[Detection3D]:
        detector_2d = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        detector_3d = LiDARDetector()
        return []

    def _detect_deep(self, image: np.ndarray, point_cloud: np.ndarray, frustum: Tuple) -> List[Detection3D]:
        return []


class Detection3DEvaluator:
    @staticmethod
    def compute_iou_3d(box1: np.ndarray, box2: np.ndarray) -> float:
        return 0.0

    @staticmethod
    def compute_ap_3d(predictions: List, ground_truth: List, iou_threshold: float = 0.5) -> Dict:
        return {}

    @staticmethod
    def evaluate_detections(predictions: List[Detection3D], ground_truth: List[Detection3D]) -> Dict[str, float]:
        return {
            'ap_3d': 0.0,
            'ap_bev': 0.0,
            'ap_2d': 0.0,
        }


class Detection3DVisualizer:
    @staticmethod
    def draw_3d_boxes(image: np.ndarray, detections: List[Detection3D], camera_matrix: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        vis = image.copy()
        for det in detections:
            if det.confidence < 0.5:
                continue
            corners_3d = det.get_corners()
            corners_2d = (camera_matrix @ corners_3d.T).T
            corners_2d = corners_2d[:, :2] / corners_2d[:, 2:3]
            corners_2d = corners_2d.astype(np.int32)

            for i in range(4):
                cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[(i+1)%4]), (0, 255, 0), 2)
                cv2.line(image, tuple(corners_2d[i+4]), tuple(corners_2d[(i+1)%4+4]), (0, 255, 0), 2)
                cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[i+4]), (0, 255, 0), 2)

        return image

    @staticmethod
    def draw_bird_eye_view(detections: List[Detection3D], range_m: float = 50.0, resolution: float = 0.1) -> np.ndarray:
        size = int(range_m / 0.1)
        bev = np.zeros((int(range_m/0.1)*2, int(range_m/0.1)*2, 3), dtype=np.uint8)

        for det in detections:
            if det.confidence < 0.5:
                continue
            center = det.center_3d
            x = int((det.center_3d[0] + range_m) / 0.1)
            y = int((det.center_3d[1] + range_m) / 0.1)
            cv2.circle(bev, (x, y), 5, (0, 255, 0), -1)

        return bev
