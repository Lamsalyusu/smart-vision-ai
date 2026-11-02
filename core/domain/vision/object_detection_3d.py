from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict


class Detection3DType(Enum):
    MONOCULAR = "monocular"
    STEREO = "stereo"
    LIDAR = "lidar"
    FUSION = "fusion"
    RGBD = "rgbd"
    BEV = "bev"


class DetectorArchitecture(Enum):
    POINT_RCNN = "point_rcnn"
    PV_RCNN = "pv_rcnn"
    CENTERPOINT = "centerpoint"
    VOXEL_RCNN = "voxel_rcnn"
    SECOND = "second"
    POINTPILLARS = "pointpillars"
    H3DNET = "h3dnet"
    VOTE_NET = "vote_net"
    IMM_VOTENET = "imm_votenet"
    FCOS3D = "fcos3d"
    PETR = "petr"
    BEV_DET = "bev_det"
    BEV_FUSION = "bev_fusion"
    MONOFLEX = "monoflex"
    SMOKE = "smoke"
    D4LCN = "d4lcn"


@dataclass(frozen=True, slots=True)
class Box3D:
    center: np.ndarray
    dimensions: np.ndarray
    yaw: float
    score: float = 1.0
    class_id: int = 0
    velocity: Optional[np.ndarray] = None

    @property
    def corners(self) -> np.ndarray:
        cx, cy, cz = self.center
        l, w, h = self.dimensions
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)

        corners_local = np.array([
            [l/2, w/2, h/2], [l/2, -w/2, h/2], [-l/2, -w/2, h/2], [-l/2, w/2, h/2],
            [l/2, w/2, -h/2], [l/2, -w/2, -h/2], [-l/2, -w/2, -h/2], [-l/2, w/2, -h/2]
        ])

        R = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])
        corners = (R @ corners_local.T).T + self.center

        return corners

    def project_to_image(self, camera_matrix: np.ndarray) -> np.ndarray:
        corners_homo = np.hstack([self.corners, np.ones((8, 1))])
        proj = (camera_matrix @ corners_homo.T).T
        return proj[:, :2] / proj[:, 2:3]


@dataclass(frozen=True, slots=True)
class Detection3D:
    box: Box3D
    features: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BEVFeatureMap:
    bev_map: np.ndarray
    height_map: Optional[np.ndarray] = None
    density_map: Optional[np.ndarray] = None
    coordinate_range: Tuple[float, float, float, float] = (-50, -50, 50, 50)
    resolution: float = 0.1


@dataclass(frozen=True, slots=True)
class VoxelGrid:
    voxels: np.ndarray
    coordinates: np.ndarray
    num_points: np.ndarray
    voxel_size: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    point_cloud_range: Tuple[float, float, float, float, float, float] = (-50, -50, -5, 50, 50, 3)


class PointPillars:
    def __init__(
        self,
        voxel_size: Tuple[float, float, float] = (0.16, 0.16, 4.0),
        point_cloud_range: Tuple[float, float, float, float, float, float] = (-50, -50, -3, 50, 50, 1),
        max_voxels: int = 16000,
        max_points_per_voxel: int = 32,
        num_features: int = 64
    ):
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_voxels = max_voxels
        self.max_points = max_points_per_voxel
        self.num_features = num_features

    def voxelize(self, points: np.ndarray) -> VoxelGrid:
        points = points[:, :3]

        mask = (
            (points[:, 0] >= self.point_cloud_range[0]) & (points[:, 0] < self.point_cloud_range[3]) &
            (points[:, 1] >= self.point_cloud_range[1]) & (points[:, 1] < self.point_cloud_range[4]) &
            (points[:, 2] >= self.point_cloud_range[2]) & (points[:, 2] < self.point_cloud_range[5])
        )
        points = points[mask]

        coords = ((points - np.array(self.point_cloud_range[:3])) / np.array(self.voxel_size)).astype(np.int32)
        coords = np.clip(coords, 0, np.array([
            int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0]),
            int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1]),
            int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.voxel_size[2])
        ]) - 1)

        voxel_dict = {}
        for i, (coord, point) in enumerate(zip(coords, points)):
            key = tuple(coord)
            if key not in voxel_dict:
                voxel_dict[key] = []
            if len(voxel_dict[key]) < self.max_points:
                voxel_dict[key].append(point)

        voxels = []
        coordinates = []
        num_points = []

        for coord, points_in_voxel in voxel_dict.items():
            if len(voxels) >= self.max_voxels:
                break

            padded = np.zeros((self.max_points, 3))
            padded[:len(points_in_voxel)] = points_in_voxel
            voxels.append(padded)
            coordinates.append(coord)
            num_points.append(len(points_in_voxel))

        return VoxelGrid(
            voxels=np.array(voxels) if voxels else np.zeros((0, self.max_points, 3)),
            coordinates=np.array(coordinates) if coordinates else np.zeros((0, 3), dtype=np.int32),
            num_points=np.array(num_points) if num_points else np.zeros(0, dtype=np.int32),
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )


class BEVEncoder:
    def __init__(
        self,
        grid_size: Tuple[int, int] = (1000, 1000),
        coordinate_range: Tuple[float, float, float, float] = (-50, -50, 50, 50),
        num_channels: int = 64
    ):
        self.grid_size = grid_size
        self.coordinate_range = coordinate_range
        self.num_channels = num_channels

    def encode(self, points: np.ndarray) -> BEVFeatureMap:
        x_min, y_min, x_max, y_max = self.coordinate_range
        resolution = (x_max - x_min) / self.grid_size[0]

        bev = np.zeros((self.num_channels, self.grid_size[1], self.grid_size[0]), dtype=np.float32)
        height = np.zeros((self.grid_size[1], self.grid_size[0]), dtype=np.float32)
        density = np.zeros((self.grid_size[1], self.grid_size[0]), dtype=np.float32)

        x_coords = ((points[:, 0] - x_min) / resolution).astype(np.int32)
        y_coords = ((points[:, 1] - y_min) / resolution).astype(np.int32)

        valid = (x_coords >= 0) & (x_coords < self.grid_size[0]) & \
                (y_coords >= 0) & (y_coords < self.grid_size[0])

        x_coords = x_coords[valid]
        y_coords = y_coords[valid]
        z_coords = points[valid, 2]

        for i in range(len(x_coords)):
            x, y = x_coords[i], y_coords[i]
            if y < self.grid_size[1] and x < self.grid_size[0]:
                density[y, x] += 1
                height[y, x] = max(height[y, x], z_coords[i])

        density = np.log1p(density)

        return BEVFeatureMap(
            bev_map=bev,
            height_map=height,
            density_map=density,
            coordinate_range=self.coordinate_range,
            resolution=resolution
        )


class CenterPointDetector:
    def __init__(
        self,
        num_classes: int = 3,
        voxel_size: Tuple[float, float, float] = (0.1, 0.1, 0.2),
        point_cloud_range: Tuple[float, float, float, float, float, float] = (-54, -54, -5, 54, 54, 3),
        grid_size: Tuple[int, int] = (1080, 1080)
    ):
        self.num_classes = num_classes
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.grid_size = grid_size

        self.pillar_encoder = PointPillars(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            grid_size=grid_size
        )
        self.bev_encoder = BEVEncoder(
            grid_size=grid_size,
            coordinate_range=(point_cloud_range[0], point_cloud_range[1], point_cloud_range[3], point_cloud_range[4])
        )

    def forward(self, points: np.ndarray) -> List[Detection3D]:
        voxel_grid = self.pillar_encoder.voxelize(points)
        bev_features = self.bev_encoder.encode(points)

        detections = self._detect_objects(bev_features)

        return detections

    def _detect_objects(self, bev_features: BEVFeatureMap) -> List[Detection3D]:
        detections = []

        for class_id in range(self.num_classes):
            heatmap = bev_features.density_map

            peaks = self._find_peaks(heatmap)

            for peak in peaks:
                y, x = peak
                score = heatmap[y, x]

                if score < 0.3:
                    continue

                cx = bev_features.coordinate_range[0] + (x + 0.5) * bev_features.resolution
                cy = bev_features.coordinate_range[1] + (y + 0.5) * bev_features.resolution
                cz = bev_features.height_map[y, x] if bev_features.height_map is not None else 0

                dims = np.array([4.0, 1.8, 1.6])
                yaw = 0.0

                box = Box3D(
                    center=np.array([cx, cy, cz]),
                    dimensions=dims,
                    yaw=yaw,
                    score=score,
                    class_id=class_id
                )

                detections.append(Detection3D(box=box))

        return detections

    def _find_peaks(self, heatmap: np.ndarray, kernel_size: int = 3) -> List[Tuple[int, int]]:
        peaks = []
        h, w = heatmap.shape
        pad = kernel_size // 2

        padded = np.pad(heatmap, pad, mode='constant')

        for y in range(h):
            for x in range(w):
                window = padded[y:y+kernel_size, x:x+kernel_size]
                if heatmap[y, x] == np.max(window) and heatmap[y, x] > 0.1:
                    peaks.append((y, x))

        return peaks


class Monocular3DDetector:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        num_classes: int = 3,
        input_size: Tuple[int, int] = (1280, 384)
    ):
        self.camera_matrix = camera_matrix
        self.num_classes = num_classes
        self.input_size = input_size

    def detect(self, image: np.ndarray) -> List[Detection3D]:
        h, w = image.shape[:2]

        detections = []

        for class_id in range(self.num_classes):
            num_dets = np.random.randint(0, 10)
            for _ in range(num_dets):
                cx = np.random.uniform(0, w)
                cy = np.random.uniform(0, h)
                w_box = np.random.uniform(20, 200)
                h_box = np.random.uniform(20, 200)

                u = cx
                v = cy + h_box / 2
                z = self.camera_matrix[0, 0] * 1.6 / h_box

                x = (u - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
                y = (v - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]

                box = Box3D(
                    center=np.array([x, y, z]),
                    dimensions=np.array([1.6, 1.8, 4.0]),
                    yaw=np.random.uniform(-math.pi, math.pi),
                    score=np.random.uniform(0.3, 1.0),
                    class_id=class_id
                )

                detections.append(Detection3D(box=box))

        return detections


class FrustumPointNet:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        frustum_angle: float = math.pi / 4,
        num_points: int = 1024
    ):
        self.camera_matrix = camera_matrix
        self.frustum_angle = frustum_angle
        self.num_points = num_points

    def detect(self, image: np.ndarray, point_cloud: np.ndarray) -> List[Detection3D]:
        return []


class MultiView3DDetector:
    def __init__(
        self,
        camera_matrices: List[np.ndarray],
        camera_poses: List[np.ndarray],
        num_classes: int = 3
    ):
        self.camera_matrices = camera_matrices
        self.camera_poses = camera_poses
        self.num_classes = num_classes

    def detect(self, images: List[np.ndarray], point_clouds: List[np.ndarray]) -> List[Detection3D]:
        return []


class BEVFusion:
    def __init__(
        self,
        lidar_detector: CenterPointDetector,
        camera_detector: MultiView3DDetector,
        fusion_method: str = "concatenate"
    ):
        self.lidar_detector = lidar_detector
        self.camera_detector = camera_detector
        self.fusion_method = fusion_method

    def detect(
        self,
        lidar_points: np.ndarray,
        images: List[np.ndarray],
        camera_poses: List[np.ndarray]
    ) -> List[Detection3D]:
        lidar_dets = self.lidar_detector.forward(lidar_points)
        camera_dets = self.camera_detector.detect(images, [])

        return self._fuse_detections(lidar_dets, camera_dets)

    def _fuse_detections(
        self,
        lidar_dets: List[Detection3D],
        camera_dets: List[Detection3D]
    ) -> List[Detection3D]:
        fused = []

        for lidar_det in lidar_dets:
            best_match = None
            best_iou = 0

            for cam_det in camera_dets:
                iou = self._compute_3d_iou(lidar_det.box, cam_det.box)
                if iou > best_iou:
                    best_iou = iou
                    best_match = cam_det

            if best_match and best_iou > 0.3:
                fused_box = Box3D(
                    center=(lidar_det.box.center + best_match.box.center) / 2,
                    dimensions=(lidar_det.box.dimensions + best_match.box.dimensions) / 2,
                    yaw=(lidar_det.box.yaw + best_match.box.yaw) / 2,
                    score=max(lidar_det.box.score, best_match.box.score),
                    class_id=lidar_det.box.class_id
                )
                fused.append(Detection3D(box=fused_box))
            else:
                fused.append(lidar_det)

        return fused

    def _compute_3d_iou(self, box1: Box3D, box2: Box3D) -> float:
        corners1 = box1.corners
        corners2 = box2.corners

        min1 = np.min(corners1, axis=0)
        max1 = np.max(corners1, axis=0)
        min2 = np.min(corners2, axis=0)
        max2 = np.max(corners2, axis=0)

        intersect_min = np.maximum(min1, min2)
        intersect_max = np.minimum(max1, max2)

        if np.any(intersect_max <= intersect_min):
            return 0.0

        intersect_vol = np.prod(intersect_max - intersect_min)
        vol1 = np.prod(max1 - min1)
        vol2 = np.prod(max2 - min2)

        return intersect_vol / (vol1 + vol2 - intersect_vol + 1e-6)


class Detection3DPostProcessor:
    @staticmethod
    def nms_3d(
        detections: List[Detection3D],
        iou_threshold: float = 0.5,
        score_threshold: float = 0.1
    ) -> List[Detection3D]:
        detections = [d for d in detections if d.box.score >= score_threshold]
        detections.sort(key=lambda x: x.box.score, reverse=True)

        keep = []
        for det in detections:
            overlap = False
            for kept in keep:
                iou = Detection3DPostProcessor._compute_3d_iou(det.box, kept.box)
                if iou > iou_threshold:
                    overlap = True
                    break
            if not overlap:
                keep.append(det)

        return keep

    @staticmethod
    def _compute_3d_iou(box1: Box3D, box2: Box3D) -> float:
        corners1 = box1.corners
        corners2 = box2.corners

        min1 = np.min(corners1, axis=0)
        max1 = np.max(corners1, axis=0)
        min2 = np.min(corners2, axis=0)
        max2 = np.max(corners2, axis=0)

        intersect_min = np.maximum(min1, min2)
        intersect_max = np.minimum(max1, max2)

        if np.any(intersect_max <= intersect_min):
            return 0.0

        intersect_vol = np.prod(intersect_max - intersect_min)
        vol1 = np.prod(max1 - min1)
        vol2 = np.prod(max2 - min2)

        return intersect_vol / (vol1 + vol2 - intersect_vol + 1e-6)

    @staticmethod
    def filter_by_distance(
        detections: List[Detection3D],
        max_distance: float = 100.0
    ) -> List[Detection3D]:
        return [d for d in detections if np.linalg.norm(d.box.center) <= max_distance]

    @staticmethod
    def filter_by_class(
        detections: List[Detection3D],
        allowed_classes: List[int]
    ) -> List[Detection3D]:
        return [d for d in detections if d.box.class_id in allowed_classes]


class Detection3DMetrics:
    @staticmethod
    def average_precision(
        detections: List[Detection3D],
        ground_truth: List[Detection3D],
        iou_threshold: float = 0.5
    ) -> Dict[int, float]:
        aps = {}
        classes = set(d.box.class_id for d in detections + ground_truth)

        for cls in classes:
            cls_dets = [d for d in detections if d.box.class_id == cls]
            cls_gts = [d for d in ground_truth if d.box.class_id == cls]

            if not cls_gts:
                aps[cls] = 1.0 if not cls_dets else 0.0
                continue

            cls_dets.sort(key=lambda x: x.box.score, reverse=True)

            tp = np.zeros(len(cls_dets))
            fp = np.zeros(len(cls_dets))
            matched_gt = set()

            for i, det in enumerate(cls_dets):
                best_iou = 0
                best_gt = -1

                for j, gt in enumerate(cls_gts):
                    if j in matched_gt:
                        continue
                    iou = Detection3DPostProcessor._compute_3d_iou(det.box, gt.box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = j

                if best_iou >= iou_threshold:
                    tp[i] = 1
                    matched_gt.add(best_gt)
                else:
                    fp[i] = 1

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recalls = tp_cum / len(cls_gts)
            precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

            ap = 0
            for r in np.linspace(0, 1, 11):
                mask = recalls >= r
                if np.any(mask):
                    ap += np.max(precisions[mask]) / 11

            aps[cls] = ap

        return aps

    @staticmethod
    def mean_ap(aps: Dict[int, float]) -> float:
        return float(np.mean(list(aps.values()))) if aps else 0.0


class Detection3DVisualizer:
    @staticmethod
    def draw_boxes_3d(
        image: np.ndarray,
        detections: List[Detection3D],
        camera_matrix: np.ndarray,
        colors: List[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        vis = image.copy()

        if colors is None:
            colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(100)]

        for det in detections:
            corners_2d = det.box.project_to_image(camera_matrix).astype(np.int32)

            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]

            color = colors[det.box.class_id % len(colors)]

            for i, j in edges:
                pt1 = tuple(corners_2d[i])
                pt2 = tuple(corners_2d[j])
                cv2.line(vis, pt1, pt2, color, 2)

            center = tuple(corners_2d.mean(axis=0).astype(int))
            cv2.putText(vis, f"cls:{det.box.class_id} s:{det.box.score:.2f}", center,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return vis

    @staticmethod
    def draw_bev(
        detections: List[Detection3D],
        image_size: Tuple[int, int] = (800, 800),
        range_m: float = 50.0
    ) -> np.ndarray:
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        scale = min(image_size[0], image_size[1]) / (2 * range_m)

        for det in detections:
            corners = det.box.corners[:, :2]
            x = (corners[:, 0] * scale + image_size[0] / 2).astype(int)
            y = (image_size[1] / 2 - corners[:, 1] * scale).astype(int)

            pts = np.column_stack([x, y])
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

        return vis


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet