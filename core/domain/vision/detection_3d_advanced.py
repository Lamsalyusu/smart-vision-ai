from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict, deque


class ObjectType(Enum):
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    TRAFFIC_SIGN = "traffic_sign"
    TRAFFIC_LIGHT = "traffic_light"
    BARRIER = "barrier"
    ANIMAL = "animal"
    DEBRIS = "debris"
    UNKNOWN = "unknown"


class DetectionSource(Enum):
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"
    FUSION = "fusion"


@dataclass(frozen=True, slots=True)
class BoundingBox3D:
    center: np.ndarray
    dimensions: np.ndarray
    yaw: float
    score: float = 1.0

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

        R = np.array([
            [cos_y, -sin_y, 0],
            [sin_y, cos_y, 0],
            [0, 0, 1]
        ])

        corners = (R @ corners_local.T).T + self.center
        return corners


@dataclass(frozen=True, slots=True)
class Detection3D:
    bbox: BoundingBox3D
    class_id: int
    class_name: str
    confidence: float
    source: DetectionSource
    velocity: Optional[np.ndarray] = None
    track_id: Optional[int] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Track3D:
    track_id: int
    bbox: BoundingBox3D
    class_id: int
    class_name: str
    velocity: np.ndarray
    age: int
    hits: int
    score: float
    last_update: float
    state: str = "active"


class BaseDetector3D(ABC):
    @abstractmethod
    def detect(self, data: Any) -> List[Detection3D]:
        pass


class LiDARDetector3D(BaseDetector3D):
    def __init__(
        self,
        voxel_size: float = 0.1,
        detection_range: float = 100.0,
        min_points_per_cluster: int = 10,
        cluster_tolerance: float = 0.5
    ):
        self.voxel_size = voxel_size
        self.detection_range = detection_range
        self.min_points = min_points_per_cluster
        self.cluster_tolerance = cluster_tolerance

    def detect(self, point_cloud: np.ndarray) -> List[Detection3D]:
        if len(point_cloud) == 0:
            return []

        filtered = self._range_filter(point_cloud)
        downsampled = self._voxel_downsample(filtered)
        clusters = self._euclidean_clustering(downsampled)

        detections = []
        for cluster in clusters:
            if len(cluster) < self.min_points:
                continue

            bbox = self._fit_bounding_box(cluster)
            detections.append(Detection3D(
                bbox=bbox,
                class_id=0,
                class_name="object",
                confidence=0.8,
                source=DetectionSource.LIDAR
            ))

        return detections

    def _range_filter(self, points: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(points[:, :3], axis=1)
        return points[dist <= self.detection_range]

    def _voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points

        voxel_indices = np.floor(points[:, :3] / self.voxel_size).astype(int)
        unique_indices, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
        downsampled = np.zeros((len(unique_indices), points.shape[1]))

        for i, idx in enumerate(unique_indices):
            mask = np.all(voxel_indices == idx, axis=1)
            downsampled[i] = np.mean(points[mask], axis=0)

        return downsampled

    def _euclidean_clustering(self, points: np.ndarray) -> List[np.ndarray]:
        if len(points) == 0:
            return []

        from scipy.spatial import KDTree
        tree = KDTree(points[:, :3])

        visited = np.zeros(len(points), dtype=bool)
        clusters = []

        for i in range(len(points)):
            if visited[i]:
                continue

            cluster = []
            queue = [i]
            visited[i] = True

            while queue:
                idx = queue.pop(0)
                cluster.append(points[idx])

                neighbors = tree.query_ball_point(points[idx], self.cluster_tolerance)
                for n_idx in neighbors:
                    if not visited[n_idx]:
                        visited[n_idx] = True
                        queue.append(n_idx)

            if len(cluster) >= 3:
                clusters.append(np.array(cluster))

        return clusters

    def _fit_bounding_box(self, cluster: np.ndarray) -> BoundingBox3D:
        points = cluster[:, :3]
        centroid = np.mean(points, axis=0)

        centered = points - centroid
        cov = centered.T @ centered
        _, _, Vt = np.linalg.svd(cov)
        rotation = Vt[:3, :3].T

        yaw = math.atan2(rotation[1, 0], rotation[0, 0])

        rotated = points @ rotation
        min_coords = np.min(rotated, axis=0)
        max_coords = np.max(rotated, axis=0)
        dimensions = max_coords - min_coords

        center = centroid + rotation @ ((min_coords + max_coords) / 2)

        return BoundingBox3D(
            center=center,
            dimensions=dimensions,
            yaw=yaw
        )


class CameraDetector3D(BaseDetector3D):
    def __init__(
        self,
        camera_matrix: np.ndarray,
        image_width: int,
        image_height: int,
        confidence_threshold: float = 0.5
    ):
        self.camera_matrix = camera_matrix
        self.image_width = image_width
        self.image_height = image_height
        self.confidence_threshold = confidence_threshold

    def detect(self, image: np.ndarray) -> List[Detection3D]:
        return []

    def project_2d_to_3d(
        self,
        bbox_2d: List[float],
        depth: float
    ) -> np.ndarray:
        u = (bbox_2d[0] + bbox_2d[2]) / 2
        v = (bbox_2d[1] + bbox_2d[3]) / 2

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return np.array([x, y, z])


class FusionDetector3D:
    def __init__(
        self,
        lidar_detector: LiDARDetector3D,
        camera_detector: CameraDetector3D,
        fusion_method: str = "early"
    ):
        self.lidar_detector = lidar_detector
        self.camera_detector = camera_detector
        self.fusion_method = fusion_method

    def detect(
        self,
        point_cloud: np.ndarray,
        image: np.ndarray,
        camera_matrix: np.ndarray
    ) -> List[Detection3D]:
        lidar_detections = self.lidar_detector.detect(point_cloud)
        camera_detections = self.camera_detector.detect(image)

        if self.fusion_method == "early":
            return self._early_fusion(lidar_detections, camera_detections)
        elif self.fusion_method == "late":
            return self._late_fusion(lidar_detections, camera_detections)
        else:
            return lidar_detections + camera_detections

    def _early_fusion(
        self,
        lidar_dets: List[Detection3D],
        camera_dets: List[Detection3D]
    ) -> List[Detection3D]:
        fused = []

        for lidar_det in lidar_dets:
            best_match = None
            best_iou = 0

            for cam_det in camera_dets:
                iou = self._compute_3d_iou(lidar_det.bbox, cam_det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = cam_det

            if best_match and best_iou > 0.3:
                fused_bbox = self._fuse_bboxes(lidar_det.bbox, best_match.bbox)
                fused_det = Detection3D(
                    bbox=fused_bbox,
                    class_id=lidar_det.class_id,
                    class_name=lidar_det.class_name,
                    confidence=max(lidar_det.confidence, best_match.confidence),
                    source=DetectionSource.FUSION,
                    velocity=lidar_det.velocity,
                    attributes={**lidar_det.attributes, **best_match.attributes}
                )
                fused.append(fused_det)
            else:
                fused.append(lidar_det)

        return fused

    def _late_fusion(
        self,
        lidar_dets: List[Detection3D],
        camera_dets: List[Detection3D]
    ) -> List[Detection3D]:
        all_dets = lidar_dets + camera_dets
        return self._nms_3d(all_dets)

    def _fuse_bboxes(self, bbox1: BoundingBox3D, bbox2: BoundingBox3D) -> BoundingBox3D:
        center = (bbox1.center + bbox2.center) / 2
        dimensions = (bbox1.dimensions + bbox2.dimensions) / 2
        yaw = (bbox1.yaw + bbox2.yaw) / 2
        return BoundingBox3D(center=center, dimensions=dimensions, yaw=yaw)

    def _compute_3d_iou(self, bbox1: BoundingBox3D, bbox2: BoundingBox3D) -> float:
        corners1 = bbox1.corners
        corners2 = bbox2.corners

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

    def _nms_3d(self, detections: List[Detection3D], iou_threshold: float = 0.5) -> List[Detection3D]:
        detections.sort(key=lambda d: d.confidence, reverse=True)

        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            detections = [
                d for d in detections
                if self._compute_3d_iou(best.bbox, d.bbox) < 0.5
            ]

        return keep


class Tracker3D:
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_distance: float = 5.0
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        self.tracks: Dict[int, Track3D] = {}
        self.track_id_counter = 0

    def update(self, detections: List[Detection3D]) -> List[Track3D]:
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id].age += 1
            self.tracks[track_id].time_since_update += 1
            if self.tracks[track_id].age > self.max_age:
                del self.tracks[track_id]

        if not detections:
            return list(self.tracks.values())

        matched = self._associate_detections_to_tracks(detections)
        unmatched_detections = [i for i in range(len(detections)) if i not in [m[1] for m in matched]]
        unmatched_tracks = [tid for tid in self.tracks.keys() if tid not in [m[0] for m in matched]]

        for track_id, det_idx in matched:
            track = self.tracks[track_id]
            det = detections[det_idx]
            track.hits += 1
            track.time_since_update = 0
            track.bbox = self._update_bbox(track.bbox, det.bbox)
            track.velocity = self._update_velocity(track.velocity, det.velocity)
            track.score = (track.score * track.hits + det.confidence) / (track.hits + 1)

        for det_idx in unmatched_detections:
            self._create_new_track(detections[det_idx])

        return list(self.tracks.values())

    def _associate_detections_to_tracks(self, detections: List[Detection3D]) -> List[Tuple[int, int]]:
        if not self.tracks:
            return []

        track_ids = list(self.tracks.keys())
        track_states = np.array([self.tracks[tid].bbox.center for tid in track_ids])
        det_positions = np.array([det.bbox.center for det in detections])

        cost_matrix = np.zeros((len(track_ids), len(detections)))
        for i, t_state in enumerate(track_states):
            for j, d_pos in enumerate(det_positions):
                cost_matrix[i, j] = np.linalg.norm(t_state - d_pos)

        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = []
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.max_distance:
                matched.append((track_ids[i], j))

        return matched

    def _update_bbox(self, old_bbox: BoundingBox3D, new_bbox: BoundingBox3D) -> BoundingBox3D:
        alpha = 0.3
        center = old_bbox.center * (1 - alpha) + new_bbox.center * alpha
        dimensions = old_bbox.dimensions * (1 - alpha) + new_bbox.dimensions * alpha
        yaw = old_bbox.yaw * (1 - alpha) + new_bbox.yaw * alpha
        return BoundingBox3D(center=center, dimensions=dimensions, yaw=yaw)

    def _update_velocity(self, old_vel: np.ndarray, new_vel: np.ndarray) -> np.ndarray:
        if new_vel is None:
            return old_vel
        if old_vel is None:
            return new_vel
        return old_vel * 0.7 + new_vel * 0.3

    def _create_new_track(self, detection: Detection3D):
        track_id = self.track_id_counter
        self.track_id_counter += 1

        state = np.zeros(6)
        state[:3] = detection.bbox.center
        if detection.velocity is not None:
            state[3:6] = detection.velocity

        track = Track3D(
            track_id=track_id,
            bbox=detection.bbox,
            class_id=detection.class_id,
            class_name=detection.class_name,
            velocity=detection.velocity if detection.velocity is not None else np.zeros(3),
            age=0,
            hits=1,
            score=detection.confidence,
            last_update=time.time()
        )
        self.tracks[track_id] = track


class MultiObjectTracker:
    def __init__(
        self,
        tracker: Tracker3D,
        motion_model: str = "constant_velocity"
    ):
        self.tracker = tracker
        self.motion_model = motion_model

    def update(self, detections: List[Detection3D]) -> List[Track3D]:
        return self.tracker.update(detections)

    def get_active_tracks(self) -> List[Track3D]:
        return [t for t in self.tracker.tracks.values() if t.time_since_update == 0]


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
            corners_3d = det.bbox.corners
            corners_2d = (camera_matrix @ corners_3d.T).T
            corners_2d = corners_2d[:, :2] / corners_2d[:, 2:3]

            corners_2d = corners_2d.astype(np.int32)
            color = colors[det.class_id % len(colors)]

            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]

            for i, j in edges:
                pt1 = tuple(corners_2d[i])
                pt2 = tuple(corners_2d[j])
                cv2.line(vis, pt1, pt2, color, 2)

            center = tuple(np.mean(corners_2d[:4], axis=0).astype(int))
            cv2.putText(vis, f"{det.class_name}: {det.confidence:.2f}", center,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[det.class_id % 10], 1)

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
            corners = det.bbox.corners[:, :2]
            x = (corners[:, 0] * scale + image_size[0] / 2).astype(int)
            y = (image_size[1] / 2 - corners[:, 1] * scale).astype(int)

            pts = np.column_stack([x, y])
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

        return vis

    @staticmethod
    def draw_tracks(
        image: np.ndarray,
        tracks: List[Track3D],
        camera_matrix: np.ndarray,
        colors: List[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        vis = image.copy()

        if colors is None:
            colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(100)]

        for track in tracks:
            corners_3d = track.bbox.corners
            corners_2d = (camera_matrix @ corners_3d.T).T
            corners_2d = corners_2d[:, :2] / corners_2d[:, 2:3]
            corners_2d = corners_2d.astype(np.int32)

            color = colors[track.track_id % len(colors)]

            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]

            for i, j in edges:
                pt1 = tuple(corners_2d[i])
                pt2 = tuple(corners_2d[j])
                cv2.line(vis, pt1, pt2, color, 2)

            center = tuple(np.mean(corners_2d[:4], axis=0).astype(int))
            cv2.putText(vis, f"ID:{track.track_id}", center,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return vis


class Detection3DMetrics:
    @staticmethod
    def average_precision(
        detections: List[Detection3D],
        ground_truth: List[Detection3D],
        iou_threshold: float = 0.5
    ) -> Dict[int, float]:
        aps = {}
        classes = set(d.class_id for d in detections + ground_truth)

        for cls in classes:
            cls_dets = [d for d in detections if d.class_id == cls]
            cls_gts = [d for d in ground_truth if d.class_id == cls]

            if not cls_gts:
                aps[cls] = 1.0 if not cls_dets else 0.0
                continue

            cls_dets.sort(key=lambda d: d.confidence, reverse=True)

            tp = np.zeros(len(cls_dets))
            fp = np.zeros(len(cls_dets))
            matched_gt = set()

            for i, det in enumerate(cls_dets):
                best_iou = 0
                best_gt = -1

                for j, gt in enumerate(cls_gts):
                    if j in matched_gt:
                        continue
                    iou = Detection3DMetrics._compute_3d_iou(det.bbox, gt.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = j

                if best_iou >= 0.5:
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
    def _compute_3d_iou(bbox1: BoundingBox3D, bbox2: BoundingBox3D) -> float:
        corners1 = bbox1.corners
        corners2 = bbox2.corners

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
            corners_3d = det.bbox.corners
            corners_2d = (camera_matrix @ corners_3d.T).T
            corners_2d = corners_2d[:, :2] / corners_2d[:, 2:3]
            corners_2d = corners_2d.astype(np.int32)

            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]

            color = colors[det.class_id % len(colors)]

            for i, j in edges:
                pt1 = tuple(corners_2d[i])
                pt2 = tuple(corners_2d[j])
                cv2.line(vis, pt1, pt2, color, 2)

            center = tuple(corners_2d.mean(axis=0).astype(int))
            cv2.putText(vis, f"{det.class_name}: {det.confidence:.2f}", center,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return vis

    @staticmethod
    def draw_bev(
        detections: List[Detection3D],
        image_size: Tuple[int, int] = (800, 800),
        range_m: float = 50.0
    ) -> np.ndarray:
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        scale = min(image_size[0], image_size[1]) / (2 * 50.0)

        for det in detections:
            corners = det.bbox.corners[:, :2]
            x = (corners[:, 0] * scale + image_size[0] / 2).astype(int)
            y = (image_size[1] / 2 - corners[:, 1] * scale).astype(int)

            pts = np.column_stack([x, y])
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

        return vis

    @staticmethod
    def draw_tracks(
        image: np.ndarray,
        tracks: List[Track3D],
        camera_matrix: np.ndarray,
        colors: List[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        vis = image.copy()

        if colors is None:
            colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(100)]

        for track in tracks:
            corners_3d = track.bbox.corners
            corners_2d = (camera_matrix @ corners_3d.T).T
            corners_2d = corners_2d[:, :2] / corners_2d[:, 2:3]
            corners_2d = corners_2d.astype(np.int32)

            color = colors[track.track_id % len(colors)]

            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]

            for i, j in edges:
                pt1 = tuple(corners_2d[i])
                pt2 = tuple(corners_2d[j])
                cv2.line(vis, pt1, pt2, color, 2)

            center = tuple(corners_2d[:4].mean(axis=0).astype(int))
            cv2.putText(vis, f"ID:{track.track_id}", center,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return vis


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet