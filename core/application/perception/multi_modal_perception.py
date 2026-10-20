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


class Modality(Enum):
    RGB = "rgb"
    DEPTH = "depth"
    THERMAL = "thermal"
    LIDAR = "lidar"
    RADAR = "radar"
    EVENT = "event"
    IMU = "imu"
    GPS = "gps"
    AUDIO = "audio"


@dataclass(frozen=True, slots=True)
class SensorData:
    modality: Modality
    timestamp: float
    data: np.ndarray
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})


@dataclass(frozen=True, slots=True)
class FusedPerception:
    timestamp: float
    detections: List[Dict]
    segmentations: List[np.ndarray]
    depth_map: Optional[np.ndarray]
    point_cloud: Optional[np.ndarray]
    poses: List[np.ndarray]
    uncertainties: Dict[str, float]


class BaseModalityProcessor(ABC):
    @abstractmethod
    def process(self, data: SensorData) -> Any:
        pass


class RGBProcessor(BaseModalityProcessor):
    def __init__(
        self,
        detector_type: str = "yolo",
        confidence_threshold: float = 0.5,
    ):
        self.detector_type = detector_type
        self.confidence_threshold = confidence_threshold

    def process(self, data: SensorData) -> Dict[str, Any]:
        if len(data.data.shape) == 3:
            gray = cv2.cvtColor(data.data, cv2.COLOR_RGB2GRAY)
        else:
            gray = data.data

        detections = self._detect_objects(gray)

        return {
            'detections': detections,
            'features': self._extract_features(data.data),
            'timestamp': data.timestamp,
        }

    def _detect_objects(self, image: np.ndarray) -> List[Dict]:
        detections = []
        edges = cv2.Canny(image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detections.append({
                'bbox': [x, y, x + w, y + h],
                'confidence': 0.5,
                'class': 'object',
            })

        return detections

    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        hog = cv2.HOGDescriptor()
        return hog.compute(gray).flatten()


class DepthProcessor(BaseModalityProcessor):
    def __init__(
        self,
        min_depth: float = 0.1,
        max_depth: float = 10.0,
    ):
        self.min_depth = min_depth
        self.max_depth = max_depth

    def process(self, data: SensorData) -> Dict[str, Any]:
        depth = data.data

        depth = np.clip(depth, self.min_depth, self.max_depth)

        point_cloud = self._depth_to_point_cloud(depth)

        return {
            'depth_map': depth,
            'point_cloud': point_cloud,
            'normals': self._estimate_normals(depth),
            'timestamp': data.timestamp,
        }

    def _depth_to_point_cloud(self, depth: np.ndarray) -> np.ndarray:
        h, w = depth.shape
        fx, fy = w, w
        cx, cy = w / 2, h / 2

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.flatten()
        valid = (z > self.min_depth) & (z < self.max_depth)

        u, v, z = u.flatten()[valid], v.flatten()[valid], z[valid]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        return np.column_stack([x, y, z])

    def _estimate_normals(self, depth: np.ndarray) -> np.ndarray:
        h, w = depth.shape
        normals = np.zeros((h, w, 3))

        for v in range(1, h - 1):
            for u in range(1, w - 1):
                z_c = depth[v, u]
                z_u = depth[v, u + 1]
                z_v = depth[v + 1, u]

                if z_u <= 0 or z_v <= 0 or z_c <= 0:
                    continue

                fx, fy = w, w
                cx, cy = w / 2, h / 2

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
                    normals[v, u] = normal

        return normals


class ThermalProcessor(BaseModalityProcessor):
    def __init__(
        self,
        temp_range: Tuple[float, float] = (0, 100),
    ):
        self.temp_range = temp_range

    def process(self, data: SensorData) -> Dict[str, Any]:
        thermal = data.data

        thermal = np.clip(thermal, self.temp_range[0], self.temp_range[1])
        normalized = (thermal - self.temp_range[0]) / (self.temp_range[1] - self.temp_range[0])

        hotspots = self._detect_hotspots(thermal)

        return {
            'thermal_map': thermal,
            'normalized': normalized,
            'hotspots': hotspots,
            'max_temp': float(np.max(thermal)),
            'min_temp': float(np.min(thermal)),
            'timestamp': data.timestamp,
        }

    def _detect_hotspots(self, thermal: np.ndarray) -> List[Dict]:
        hotspots = []
        threshold = np.mean(thermal) + 2 * np.std(thermal)

        hot_mask = thermal > threshold
        if not np.any(hot_mask):
            return hotspots

        num_labels, labels = cv2.connectedComponents(hot_mask.astype(np.uint8))

        for i in range(1, num_labels):
            mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 10:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                roi_temp = thermal[mask == 1]
                hotspots.append({
                    'bbox': (x, y, x + w, y + h),
                    'max_temp': float(np.max(roi_temp)),
                    'mean_temp': float(np.mean(roi_temp)),
                    'area': float(cv2.contourArea(contour)),
                })

        return hotspots


class LiDARProcessor(BaseModalityProcessor):
    def __init__(
        self,
        voxel_size: float = 0.1,
        max_range: float = 100.0,
    ):
        self.voxel_size = voxel_size
        self.max_range = max_range

    def process(self, data: SensorData) -> Dict[str, Any]:
        points = data.data

        if points.ndim == 2 and points.shape[1] >= 3:
            points = points[:, :3]
        else:
            return {'point_cloud': np.array([]), 'timestamp': data.timestamp}

        points = points[np.linalg.norm(points, axis=1) < self.max_range]

        downsampled = self._voxel_downsample(points)
        clusters = self._euclidean_cluster(downsampled)

        return {
            'point_cloud': downsampled,
            'clusters': clusters,
            'num_points': len(downsampled),
            'timestamp': data.timestamp,
        }

    def _voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        coords = np.floor(points / self.voxel_size).astype(np.int32)
        _, indices = np.unique(coords, axis=0, return_index=True)
        return points[indices]

    def _euclidean_cluster(self, points: np.ndarray, tolerance: float = 0.5, min_size: int = 10) -> List[np.ndarray]:
        if len(points) < 2:
            return [points]

        from scipy.spatial import KDTree
        kdtree = KDTree(points)

        visited = np.zeros(len(points), dtype=bool)
        clusters = []

        for i in range(len(points)):
            if visited[i]:
                continue

            cluster_indices = []
            stack = [i]
            visited[i] = True

            while stack:
                idx = stack.pop()
                cluster_indices.append(idx)

                neighbors = kdtree.query_radius(points[idx], tolerance)
                for n_idx in neighbors:
                    if not visited[n_idx]:
                        visited[n_idx] = True
                        stack.append(n_idx)

            if len(cluster_indices) >= 10:
                clusters.append(points[cluster_indices])

        return clusters


class RadarProcessor(BaseModalityProcessor):
    def __init__(
        self,
        max_range: float = 200.0,
        max_velocity: float = 50.0,
    ):
        self.max_range = max_range
        self.max_velocity = max_velocity

    def process(self, data: SensorData) -> Dict[str, Any]:
        radar_data = data.data

        detections = []
        for detection in radar_data:
            range_val = detection[0]
            azimuth = detection[1]
            elevation = detection[2]
            velocity = detection[3] if len(detection) > 3 else 0

            if range_val > self.max_range:
                continue

            x = range_val * np.cos(elevation) * np.cos(azimuth)
            y = range_val * np.cos(elevation) * np.sin(azimuth)
            z = range_val * np.sin(elevation)

            detections.append({
                'position': np.array([x, y, z]),
                'velocity': velocity,
                'range': range_val,
                'azimuth': azimuth,
                'elevation': elevation,
            })

        return {
            'detections': detections,
            'num_detections': len(detections),
            'timestamp': data.timestamp,
        }


class IMUProcessor(BaseModalityProcessor):
    def __init__(
        self,
        gravity: np.ndarray = None,
    ):
        self.gravity = gravity if gravity is not None else np.array([0, 0, -9.81])

    def process(self, data: SensorData) -> Dict[str, Any]:
        imu_data = data.data

        accel = imu_data[:3] if len(imu_data) >= 3 else np.zeros(3)
        gyro = imu_data[3:6] if len(imu_data) >= 6 else np.zeros(3)

        orientation = self._integrate_orientation(gyro, data.timestamp)
        linear_acc = self._remove_gravity(accel, self._get_orientation_quaternion(data.timestamp))

        return {
            'acceleration': accel,
            'angular_velocity': gyro,
            'orientation': orientation,
            'linear_acceleration': linear_acc,
            'timestamp': data.timestamp,
        }

    def _integrate_orientation(self, gyro: np.ndarray, timestamp: float) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0])

    def _get_orientation_quaternion(self, timestamp: float) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0])

    def _remove_gravity(self, accel: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        return accel


class GPSProcessor(BaseModalityProcessor):
    def __init__(
        self,
        datum: str = "WGS84",
    ):
        self.datum = datum

    def process(self, data: SensorData) -> Dict[str, Any]:
        gps_data = data.data

        lat = gps_data[0] if len(gps_data) > 0 else 0
        lon = gps_data[1] if len(gps_data) > 1 else 0
        alt = gps_data[2] if len(gps_data) > 2 else 0

        ecef = self._lla_to_ecef(lat, lon, alt)

        return {
            'latitude': lat,
            'longitude': lon,
            'altitude': alt,
            'ecef_position': ecef,
            'timestamp': data.timestamp,
        }

    def _lla_to_ecef(self, lat: float, lon: float, alt: float) -> np.ndarray:
        a = 6378137.0
        f = 1 / 298.257223563
        e2 = 2 * f - f**2

        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)

        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - e2) + alt) * np.sin(lat_rad)

        return np.array([x, y, z])


class MultiModalFusion:
    def __init__(
        self,
        processors: Dict[Modality, BaseModalityProcessor],
        fusion_method: str = "kalman",
    ):
        self.processors = processors
        self.fusion_method = fusion_method
        self.kf = None
        self.last_timestamp = 0.0

    def add_processor(self, modality: Modality, processor: BaseModalityProcessor):
        self.processors[modality] = processor

    def process(self, sensor_data: List[SensorData]) -> FusedPerception:
        results = {}
        for sensor in sensor_data:
            if sensor.modality in self.processors:
                results[sensor.modality] = self.processors[sensor.modality].process(sensor)

        fused = self._fuse_results(results)
        return fused

    def _fuse_results(self, results: Dict[Modality, Dict]) -> FusedPerception:
        detections = []
        segmentations = []
        depth_map = None
        point_cloud = None
        poses = []
        uncertainties = {}

        if Modality.RGB in results:
            rgb_result = results[Modality.RGB]
            detections.extend(rgb_result.get('detections', []))
            uncertainties['rgb'] = 0.1

        if Modality.DEPTH in results:
            depth_result = results[Modality.DEPTH]
            depth_map = depth_result.get('depth_map')
            point_cloud = depth_result.get('point_cloud')
            uncertainties['depth'] = 0.05

        if Modality.LIDAR in results:
            lidar_result = results[Modality.LIDAR]
            if point_cloud is None:
                point_cloud = lidar_result.get('point_cloud')
            uncertainties['lidar'] = 0.02

        if Modality.RADAR in results:
            radar_result = results[Modality.RADAR]
            uncertainties['radar'] = 0.1

        if Modality.IMU in results:
            imu_result = results[Modality.IMU]
            poses.append(imu_result.get('orientation', np.eye(3)))
            uncertainties['imu'] = 0.01

        if Modality.GPS in results:
            gps_result = results[Modality.GPS]
            poses.append(gps_result.get('ecef_position', np.zeros(3)))
            uncertainties['gps'] = 1.0

        return FusedPerception(
            timestamp=time.time(),
            detections=detections,
            segmentations=segmentations,
            depth_map=depth_map,
            point_cloud=point_cloud,
            poses=poses,
            uncertainties=uncertainties,
        )


class MultiModalTracker:
    def __init__(
        self,
        max_tracks: int = 100,
        iou_threshold: float = 0.3,
        max_age: int = 30,
    ):
        self.max_tracks = max_tracks
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: Dict[str, Dict] = {}
        self.next_id = 0

    def update(self, fused: FusedPerception) -> List[Dict]:
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            track['age'] += 1
            track['time_since_update'] += 1

        detections = fused.detections
        if not detections:
            for track_id in list(self.tracks.keys()):
                if self.tracks[track_id]['time_since_update'] > self.max_age:
                    del self.tracks[track_id]
            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())
        track_bboxes = np.array([self.tracks[tid]['bbox'] for tid in track_ids])
        det_bboxes = np.array([d['bbox'] for d in detections])

        iou_matrix = self._iou_matrix(track_bboxes, det_bboxes)
        cost_matrix = 1 - iou_matrix

        matches = self._hungarian(cost_matrix)

        matched_tracks = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(track_ids)))

        for track_idx, det_idx in matches:
            if cost_matrix[track_idx, det_idx] > 1 - self.iou_threshold:
                continue
            matched_tracks.append((track_ids[track_idx], det_idx))
            unmatched_dets.remove(det_idx)
            unmatched_tracks.remove(track_idx)

        for track_id, det_idx in matched_tracks:
            det = detections[det_idx]
            track = self.tracks[track_id]

            track['bbox'] = det['bbox']
            track['confidence'] = det['confidence']
            track['age'] += 1
            track['hits'] += 1
            track['hit_streak'] += 1
            track['time_since_update'] = 0

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            track_id = f"track_{self.next_id}"
            self.next_id += 1

            self.tracks[track_id] = {
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'age': 1,
                'hits': 1,
                'hit_streak': 1,
                'time_since_update': 0,
            }

        for track_idx in unmatched_tracks:
            track_id = track_ids[track_idx]
            if self.tracks[track_id]['time_since_update'] > self.max_age:
                del self.tracks[track_id]

        return [t for t in self.tracks.values() if t['hits'] >= 3 or t['time_since_update'] == 0]

    def _hungarian(self, cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return list(zip(row_ind, col_ind))

    def _iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        n1, n2 = len(boxes1), len(boxes2)
        iou = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                iou[i, j] = self._iou(boxes1[i], boxes2[j])
        return iou

    def _iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-6)


class MultiModalCalibration:
    def __init__(self):
        self.extrinsics: Dict[str, np.ndarray] = {}
        self.intrinsics: Dict[str, np.ndarray] = {}
        self.time_offsets: Dict[str, float] = {}

    def set_extrinsic(self, sensor_a: str, sensor_b: str, transform: np.ndarray):
        key = f"{sensor_a}_{sensor_b}"
        self.extrinsics[key] = transform

    def set_intrinsic(self, sensor_id: str, matrix: np.ndarray):
        self.intrinsics[sensor_id] = matrix

    def set_time_offset(self, sensor_id: str, offset: float):
        self.time_offsets[sensor_id] = offset

    def get_transform(self, sensor_a: str, sensor_b: str) -> Optional[np.ndarray]:
        key = f"{sensor_a}_{sensor_b}"
        return self.extrinsics.get(key)

    def calibrate_camera_lidar(self, camera_points: np.ndarray, lidar_points: np.ndarray) -> np.ndarray:
        if len(camera_points) != len(lidar_points) or len(camera_points) < 6:
            return np.eye(4)

        centroid_cam = np.mean(camera_points, axis=0)
        centroid_lidar = np.mean(lidar_points, axis=0)

        centered_cam = camera_points - centroid_cam
        centered_lidar = lidar_points - centroid_lidar

        H = centered_lidar.T @ centered_cam
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_cam - R @ centroid_lidar

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return T


class MultiModalDataset:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.samples: List[Dict] = []

    def add_sample(self, sample: Dict):
        self.samples.append(sample)

    def get_sample(self, index: int) -> Dict:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        return self.samples[index]

    def split(self, train_ratio: float = 0.8) -> Tuple['MultiModalDataset', 'MultiModalDataset']:
        n = len(self.samples)
        train_size = int(n * train_ratio)
        train_ds = MultiModalDataset(self.root_dir)
        val_ds = MultiModalDataset(self.root_dir)
        train_ds.samples = self.samples[:train_size]
        val_ds.samples = self.samples[train_size:]
        return train_ds, val_ds


class MultiModalTrainer:
    def __init__(
        self,
        model: Any,
        dataset: MultiModalDataset,
        optimizer: Any,
        loss_fn: Callable,
    ):
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, epochs: int, batch_size: int = 32) -> List[float]:
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(0, len(self.dataset), batch_size):
                batch = [self.dataset[j] for j in range(i, min(i + batch_size, len(self.dataset)))]

                loss = self._train_batch(batch)
                epoch_loss += loss

            losses.append(epoch_loss / max(1, len(self.dataset) // batch_size))
        return losses

    def _train_batch(self, batch: List[Dict]) -> float:
        return 0.0


class MultiModalEvaluator:
    def __init__(self, model: Any, dataset: MultiModalDataset):
        self.model = model
        self.dataset = dataset

    def evaluate(self, metrics: List[str] = None) -> Dict[str, float]:
        metrics = metrics or ['accuracy', 'precision', 'recall', 'f1']
        results = {}

        for metric in metrics:
            if metric == 'accuracy':
                results[metric] = self._compute_accuracy()
            elif metric == 'precision':
                results[metric] = self._compute_precision()
            elif metric == 'recall':
                results[metric] = self._compute_recall()
            elif metric == 'f1':
                results[metric] = self._compute_f1()

        return results

    def _compute_accuracy(self) -> float:
        return 0.0

    def _compute_precision(self) -> float:
        return 0.0

    def _compute_recall(self) -> float:
        return 0.0

    def _compute_f1(self) -> float:
        return 0.0
