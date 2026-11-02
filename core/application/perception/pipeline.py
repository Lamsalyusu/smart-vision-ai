from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Union
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
import threading
import queue
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
import warnings

class SensorType(Enum):
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"
    IMU = "imu"
    GPS = "gps"
    DEPTH = "depth"
    STEREO = "stereo"
    EVENT = "event"
    THERMAL = "thermal"

class ProcessingStage(Enum):
    ACQUISITION = "acquisition"
    PREPROCESSING = "preprocessing"
    FEATURE_EXTRACTION = "feature_extraction"
    DETECTION = "detection"
    TRACKING = "tracking"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    POSE_ESTIMATION = "pose_estimation"
    FUSION = "fusion"
    OUTPUT = "output"

class DataFormat(Enum):
    RAW = "raw"
    IMAGE = "image"
    POINT_CLOUD = "point_cloud"
    TENSOR = "tensor"
    DETECTIONS = "detections"
    TRACKS = "tracks"
    POSES = "poses"
    MAP = "map"

@dataclass(frozen=True, slots=True)
class SensorData:
    sensor_id: str
    sensor_type: SensorType
    timestamp: float
    frame_id: int
    data: np.ndarray
    format: DataFormat
    metadata: Dict[str, Any] = field(default_factory=dict)
    calibration: Optional[Dict[str, Any]] = None

@dataclass(frozen=True, slots=True)
class ProcessedData:
    source_id: str
    stage: ProcessingStage
    timestamp: float
    data: Any
    format: DataFormat
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: np.ndarray
    mask: Optional[np.ndarray] = None
    keypoints: Optional[np.ndarray] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    track_id: Optional[int] = None

@dataclass(frozen=True, slots=True)
class Track:
    track_id: int
    detections: List[Detection]
    state: np.ndarray
    covariance: np.ndarray
    age: int
    hits: int
    time_since_update: int
    class_id: int
    class_name: str

class SensorInterface(ABC):
    @abstractmethod
    def start(self) -> bool:
        pass

    @abstractmethod
    def stop(self) -> bool:
        pass

    @abstractmethod
    def read(self) -> Optional[SensorData]:
        pass

    @abstractmethod
    def get_calibration(self) -> Dict[str, Any]:
        pass

class CameraInterface(SensorInterface):
    def __init__(self, camera_id: str, index: int = 0,
                 resolution: Tuple[int, int] = (1920, 1080),
                 fps: int = 30, format: str = "bgr"):
        self.camera_id = camera_id
        self.index = index
        self.resolution = resolution
        self.fps = fps
        self.format = format
        self.cap = None
        self.calibration = {}
        self.running = False
        self.frame_count = 0

    def start(self) -> bool:
        import cv2
        self.cap = cv2.VideoCapture(self.index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.running = True
        return self.cap.isOpened()

    def stop(self) -> bool:
        if self.cap:
            self.cap.release()
        self.running = False
        return True

    def read(self) -> Optional[SensorData]:
        if not self.running or not self.cap:
            return None

        import cv2
        ret, frame = self.cap.read()
        if not ret:
            return None

        self.frame_count += 1
        timestamp = time.time()

        return SensorData(
            sensor_id=self.camera_id,
            sensor_type=SensorType.CAMERA,
            timestamp=timestamp,
            frame_id=self.frame_count,
            data=frame,
            format=DataFormat.IMAGE,
            calibration=self.calibration
        )

    def get_calibration(self) -> Dict[str, Any]:
        return self.calibration

    def set_calibration(self, K: np.ndarray, D: np.ndarray,
                        R: np.ndarray = None, P: np.ndarray = None):
        self.calibration = {
            "camera_matrix": K.tolist(),
            "distortion_coeffs": D.tolist(),
            "rectification_matrix": R.tolist() if R is not None else np.eye(3).tolist(),
            "projection_matrix": P.tolist() if P is not None else np.eye(3, 4).tolist()
        }

class LidarInterface(SensorInterface):
    def __init__(self, lidar_id: str, port: str = "/dev/ttyUSB0",
                 baudrate: int = 230400):
        self.lidar_id = lidar_id
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.running = False
        self.frame_count = 0
        self.calibration = {}

    def start(self) -> bool:
        import serial
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            self.running = True
            return True
        except:
            return False

    def stop(self) -> bool:
        if self.serial:
            self.serial.close()
        self.running = False
        return True

    def read(self) -> Optional[SensorData]:
        if not self.running or not self.serial:
            return None

        self.frame_count += 1
        timestamp = time.time()

        points = self._parse_scan()
        if points is None:
            return None

        return SensorData(
            sensor_id=self.lidar_id,
            sensor_type=SensorType.LIDAR,
            timestamp=timestamp,
            frame_id=self.frame_count,
            data=points,
            format=DataFormat.POINT_CLOUD,
            calibration=self.calibration
        )

    def _parse_scan(self) -> Optional[np.ndarray]:
        try:
            line = self.serial.readline().decode('utf-8').strip()
            if not line:
                return None

            parts = line.split(',')
            if len(parts) < 3:
                return None

            angles = np.array([float(p) for p in parts[::3]])
            ranges = np.array([float(p) for p in parts[1::3]])
            intensities = np.array([float(p) for p in parts[2::3]])

            x = ranges * np.cos(np.deg2rad(angles))
            y = ranges * np.sin(np.deg2rad(angles))
            z = np.zeros_like(ranges)

            return np.column_stack([x, y, z, intensities])
        except:
            return None

    def get_calibration(self) -> Dict[str, Any]:
        return self.calibration

class IMUInterface(SensorInterface):
    def __init__(self, imu_id: str, port: str = "/dev/ttyACM0",
                 baudrate: int = 115200):
        self.imu_id = imu_id
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.running = False
        self.frame_count = 0
        self.calibration = {}

    def start(self) -> bool:
        import serial
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            self.running = True
            return True
        except:
            return False

    def stop(self) -> bool:
        if self.serial:
            self.serial.close()
        self.running = False
        return True

    def read(self) -> Optional[SensorData]:
        if not self.running or not self.serial:
            return None

        self.frame_count += 1
        timestamp = time.time()

        data = self._parse_imu()
        if data is None:
            return None

        return SensorData(
            sensor_id=self.imu_id,
            sensor_type=SensorType.IMU,
            timestamp=timestamp,
            frame_id=self.frame_count,
            data=data,
            format=DataFormat.TENSOR,
            calibration=self.calibration
        )

    def _parse_imu(self) -> Optional[np.ndarray]:
        try:
            line = self.serial.readline().decode('utf-8').strip()
            if not line:
                return None

            values = [float(v) for v in line.split(',')]
            if len(values) < 9:
                return None

            accel = np.array(values[0:3])
            gyro = np.array(values[3:6])
            mag = np.array(values[6:9])

            return np.concatenate([accel, gyro, mag])
        except:
            return None

    def get_calibration(self) -> Dict[str, Any]:
        return self.calibration

class Preprocessor:
    def __init__(self):
        self.filters: Dict[str, Callable] = {}

    def add_filter(self, name: str, filter_fn: Callable[[np.ndarray], np.ndarray]):
        self.filters[name] = filter_fn

    def process(self, data: SensorData) -> ProcessedData:
        start = time.time()
        processed = data.data.copy()

        for name, filter_fn in self.filters.items():
            processed = filter_fn(processed)

        return ProcessedData(
            source_id=data.sensor_id,
            stage=ProcessingStage.PREPROCESSING,
            timestamp=data.timestamp,
            data=processed,
            format=data.format,
            processing_time=time.time() - start
        )

    @staticmethod
    def resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        import cv2
        return cv2.resize(image, size)

    @staticmethod
    def normalize(image: np.ndarray, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                  std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        mean = np.array(mean).reshape(1, 1, 3)
        std = np.array(std).reshape(1, 1, 3)
        return (img - mean) / std

    @staticmethod
    def denoise_pointcloud(points: np.ndarray, radius: float = 0.05,
                           min_neighbors: int = 10) -> np.ndarray:
        from scipy.spatial import cKDTree
        tree = cKDTree(points[:, :3])
        counts = tree.query_ball_point(points[:, :3], radius, return_length=True)
        return points[counts >= min_neighbors]

    @staticmethod
    def crop_roi(image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = roi
        return image[y:y+h, x:x+w]

    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        import cv2
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return cv2.equalizeHist(image)

class FeatureExtractor:
    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.input_size = (224, 224)

    def load_model(self, model_path: str):
        self.model_path = model_path

    def extract(self, data: ProcessedData) -> ProcessedData:
        start = time.time()
        features = self._extract_features(data.data)
        return ProcessedData(
            source_id=data.source_id,
            stage=ProcessingStage.FEATURE_EXTRACTION,
            timestamp=data.timestamp,
            data=features,
            format=DataFormat.TENSOR,
            processing_time=time.time() - start
        )

    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.random.randn(512)
        return np.random.randn(512)

class Detector:
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.45, device: str = "cpu"):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self.model = None
        self.class_names = []

    def load_model(self, model_path: str, class_names: List[str] = None):
        self.model_path = model_path
        if class_names:
            self.class_names = class_names

    def detect(self, data: ProcessedData) -> ProcessedData:
        start = time.time()
        detections = self._detect(data.data)
        return ProcessedData(
            source_id=data.source_id,
            stage=ProcessingStage.DETECTION,
            timestamp=data.timestamp,
            data=detections,
            format=DataFormat.DETECTIONS,
            processing_time=time.time() - start
        )

    def _detect(self, image: np.ndarray) -> List[Detection]:
        return []

    def _nms(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            return []

        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.confidence for d in detections])

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= self.nms_threshold)[0]
            order = order[inds + 1]

        return [detections[i] for i in keep]

class Tracker:
    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.frame_count = 0

    def update(self, detections: List[Detection]) -> List[Track]:
        self.frame_count += 1

        if not self.tracks:
            for det in detections:
                self._create_track(det)
            return list(self.tracks.values())

        matched, unmatched_dets, unmatched_trks = self._associate(detections)

        for m in matched:
            self.tracks[m[1]].detections.append(detections[m[0]])
            self._update_track(self.tracks[m[1]], detections[m[0]])

        for i in unmatched_dets:
            self._create_track(detections[i])

        for i in unmatched_trks:
            self.tracks[i].time_since_update += 1

        to_delete = []
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.max_age:
                to_delete.append(track_id)

        for tid in to_delete:
            del self.tracks[tid]

        return list(self.tracks.values())

    def _associate(self, detections: List[Detection]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(self.tracks.keys())

        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(detections), len(track_ids)))

        for d, det in enumerate(detections):
            for t, tid in enumerate(track_ids):
                track = self.tracks[tid]
                last_det = track.detections[-1] if track.detections else None
                if last_det:
                    iou_matrix[d, t] = self._iou(det.bbox, last_det.bbox)

        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(track_ids)))

        if iou_matrix.size > 0:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched.append((r, track_ids[c]))
                    unmatched_dets.remove(r)
                    unmatched_trks.remove(c)

        return matched, unmatched_dets, [track_ids[i] for i in unmatched_trks]

    def _create_track(self, detection: Detection):
        track = Track(
            track_id=self.next_id,
            detections=[detection],
            state=np.array([detection.bbox[0], detection.bbox[1],
                           detection.bbox[2], detection.bbox[3], 0, 0]),
            covariance=np.eye(6) * 10,
            age=1,
            hits=1,
            time_since_update=0,
            class_id=detection.class_id,
            class_name=detection.class_name
        )
        self.tracks[self.next_id] = track
        self.next_id += 1

    def _update_track(self, track: Track, detection: Detection):
        track.age += 1
        track.hits += 1
        track.time_since_update = 0
        track.state[:4] = detection.bbox

    def _iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter)

class Classifier:
    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.class_names = []

    def classify(self, data: ProcessedData) -> ProcessedData:
        start = time.time()
        results = self._classify(data.data)
        return ProcessedData(
            source_id=data.source_id,
            stage=ProcessingStage.CLASSIFICATION,
            timestamp=data.timestamp,
            data=results,
            format=DataFormat.TENSOR,
            processing_time=time.time() - start
        )

    def _classify(self, features: np.ndarray) -> Dict[str, float]:
        return {}

class Segmenter:
    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None

    def segment(self, data: ProcessedData) -> ProcessedData:
        start = time.time()
        masks = self._segment(data.data)
        return ProcessedData(
            source_id=data.source_id,
            stage=ProcessingStage.SEGMENTATION,
            timestamp=data.timestamp,
            data=masks,
            format=DataFormat.IMAGE,
            processing_time=time.time() - start
        )

    def _segment(self, image: np.ndarray) -> np.ndarray:
        return np.zeros(image.shape[:2], dtype=np.uint8)

class PoseEstimator:
    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None

    def estimate(self, data: ProcessedData) -> ProcessedData:
        start = time.time()
        poses = self._estimate(data.data)
        return ProcessedData(
            source_id=data.source_id,
            stage=ProcessingStage.POSE_ESTIMATION,
            timestamp=data.timestamp,
            data=poses,
            format=DataFormat.POSES,
            processing_time=time.time() - start
        )

    def _estimate(self, image: np.ndarray) -> List[np.ndarray]:
        return []

class SensorFusion:
    def __init__(self):
        self.sensors: Dict[str, SensorInterface] = {}
        self.filters: Dict[str, Any] = {}

    def add_sensor(self, sensor: SensorInterface):
        self.sensors[sensor.camera_id if hasattr(sensor, 'camera_id') else
                     sensor.lidar_id if hasattr(sensor, 'lidar_id') else
                     sensor.imu_id] = sensor

    def fuse(self, sensor_data: Dict[str, SensorData]) -> ProcessedData:
        start = time.time()
        fused = self._fuse(sensor_data)
        return ProcessedData(
            source_id="fusion",
            stage=ProcessingStage.FUSION,
            timestamp=time.time(),
            data=fused,
            format=DataFormat.TENSOR,
            processing_time=time.time() - start
        )

    def _fuse(self, sensor_data: Dict[str, SensorData]) -> np.ndarray:
        return np.array([])

class PerceptionPipeline:
    def __init__(self, name: str, num_workers: int = 4):
        self.name = name
        self.num_workers = num_workers
        self.sensors: Dict[str, SensorInterface] = {}
        self.stages: Dict[ProcessingStage, Callable] = {}
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.data_queue: queue.Queue = queue.Queue(maxsize=100)
        self.result_queue: queue.Queue = queue.Queue(maxsize=100)
        self.callbacks: Dict[ProcessingStage, List[Callable]] = defaultdict(list)
        self.statistics: Dict[str, Any] = defaultdict(float)
        self.frame_count = 0

    def add_sensor(self, sensor: SensorInterface):
        sensor_id = getattr(sensor, 'camera_id', None) or \
                    getattr(sensor, 'lidar_id', None) or \
                    getattr(sensor, 'imu_id', None) or \
                    str(len(self.sensors))
        self.sensors[sensor_id] = sensor

    def add_stage(self, stage: ProcessingStage, processor: Callable):
        self.stages[stage] = processor

    def add_callback(self, stage: ProcessingStage, callback: Callable):
        self.callbacks[stage].append(callback)

    def start(self) -> bool:
        for sensor in self.sensors.values():
            if not sensor.start():
                return False
        self.running = True
        self._start_processing_threads()
        return True

    def stop(self):
        self.running = False
        for sensor in self.sensors.values():
            sensor.stop()
        self.executor.shutdown(wait=True)

    def _start_processing_threads(self):
        for sensor_id, sensor in self.sensors.items():
            self.executor.submit(self._sensor_loop, sensor_id, sensor)

    def _sensor_loop(self, sensor_id: str, sensor: SensorInterface):
        while self.running:
            data = sensor.read()
            if data:
                try:
                    self.data_queue.put((sensor_id, data), timeout=0.1)
                except queue.Full:
                    pass
            time.sleep(0.001)

    def process_frame(self, sensor_data: Dict[str, SensorData]) -> Dict[ProcessingStage, ProcessedData]:
        results = {}

        for stage in ProcessingStage:
            if stage not in self.stages:
                continue

            stage_start = time.time()

            if stage == ProcessingStage.ACQUISITION:
                results[stage] = ProcessedData(
                    source_id="pipeline",
                    stage=stage,
                    timestamp=time.time(),
                    data=sensor_data,
                    format=DataFormat.RAW,
                    processing_time=0
                )
            else:
                prev_data = results.get(list(self.stages.keys())[list(self.stages.keys()).index(stage) - 1])
                if prev_data:
                    result = self.stages[stage](prev_data)
                    results[stage] = result

                    for callback in self.callbacks[stage]:
                        callback(result)

            self.statistics[f"stage_{stage.value}_time"] += time.time() - stage_start

        self.frame_count += 1
        return results

    def get_results(self, timeout: float = 1.0) -> Optional[Dict]:
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_statistics(self) -> Dict[str, Any]:
        stats = dict(self.statistics)
        stats["frame_count"] = self.frame_count
        stats["fps"] = self.frame_count / max(sum(self.statistics.values()), 1e-6)
        return stats

class AsyncPerceptionPipeline(PerceptionPipeline):
    def __init__(self, name: str, num_workers: int = 4, buffer_size: int = 10):
        super().__init__(name, num_workers)
        self.buffer_size = buffer_size
        self.frame_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        self.sync_tolerance = 0.05

    def _sensor_loop(self, sensor_id: str, sensor: SensorInterface):
        while self.running:
            data = sensor.read()
            if data:
                self.frame_buffers[sensor_id].append(data)
                self._try_sync_and_process()
            time.sleep(0.001)

    def _try_sync_and_process(self):
        if len(self.frame_buffers) < len(self.sensors):
            return

        latest_times = {sid: buf[-1].timestamp for sid, buf in self.frame_buffers.items()}
        max_time = max(latest_times.values())
        min_time = min(latest_times.values())

        if max_time - min_time > self.sync_tolerance:
            return

        synced_data = {}
        for sid, buf in self.frame_buffers.items():
            best_frame = min(buf, key=lambda f: abs(f.timestamp - max_time))
            synced_data[sid] = best_frame

        future = self.executor.submit(self.process_frame, synced_data)
        future.add_done_callback(lambda f: self.result_queue.put(f.result()))

class MultiCameraPipeline(PerceptionPipeline):
    def __init__(self, name: str, cameras: List[CameraInterface], num_workers: int = 4):
        super().__init__(name, num_workers)
        for cam in cameras:
            self.add_sensor(cam)

    def calibrate_stereo(self, cam1_id: str, cam2_id: str,
                         chessboard_size: Tuple[int, int] = (9, 6),
                         square_size: float = 0.025) -> Dict[str, Any]:
        cam1 = self.sensors[cam1_id]
        cam2 = self.sensors[cam2_id]

        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

        objpoints = []
        imgpoints1 = []
        imgpoints2 = []

        for _ in range(20):
            frame1 = cam1.read()
            frame2 = cam2.read()

            if frame1 is None or frame2 is None:
                continue

            gray1 = cv2.cvtColor(frame1.data, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2.data, cv2.COLOR_BGR2GRAY)

            ret1, corners1 = cv2.findChessboardCorners(gray1, chessboard_size, None)
            ret2, corners2 = cv2.findChessboardCorners(gray2, chessboard_size, None)

            if ret1 and ret2:
                objpoints.append(objp)
                corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                imgpoints1.append(corners1)
                imgpoints2.append(corners2)

        if len(objpoints) < 5:
            return {}

        ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints1, imgpoints2,
            cam1.get_calibration().get("camera_matrix", np.eye(3)),
            cam1.get_calibration().get("distortion_coeffs", np.zeros(5)),
            cam2.get_calibration().get("camera_matrix", np.eye(3)),
            cam2.get_calibration().get("distortion_coeffs", np.zeros(5)),
            gray1.shape[::-1]
        )

        return {
            "camera1_matrix": K1.tolist(),
            "camera1_dist": D1.tolist(),
            "camera2_matrix": K2.tolist(),
            "camera2_dist": D2.tolist(),
            "R": R.tolist(),
            "T": T.tolist(),
            "E": E.tolist(),
            "F": F.tolist(),
            "rms_error": ret
        }

class LidarCameraPipeline(PerceptionPipeline):
    def __init__(self, name: str, lidar: LidarInterface, camera: CameraInterface,
                 num_workers: int = 4):
        super().__init__(name, num_workers)
        self.add_sensor(lidar)
        self.add_sensor(camera)
        self.lidar_camera_transform = np.eye(4)

    def set_calibration(self, transform: np.ndarray):
        self.lidar_camera_transform = transform

    def project_lidar_to_camera(self, points: np.ndarray, K: np.ndarray) -> np.ndarray:
        points_h = np.hstack([points[:, :3], np.ones((len(points), 1))])
        points_cam = (self.lidar_camera_transform @ points_h.T).T
        points_cam = points_cam[:, :3]

        valid = points_cam[:, 2] > 0
        points_cam = points_cam[valid]

        uv = (K @ points_cam.T).T
        uv = uv[:, :2] / uv[:, 2:3]

        return uv.astype(int)

    def fuse_lidar_camera(self, lidar_data: SensorData, camera_data: SensorData) -> ProcessedData:
        start = time.time()

        points = lidar_data.data
        image = camera_data.data

        K = np.array(camera_data.calibration.get("camera_matrix", np.eye(3)))
        uv = self.project_lidar_to_camera(points, K)

        h, w = image.shape[:2]
        valid = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        uv = uv[valid]
        points = points[valid]

        fused_image = image.copy()
        for pt, p3d in zip(uv, points):
            color = (0, 255, 0) if p3d[2] < 10 else (0, 0, 255)
            cv2.circle(fused_image, tuple(pt), 2, color, -1)

        return ProcessedData(
            source_id="lidar_camera_fusion",
            stage=ProcessingStage.FUSION,
            timestamp=time.time(),
            data={"image": fused_image, "points": points, "uv": uv},
            format=DataFormat.IMAGE,
            processing_time=time.time() - start
        )

def create_default_pipeline(name: str = "default") -> PerceptionPipeline:
    pipeline = PerceptionPipeline(name)

    preprocessor = Preprocessor()
    preprocessor.add_filter("resize", lambda img: Preprocessor.resize(img, (640, 480)))
    preprocessor.add_filter("normalize", Preprocessor.normalize)
    pipeline.add_stage(ProcessingStage.PREPROCESSING, preprocessor.process)

    detector = Detector(confidence_threshold=0.5, nms_threshold=0.45)
    pipeline.add_stage(ProcessingStage.DETECTION, detector.detect)

    tracker = Tracker()
    pipeline.add_stage(ProcessingStage.TRACKING, lambda d: ProcessedData(
        source_id=d.source_id, stage=ProcessingStage.TRACKING,
        timestamp=d.timestamp, data=tracker.update(d.data),
        format=DataFormat.TRACKS, processing_time=0
    ))

    return pipeline