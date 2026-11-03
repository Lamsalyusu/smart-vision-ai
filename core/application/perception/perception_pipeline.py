from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict, deque
import copy


class SensorType(Enum):
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"
    IMU = "imu"
    GPS = "gps"
    DEPTH = "depth"
    THERMAL = "thermal"
    EVENT = "event"


class PerceptionTask(Enum):
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    TRACKING = "tracking"
    POSE_ESTIMATION = "pose_estimation"
    DEPTH_ESTIMATION = "depth_estimation"
    OPTICAL_FLOW = "optical_flow"
    SCENE_UNDERSTANDING = "scene_understanding"
    OBJECT_RECOGNITION = "object_recognition"


@dataclass(frozen=True, slots=True)
class SensorData:
    sensor_id: str
    sensor_type: SensorType
    timestamp: float
    frame_id: int
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    transform: Optional[np.ndarray] = None


@dataclass(frozen=True, slots=True)
class PerceptionResult:
    task: PerceptionTask
    sensor_id: str
    timestamp: float
    frame_id: int
    detections: List[Dict[str, Any]] = field(default_factory=list)
    segmentation: Optional[np.ndarray] = None
    tracks: List[Dict[str, Any]] = field(default_factory=list)
    poses: List[np.ndarray] = field(default_factory=list)
    depth: Optional[np.ndarray] = None
    flow: Optional[np.ndarray] = None
    scene_graph: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    processing_time: float = 0.0


@dataclass(frozen=True, slots=True)
class FusedPerceptionResult:
    timestamp: float
    frame_id: int
    objects: List[Dict[str, Any]] = field(default_factory=list)
    tracks: List[Dict[str, Any]] = field(default_factory=list)
    scene_graph: Optional[Dict[str, Any]] = None
    occupancy_grid: Optional[np.ndarray] = None
    confidence: float = 1.0
    sensor_contributions: Dict[str, float] = field(default_factory=dict)


class PerceptionModule(ABC):
    def __init__(self, name: str, tasks: List[PerceptionTask]):
        self.name = name
        self.tasks = tasks
        self.enabled = True

    @abstractmethod
    def process(self, sensor_data: SensorData) -> List[PerceptionResult]:
        pass

    def can_process(self, task: PerceptionTask) -> bool:
        return task in self.tasks and self.enabled


class DetectionModule(PerceptionModule):
    def __init__(
        self,
        model_path: str = "detector.onnx",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640)
    ):
        super().__init__("detection", [PerceptionTask.DETECTION])
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self._model = None

    def _initialize(self):
        try:
            import onnxruntime as ort
            self._model = ort.InferenceSession(self.model_path)
        except ImportError:
            self._model = None

    def process(self, sensor_data: SensorData) -> List[PerceptionResult]:
        if sensor_data.sensor_type != SensorType.CAMERA:
            return []

        if self._model is None:
            self._initialize()

        if self._model is None:
            return []

        start_time = time.time()
        image = sensor_data.data

        if len(image.shape) == 3:
            h, w = image.shape[:2]
            resized = cv2.resize(image, self.input_size)
            input_tensor = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
            input_tensor = np.expand_dims(input_tensor, axis=0)

            outputs = self._model.run(None, {self._model.get_inputs()[0].name: input_tensor})

            detections = self._postprocess(outputs[0], (h, w))

            return [PerceptionResult(
                task=PerceptionTask.DETECTION,
                sensor_id=sensor_data.sensor_id,
                timestamp=sensor_data.timestamp,
                frame_id=sensor_data.frame_id,
                detections=detections,
                processing_time=time.time() - start_time
            )]

        return []

    def _postprocess(self, output: np.ndarray, original_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        h, w = original_shape
        detections = []

        for det in output:
            if det[4] > self.confidence_threshold:
                x1, y1, x2, y2 = det[:4]
                x1 = int(x1 * w / self.input_size[0])
                y1 = int(y1 * h / self.input_size[1])
                x2 = int(x2 * w / self.input_size[0])
                y2 = int(y2 * h / self.input_size[1])

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(det[4]),
                    'class_id': int(det[5]),
                    'class_name': f"class_{int(det[5])}"
                })

        return detections


class SegmentationModule(PerceptionModule):
    def __init__(
        self,
        model_path: str = "segmentation.onnx",
        input_size: Tuple[int, int] = (512, 512),
        num_classes: int = 19
    ):
        super().__init__("segmentation", [PerceptionTask.SEGMENTATION])
        self.model_path = model_path
        self.input_size = input_size
        self.num_classes = num_classes
        self._model = None

    def _initialize(self):
        try:
            import onnxruntime as ort
            self._model = ort.InferenceSession(self.model_path)
        except ImportError:
            self._model = None

    def process(self, sensor_data: SensorData) -> List[PerceptionResult]:
        if sensor_data.sensor_type != SensorType.CAMERA:
            return []

        if self._model is None:
            self._initialize()

        if self._model is None:
            return []

        start_time = time.time()
        image = sensor_data.data

        if len(image.shape) == 3:
            h, w = image.shape[:2]
            resized = cv2.resize(image, self.input_size)
            input_tensor = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
            input_tensor = np.expand_dims(input_tensor, axis=0)

            outputs = self._model.run(None, {self._model.get_inputs()[0].name: input_tensor})
            segmentation = outputs[0].argmax(axis=1)[0]
            segmentation = cv2.resize(segmentation.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

            return [PerceptionResult(
                task=PerceptionTask.SEGMENTATION,
                sensor_id=sensor_data.sensor_id,
                timestamp=sensor_data.timestamp,
                frame_id=sensor_data.frame_id,
                segmentation=segmentation,
                processing_time=time.time() - start_time
            )]

        return []


class TrackingModule(PerceptionModule):
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        super().__init__("tracking", [PerceptionTask.TRACKING])
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: Dict[int, Dict] = {}
        self.next_id = 0

    def process(self, sensor_data: SensorData) -> List[PerceptionResult]:
        if sensor_data.sensor_type != SensorType.CAMERA:
            return []

        start_time = time.time()

        detections = sensor_data.metadata.get('detections', [])

        self._update_trackers(detections)

        tracks = []
        for track_id, track in self.trackers.items():
            if track['hits'] >= self.min_hits and track['age'] <= self.max_age:
                tracks.append({
                    'track_id': track_id,
                    'bbox': track['bbox'],
                    'confidence': track['confidence'],
                    'class_id': track['class_id'],
                    'age': track['age'],
                    'hits': track['hits']
                })

        return [PerceptionResult(
            task=PerceptionTask.TRACKING,
            sensor_id=sensor_data.sensor_id,
            timestamp=sensor_data.timestamp,
            frame_id=sensor_data.frame_id,
            tracks=tracks,
            processing_time=time.time() - start_time
        )]

    def _update_trackers(self, detections: List[Dict]):
        for track_id in list(self.trackers.keys()):
            self.trackers[track_id]['age'] += 1
            if self.trackers[track_id]['age'] > self.max_age:
                del self.trackers[track_id]

        if not detections:
            return

        matched = set()
        for det in detections:
            best_iou = 0
            best_id = -1

            for track_id, track in self.trackers.items():
                if track_id in matched:
                    continue
                iou = self._compute_iou(det['bbox'], track['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_id = track_id

            if best_id != -1:
                track = self.trackers[best_id]
                track['bbox'] = det['bbox']
                track['confidence'] = det['confidence']
                track['age'] = 0
                track['hits'] += 1
                matched.add(best_id)
            else:
                self.trackers[self.next_id] = {
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'class_id': det['class_id'],
                    'age': 0,
                    'hits': 1
                }
                self.next_id += 1

    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0


class DepthEstimationModule(PerceptionModule):
    def __init__(
        self,
        model_path: str = "depth.onnx",
        input_size: Tuple[int, int] = (384, 384),
        min_depth: float = 0.1,
        max_depth: float = 100.0
    ):
        super().__init__("depth_estimation", [PerceptionTask.DEPTH_ESTIMATION])
        self.model_path = model_path
        self.input_size = input_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self._model = None

    def _initialize(self):
        try:
            import onnxruntime as ort
            self._model = ort.InferenceSession(self.model_path)
        except ImportError:
            self._model = None

    def process(self, sensor_data: SensorData) -> List[PerceptionResult]:
        if sensor_data.sensor_type not in [SensorType.CAMERA, SensorType.DEPTH]:
            return []

        if self._model is None:
            self._initialize()

        if self._model is None:
            return []

        start_time = time.time()
        image = sensor_data.data

        if len(image.shape) == 3:
            h, w = image.shape[:2]
            resized = cv2.resize(image, self.input_size)
            input_tensor = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
            input_tensor = np.expand_dims(input_tensor, axis=0)

            outputs = self._model.run(None, {self._model.get_inputs()[0].name: input_tensor})
            depth = outputs[0][0, 0]
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            depth = np.clip(depth, self.min_depth, self.max_depth)

            return [PerceptionResult(
                task=PerceptionTask.DEPTH_ESTIMATION,
                sensor_id=sensor_data.sensor_id,
                timestamp=sensor_data.timestamp,
                frame_id=sensor_data.frame_id,
                depth=depth,
                processing_time=time.time() - start_time
            )]

        return []


class OpticalFlowModule(PerceptionModule):
    def __init__(
        self,
        method: str = "farneback",
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2
    ):
        super().__init__("optical_flow", [PerceptionTask.OPTICAL_FLOW])
        self.method = method
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.prev_frame = None

    def process(self, frame: SensorFrame) -> List[PerceptionResult]:
        if frame.modality != SensorModality.RGB:
            return []

        start_time = time.time()

        if len(frame.data.shape) == 3:
            gray = cv2.cvtColor(frame.data, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame.data

        if self.prev_frame is None:
            self.prev_frame = gray
            return []

        if self.method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_frame, gray, None,
                self.pyr_scale, self.levels, self.winsize,
                self.iterations, self.poly_n, self.poly_sigma, 0
            )
        elif self.method == "lucas_kanade":
            pass

        self.prev_frame = gray

        return [PerceptionResult(
            task=PerceptionTask.OPTICAL_FLOW,
            sensor_id=frame.sensor_id,
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
            flow=flow,
            processing_time=time.time() - start_time
        )]


class PerceptionFusion:
    def __init__(
        self,
        max_time_diff: float = 0.1,
        spatial_threshold: float = 0.5
    ):
        self.max_time_diff = max_time_diff
        self.spatial_threshold = spatial_threshold
        self.pending_results: Dict[str, List[PerceptionResult]] = defaultdict(list)

    def add_result(self, result: PerceptionResult):
        self.pending_results[result.sensor_id].append(result)
        self._cleanup_old_results(result.timestamp)

    def _cleanup_old_results(self, current_time: float):
        for sensor_id in list(self.pending_results.keys()):
            self.pending_results[sensor_id] = [
                r for r in self.pending_results[sensor_id]
                if current_time - r.timestamp < self.max_time_diff * 2
            ]

    def fuse(self, timestamp: float, frame_id: int) -> Optional[FusedPerceptionResult]:
        results_by_task = defaultdict(list)

        for sensor_id, results in self.pending_results.items():
            for r in results:
                if abs(r.timestamp - timestamp) < self.max_time_diff:
                    results_by_task[r.task].append(r)

        if not results_by_task:
            return None

        fused_objects = self._fuse_detections(results_by_task.get(PerceptionTask.DETECTION, []))
        fused_tracks = self._fuse_tracks(results_by_task.get(PerceptionTask.TRACKING, []))
        fused_segmentation = self._fuse_segmentation(results_by_task.get(PerceptionTask.SEGMENTATION, []))

        contributions = {}
        total_results = sum(len(r) for r in results_by_task.values())
        for task, results in results_by_task.items():
            for r in results:
                contributions[r.sensor_id] = contributions.get(r.sensor_id, 0) + 1 / total_results

        return FusedPerceptionResult(
            timestamp=timestamp,
            frame_id=frame_id,
            objects=fused_objects,
            tracks=fused_tracks,
            scene_graph=None,
            occupancy_grid=None,
            confidence=1.0,
            sensor_contributions=contributions
        )

    def _fuse_detections(self, detections_results: List[PerceptionResult]) -> List[Dict[str, Any]]:
        all_detections = []
        for r in detections_results:
            all_detections.extend(r.detections)

        if not all_detections:
            return []

        fused = []
        used = set()

        for i, det1 in enumerate(all_detections):
            if i in used:
                continue

            cluster = [det1]
            used.add(i)

            for j, det2 in enumerate(all_detections):
                if j in used:
                    continue
                iou = self._compute_iou(det1['bbox'], det2['bbox'])
                if iou > self.spatial_threshold:
                    cluster.append(det2)
                    used.add(j)

            avg_bbox = np.mean([d['bbox'] for d in cluster], axis=0).astype(int).tolist()
            avg_conf = np.mean([d['confidence'] for d in cluster])
            class_id = max(set(d['class_id'] for d in cluster), key=lambda c: sum(1 for d in cluster if d['class_id'] == c))

            fused.append({
                'bbox': avg_bbox,
                'confidence': float(avg_conf),
                'class_id': class_id,
                'class_name': cluster[0]['class_name'],
                'source_count': len(cluster)
            })

        return fused

    def _fuse_tracks(self, tracks_results: List[PerceptionResult]) -> List[Dict[str, Any]]:
        all_tracks = []
        for r in tracks_results:
            all_tracks.extend(r.tracks)

        if not all_tracks:
            return []

        fused = {}
        for track in all_tracks:
            tid = track['track_id']
            if tid not in fused:
                fused[tid] = track
            else:
                fused[tid]['confidence'] = max(fused[tid]['confidence'], track['confidence'])
                fused[tid]['hits'] += track['hits']

        return list(fused.values())

    def _fuse_segmentation(self, seg_results: List[PerceptionResult]) -> Optional[np.ndarray]:
        if not seg_results:
            return None

        seg = seg_results[0].segmentation
        if seg is None:
            return None

        for r in seg_results[1:]:
            if r.segmentation is not None:
                seg = np.maximum(seg, r.segmentation)

        return seg

    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0


class PerceptionPipeline:
    def __init__(
        self,
        modules: List[PerceptionModule],
        fusion: Optional[PerceptionFusion] = None,
        max_queue_size: int = 100
    ):
        self.modules = modules
        self.fusion = fusion or PerceptionFusion()
        self.input_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_id = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def submit(self, sensor_data: SensorData) -> bool:
        try:
            self.input_queue.put(sensor_data, timeout=0.1)
            return True
        except queue.Full:
            return False

    def get_result(self, timeout: float = 1.0) -> Optional[FusedPerceptionResult]:
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self):
        while self.running:
            try:
                sensor_data = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self.frame_id += 1
            sensor_data.frame_id = self.frame_id

            all_results = []
            for module in self.modules:
                if module.can_process(PerceptionTask.DETECTION):
                    results = module.process(sensor_data)
                    all_results.extend(results)

                    for r in results:
                        self.fusion.add_result(r)

            fused = self.fusion.fuse(sensor_data.timestamp, self.frame_id)
            if fused:
                try:
                    self.output_queue.put(fused, timeout=0.1)
                except queue.Full:
                    pass


class PerceptionVisualizer:
    @staticmethod
    def draw_detections(image: np.ndarray, detections: List[Dict], color_map: Dict[int, Tuple[int, int, int]] = None) -> np.ndarray:
        vis = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            conf = det['confidence']

            color = color_map.get(class_id, (0, 255, 0)) if color_map else (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{det['class_name']}: {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return vis

    @staticmethod
    def draw_segmentation(image: np.ndarray, segmentation: np.ndarray, alpha: float = 0.5, color_map: List[Tuple[int, int, int]] = None) -> np.ndarray:
        if color_map is None:
            color_map = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(256)]

        vis = image.copy()
        h, w = segmentation.shape

        for i in range(h):
            for j in range(w):
                class_id = segmentation[i, j]
                if class_id > 0 and class_id < len(color_map):
                    color = color_map[class_id]
                    vis[i, j] = (vis[i, j] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)

        return vis

    @staticmethod
    def draw_tracks(image: np.ndarray, tracks: List[Dict], color_map: Dict[int, Tuple[int, int, int]] = None) -> np.ndarray:
        vis = image.copy()

        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['track_id']

            color = color_map.get(track_id, (255, 0, 0)) if color_map else (255, 0, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"ID:{track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return vis

    @staticmethod
    def draw_depth(image: np.ndarray, depth: np.ndarray, colormap: int = cv2.COLORMAP_PLASMA) -> np.ndarray:
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(depth_norm, colormap)

        if len(image.shape) == 3:
            return cv2.addWeighted(image, 0.5, colored, 0.5, 0)
        return colored


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
