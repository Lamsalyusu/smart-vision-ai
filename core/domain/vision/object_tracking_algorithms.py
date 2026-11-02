from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math


class TrackingAlgorithm(Enum):
    KCF = "kcf"
    MOSSE = "mosse"
    CSRT = "csrt"
    TLD = "tld"
    MIL = "mil"
    BOOSTING = "boosting"
    MEDIANFLOW = "medianflow"
    GOTURN = "goturn"
    DASIAMRPN = "dasiamrpn"
    SIAMRPN = "siamrpn"
    OCEAN = "ocean"
    TRANS_T = "trans_t"
    MIX_FORMER = "mixformer"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> "BoundingBox":
        return cls(x1, y1, x2 - x1, y2 - y1)

    def iou(self, other: "BoundingBox") -> float:
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass(frozen=True, slots=True)
class TrackedObject:
    track_id: str
    bbox: BoundingBox
    confidence: float
    features: Optional[np.ndarray] = None
    age: int = 0
    hits: int = 0
    hit_streak: int = 0
    time_since_update: int = 0
    state: str = "active"


class BaseTracker(ABC):
    @abstractmethod
    def init(self, image: np.ndarray, bbox: BoundingBox) -> bool:
        pass

    @abstractmethod
    def update(self, image: np.ndarray) -> Optional[BoundingBox]:
        pass

    @abstractmethod
    def reset(self):
        pass


class KCFTracker(BaseTracker):
    def __init__(
        self,
        padding: float = 1.5,
        output_sigma_factor: float = 0.125,
        lambda_: float = 1e-4,
        interp_factor: float = 0.075,
        kernel_sigma: float = 0.5,
        cell_size: int = 4,
        template_size: int = 96,
    ):
        self.padding = padding
        self.output_sigma_factor = output_sigma_factor
        self.lambda_ = lambda_
        self.interp_factor = interp_factor
        self.kernel_sigma = kernel_sigma
        self.cell_size = cell_size
        self.template_size = template_size

        self._roi = None
        self._alphaf = None
        self._prob = None
        self._window = None
        self._hog_features = None
        self._initialized = False

    def init(self, image: np.ndarray, bbox: BoundingBox) -> bool:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        self._roi = (int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height))
        x, y, w, h = self._roi

        self._window = self._create_hanning_window(w, h)
        self._roi_size = (w, h)

        patch = image[y:y+h, x:x+w]
        if patch.size == 0:
            return False

        self._alphaf = self._train(patch)
        self._initialized = True
        return True

    def _create_hanning_window(self, width: int, height: int) -> np.ndarray:
        win_h = np.hanning(height)
        win_w = np.hanning(width)
        window = np.outer(win_h, win_w)
        return window / np.sum(window)

    def _get_hog_features(self, patch: np.ndarray) -> np.ndarray:
        win_size = (patch.shape[1] // self.cell_size * self.cell_size,
                    patch.shape[0] // self.cell_size * self.cell_size)
        patch = cv2.resize(patch, win_size)

        hog = cv2.HOGDescriptor(
            _winSize=win_size,
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(self.cell_size, self.cell_size),
            _nbins=9,
        )
        return hog.compute(patch).flatten()

    def _train(self, patch: np.ndarray) -> np.ndarray:
        features = self._get_hog_features(patch)
        if features.size == 0:
            return np.array([])

        target = self._create_gaussian_peak(features.shape[0])
        alphaf = self._complex_division(target, features, self.lambda_)
        return alphaf

    def _create_gaussian_peak(self, size: int) -> np.ndarray:
        output_sigma = math.sqrt(self.roi_size[0] * self.roi_size[1]) * self.output_sigma_factor
        target = np.zeros(size)
        for i in range(size):
            x = i % int(math.sqrt(size))
            y = i // int(math.sqrt(size))
            target[i] = math.exp(-0.5 * ((x**2 + y**2) / (output_sigma**2)))
        return target

    def _complex_division(self, target: np.ndarray, features: np.ndarray, lambda_: float) -> np.ndarray:
        return target / (features + lambda_)

    def update(self, image: np.ndarray) -> Optional[BoundingBox]:
        if not self._initialized:
            return None

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        x, y, w, h = self._roi
        patch = image[y:y+h, x:x+w]
        if patch.size == 0:
            return None

        features = self._get_hog_features(patch)
        if features.size == 0:
            return None

        response = self._detect(features)
        peak_idx = np.argmax(response)
        dy, dx = self._peak_to_shift(peak_idx, response.shape)

        new_x = x + dx * self.cell_size
        new_y = y + dy * self.cell_size

        new_x = max(0, min(new_x, image.shape[1] - w))
        new_y = max(0, min(new_y, image.shape[0] - h))

        self._roi = (int(new_x), int(new_y), w, h)
        new_patch = image[int(new_y):int(new_y)+h, int(new_x):int(new_x)+w]
        if new_patch.size > 0:
            new_alphaf = self._train(new_patch)
            self._alphaf = (1 - self.interp_factor) * self._alphaf + self.interp_factor * new_alphaf

        return BoundingBox(new_x, new_y, w, h)

    def _detect(self, features: np.ndarray) -> np.ndarray:
        return self._alphaf * features

    def _peak_to_shift(self, peak_idx: int, shape: Tuple) -> Tuple[int, int]:
        return (peak_idx // shape[1], peak_idx % shape[1])

    def reset(self):
        self._initialized = False
        self._alphaf = None


class MOSSETracker(BaseTracker):
    def __init__(self, lr: float = 0.125, sigma: float = 100.0):
        self.lr = lr
        self.sigma = sigma
        self._filter = None
        self._roi = None
        self._initialized = False

    def init(self, image: np.ndarray, bbox: BoundingBox) -> bool:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        self._roi = (int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height))
        x, y, w, h = self._roi

        patch = image[y:y+h, x:x+w]
        patch = cv2.resize(patch, (w, h)).astype(np.float32)

        self._filter = self._train_mosse(patch)
        self._initialized = True
        return True

    def _train_mosse(self, patch: np.ndarray) -> np.ndarray:
        patch = (patch - np.mean(patch)) / (np.std(patch) + 1e-5)
        G = self._gaussian_peak(patch.shape)
        F = np.fft.fft2(patch)
        G_fft = np.fft.fft2(G)
        return G_fft / (F + 1e-5)

    def _gaussian_peak(self, shape: Tuple) -> np.ndarray:
        h, w = shape
        x = np.arange(w) - w // 2
        y = np.arange(h) - h // 2
        X, Y = np.meshgrid(x, y)
        G = np.exp(-(X**2 + Y**2) / (2 * self.sigma**2))
        return G / np.max(G)

    def update(self, image: np.ndarray) -> Optional[BoundingBox]:
        if not self._initialized:
            return None

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        x, y, w, h = self._roi
        patch = image[y:y+h, x:x+w]
        patch = cv2.resize(patch, (w, h)).astype(np.float32)
        patch = (patch - np.mean(patch)) / (np.std(patch) + 1e-5)

        F = np.fft.fft2(patch)
        response = np.fft.ifft2(self._filter * F).real

        peak = np.unravel_index(np.argmax(response), response.shape)
        dy = peak[0] - w // 2
        dx = peak[1] - h // 2

        new_x = x + dx
        new_y = y + dy

        new_x = max(0, min(new_x, image.shape[1] - w))
        new_y = max(0, min(new_y, image.shape[0] - h))

        self._roi = (int(new_x), int(new_y), w, h)

        new_patch = image[int(new_y):int(new_y)+h, int(new_x):int(new_x)+w]
        new_patch = cv2.resize(new_patch, (w, h)).astype(np.float32)
        new_patch = (new_patch - np.mean(new_patch)) / (np.std(new_patch) + 1e-5)
        new_F = np.fft.fft2(new_patch)
        G = self._gaussian_peak((w, h))
        G_fft = np.fft.fft2(G)
        new_filter = G_fft / (new_F + 1e-5)

        self._filter = (1 - 0.125) * self._filter + 0.125 * new_filter

        return BoundingBox(new_x, new_y, w, h)

    def reset(self):
        self._initialized = False
        self._filter = None


class CSRTTracker(BaseTracker):
    def __init__(
        self,
        padding: float = 3.0,
        output_sigma_factor: float = 0.125,
        lambda_: float = 1e-4,
        interp_factor: float = 0.075,
        kernel_sigma: float = 0.5,
        cell_size: int = 4,
    ):
        self.padding = padding
        self.output_sigma_factor = output_sigma_factor
        self.lambda_ = lambda_
        self.interp_factor = interp_factor
        self.kernel_sigma = kernel_sigma
        self.cell_size = cell_size

        self._roi = None
        self._alphaf = None
        self._window = None
        self._initialized = False

    def init(self, image: np.ndarray, bbox: BoundingBox) -> bool:
        self._roi = (int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height))
        x, y, w, h = self._roi

        self._window = self._create_hann(w, h)
        patch = image[y:y+h, x:x+w]
        if patch.size == 0:
            return False

        self._alphaf = self._train_csrt(patch)
        self._initialized = True
        return True

    def _create_hann(self, w: int, h: int) -> np.ndarray:
        win_h = np.hanning(h)
        win_w = np.hanning(w)
        return np.outer(win_h, win_w).flatten()

    def _train_csrt(self, patch: np.ndarray) -> np.ndarray:
        features = self._extract_hog(patch)
        if features.size == 0:
            return np.array([])

        target = self._create_gaussian_label(features.shape[0])
        return self._ridge_regression(features, target)

    def _extract_hog(self, patch: np.ndarray) -> np.ndarray:
        win_size = (patch.shape[1] // self.cell_size * self.cell_size,
                    patch.shape[0] // self.cell_size * self.cell_size)
        patch = cv2.resize(patch, win_size)

        hog = cv2.HOGDescriptor(
            _winSize=win_size,
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(self.cell_size, self.cell_size),
            _nbins=9,
        )
        return hog.compute(patch).flatten()

    def _ridge_regression(self, features: np.ndarray, target: np.ndarray, lambda_: float = 1e-4) -> np.ndarray:
        return np.fft.fft(target) / (np.fft.fft(features) + lambda_)

    def update(self, image: np.ndarray) -> Optional[BoundingBox]:
        if not self._initialized:
            return None

        x, y, w, h = self._roi
        patch = image[y:y+h, x:x+w]
        if patch.size == 0:
            return None

        features = self._extract_hog(patch)
        if features.size == 0:
            return None

        response = np.fft.ifft(np.fft.fft(self._alphaf) * np.fft.fft(features)).real
        peak_idx = np.argmax(response)
        dy = peak_idx // int(math.sqrt(features.size))
        dx = peak_idx % int(math.sqrt(features.size))

        new_x = x + dx * self.cell_size
        new_y = y + dy * self.cell_size

        self._roi = (int(new_x), int(new_y), w, h)
        new_patch = image[int(new_y):int(new_y)+h, int(new_x):int(new_x)+w]
        if new_patch.size > 0:
            new_alphaf = self._train_csrt(new_patch)
            self._alphaf = 0.9 * self._alphaf + 0.1 * new_alphaf

        return BoundingBox(new_x, new_y, w, h)

    def reset(self):
        self._initialized = False
        self._alphaf = None


class DeepLearningTracker(BaseTracker):
    def __init__(
        self,
        model_path: str,
        backbone: str = "resnet50",
        confidence_threshold: float = 0.5,
    ):
        self.model_path = model_path
        self.backbone = backbone
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            self._model = torch.jit.load(self.model_path)
            self._model.eval()
            self._initialized = True
        except:
            self._initialized = False

    def init(self, image: np.ndarray, bbox: BoundingBox) -> bool:
        if not self._initialized:
            self._initialize()
        if not self._initialized:
            return False
        return True

    def update(self, image: np.ndarray) -> Optional[BoundingBox]:
        if not self._initialized:
            return None
        return None

    def reset(self):
        self._initialized = False


class SiameseTracker(BaseTracker):
    def __init__(
        self,
        model_path: str,
        exemplar_size: int = 127,
        instance_size: int = 255,
        context_amount: float = 0.5,
    ):
        self.model_path = model_path
        self.exemplar_size = exemplar_size
        self.instance_size = instance_size
        self.context_amount = context_amount
        self._model = None
        self._template = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            self._model = torch.jit.load(self.model_path)
            self._model.eval()
            self._initialized = True
        except:
            self._initialized = False

    def init(self, image: np.ndarray, bbox: BoundingBox) -> bool:
        if not self._initialized:
            self._initialize()
        if not self._initialized:
            return False

        self._template = self._extract_template(image, bbox)
        return True

    def _extract_template(self, image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        return image[int(bbox.y):int(bbox.y+bbox.height), int(bbox.x):int(bbox.x+bbox.width)]

    def update(self, image: np.ndarray) -> Optional[BoundingBox]:
        if not self._initialized:
            return None
        return None

    def reset(self):
        self._initialized = False
        self._template = None


class MultiObjectTracker:
    def __init__(
        self,
        tracker_type: TrackingAlgorithm = TrackingAlgorithm.KCF,
        max_objects: int = 100,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
    ):
        self.tracker_type = tracker_type
        self.max_objects = max_objects
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        self.tracks: Dict[str, TrackedObject] = {}
        self.next_id = 0

    def _create_tracker(self) -> BaseTracker:
        if self.tracker_type == TrackingAlgorithm.KCF:
            return KCFTracker()
        elif self.tracker_type == TrackingAlgorithm.MOSSE:
            return MOSSETracker()
        elif self.tracker_type == TrackingAlgorithm.CSRT:
            return CSRTTracker()
        return KCFTracker()

    def update(self, detections: List[BoundingBox], image: np.ndarray) -> List[TrackedObject]:
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            track = TrackedObject(
                track_id=track.track_id,
                bbox=track.bbox,
                confidence=track.confidence,
                features=track.features,
                age=track.age + 1,
                hits=track.hits,
                hit_streak=0,
                time_since_update=track.time_since_update + 1,
                state="lost" if track.time_since_update > 0 else "active",
            )
            self.tracks[track_id] = track

        if not detections:
            for track_id in list(self.tracks.keys()):
                if self.tracks[track_id].time_since_update > self.max_age:
                    del self.tracks[track_id]
            return [t for t in self.tracks.values() if t.state == "active"]

        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid].bbox for tid in track_ids]

        iou_matrix = self._iou_matrix(track_bboxes, detections)
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

            track = TrackedObject(
                track_id=track.track_id,
                bbox=det,
                confidence=det.confidence if hasattr(det, 'confidence') else 1.0,
                features=track.features,
                age=track.age + 1,
                hits=track.hits + 1,
                hit_streak=track.hit_streak + 1,
                time_since_update=0,
                state="active",
            )
            self.tracks[track_id] = track

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            if len(self.tracks) >= self.max_objects:
                continue

            track_id = f"track_{self.next_id}"
            self.next_id += 1

            track = TrackedObject(
                track_id=track_id,
                bbox=det,
                confidence=det.confidence if hasattr(det, 'confidence') else 1.0,
                age=1,
                hits=1,
                hit_streak=1,
                time_since_update=0,
                state="active",
            )
            self.tracks[track_id] = track

        for track_idx in unmatched_tracks:
            track_id = track_ids[track_idx]
            track = self.tracks[track_id]
            if track.time_since_update > self.max_age:
                del self.tracks[track_id]

        return [t for t in self.tracks.values() if t.hits >= self.min_hits or t.time_since_update == 0]

    def _hungarian(self, cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return list(zip(row_ind, col_ind))

    def _iou_matrix(self, boxes1: List[BoundingBox], boxes2: List[BoundingBox]) -> np.ndarray:
        n1, n2 = len(boxes1), len(boxes2)
        iou = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                iou[i, j] = boxes1[i].iou(boxes2[j])
        return iou


class TrackerEvaluator:
    @staticmethod
    def compute_mota(tracks: List[TrackedObject], gt_tracks: List[TrackedObject]) -> float:
        return 0.0

    @staticmethod
    def compute_motp(tracks: List[TrackedObject], gt_tracks: List[TrackedObject]) -> float:
        return 0.0

    @staticmethod
    def compute_idf1(tracks: List[TrackedObject], gt_tracks: List[TrackedObject]) -> float:
        return 0.0

    @staticmethod
    def compute_tracklet_fragmentation(tracks: List[TrackedObject]) -> float:
        return 0.0


class TrackVisualizer:
    @staticmethod
    def draw_tracks(image: np.ndarray, tracks: List[TrackedObject]) -> np.ndarray:
        vis = image.copy()
        for track in tracks:
            if track.state != "active":
                continue
            x, y, w, h = track.bbox.x, track.bbox.y, track.bbox.width, track.bbox.height
            cv2.rectangle(vis, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
            cv2.putText(vis, f"ID: {track.track_id}", (int(x), int(y)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return vis

    @staticmethod
    def draw_trajectories(image: np.ndarray, tracks: List[TrackedObject], history_length: int = 30) -> np.ndarray:
        return image