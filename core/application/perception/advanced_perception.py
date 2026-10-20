from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Dict, Any, List
from enum import Enum
from abc import ABC, abstractmethod
from collections import deque
import math


class PerceptionTask(Enum):
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    TRACKING = "tracking"
    POSE_ESTIMATION = "pose_estimation"
    DEPTH_ESTIMATION = "depth_estimation"
    OPTICAL_FLOW = "optical_flow"


@dataclass(frozen=True, slots=True)
class PerceptionResult:
    task: PerceptionTask
    timestamp: float
    data: Dict[str, Any]
    confidence: float
    processing_time_ms: float


class PerceptionModel(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> PerceptionResult:
        pass

    @abstractmethod
    def process_batch(self, images: List[np.ndarray]) -> List[PerceptionResult]:
        pass


class ObjectDetector(PerceptionModel):
    def __init__(
        self,
        model_path: str,
        classes: List[str],
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        input_size: tuple = (640, 640),
    ):
        self.model_path = model_path
        self.classes = classes
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self._model = None

    def load(self):
        try:
            import onnxruntime as ort
            self._model = ort.InferenceSession(self.model_path)
            return True
        except:
            return False

    def process(self, image: np.ndarray) -> PerceptionResult:
        start = time.time()

        if image.shape[:2] != self.input_size:
            image = cv2.resize(image, self.input_size)

        input_tensor = image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        outputs = self._model.run(None, {'images': input_tensor})
        detections = self._postprocess(outputs[0])

        return PerceptionResult(
            task=PerceptionTask.DETECTION,
            timestamp=time.time(),
            data={'detections': detections},
            confidence=np.mean([d['confidence'] for d in detections]) if detections else 0,
            processing_time_ms=(time.time() - start) * 1000,
        )

    def _postprocess(self, output: np.ndarray) -> List[Dict]:
        detections = []
        for det in output:
            if det[4] < self.conf_threshold:
                continue
            x1, y1, x2, y2, conf, cls = det[:6]
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(conf),
                'class_id': int(cls),
                'class_name': self.classes[int(cls)] if int(cls) < len(self.classes) else 'unknown',
            })
        return detections

    def process_batch(self, images: List[np.ndarray]) -> List[PerceptionResult]:
        return [self.process(img) for img in images]


class SegmentationModel(PerceptionModel):
    def __init__(
        self,
        model_path: str,
        num_classes: int,
        input_size: tuple = (512, 512),
    ):
        self.model_path = model_path
        self.num_classes = num_classes
        self.input_size = input_size
        self._model = None

    def load(self):
        try:
            import onnxruntime as ort
            self._model = ort.InferenceSession(self.model_path)
            return True
        except:
            return False

    def process(self, image: np.ndarray) -> PerceptionResult:
        start = time.time()

        if image.shape[:2] != self.input_size:
            image = cv2.resize(image, self.input_size)

        input_tensor = image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        outputs = self._model.run(None, {'input': input_tensor})
        mask = outputs[0].argmax(axis=1)[0]

        return PerceptionResult(
            task=PerceptionTask.SEGMENTATION,
            timestamp=time.time(),
            data={'mask': mask},
            confidence=1.0,
            processing_time_ms=(time.time() - start) * 1000,
        )

    def process_batch(self, images: List[np.ndarray]) -> List[PerceptionResult]:
        return [self.process(img) for img in images]


class PoseEstimator(PerceptionModel):
    def __init__(
        self,
        model_path: str,
        num_keypoints: int = 17,
        input_size: tuple = (256, 192),
    ):
        self.model_path = model_path
        self.num_keypoints = num_keypoints
        self.input_size = input_size
        self._model = None

    def load(self):
        try:
            import onnxruntime as ort
            self._model = ort.InferenceSession(self.model_path)
            return True
        except:
            return False

    def process(self, image: np.ndarray) -> PerceptionResult:
        start = time.time()

        if image.shape[:2] != self.input_size:
            image = cv2.resize(image, self.input_size)

        input_tensor = image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        outputs = self._model.run(None, {'input': input_tensor})
        keypoints = outputs[0][0]

        return PerceptionResult(
            task=PerceptionTask.POSE_ESTIMATION,
            timestamp=time.time(),
            data={'keypoints': keypoints.tolist()},
            confidence=float(keypoints[:, 2].mean()) if keypoints.shape[1] > 2 else 1.0,
            processing_time_ms=(time.time() - start) * 1000,
        )

    def process_batch(self, images: List[np.ndarray]) -> List[PerceptionResult]:
        return [self.process(img) for img in images]


class OpticalFlowEstimator(PerceptionModel):
    def __init__(
        self,
        method: str = "farneback",
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
    ):
        self.method = method
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.prev_gray = None

    def process(self, image: np.ndarray) -> PerceptionResult:
        start = time.time()

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return PerceptionResult(
                task=PerceptionTask.OPTICAL_FLOW,
                timestamp=time.time(),
                data={'flow': None},
                confidence=0.0,
                processing_time_ms=(time.time() - start) * 1000,
            )

        if self.method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                self.pyr_scale, self.levels, self.winsize,
                self.iterations, 5, 1.2, 0
            )
        elif self.method == "lucas_kanade":
            flow = self._lucas_kanade(self.prev_gray, gray)
        else:
            flow = np.zeros((gray.shape[0], gray.shape[1], 2))

        self.prev_gray = gray

        return PerceptionResult(
            task=PerceptionTask.OPTICAL_FLOW,
            timestamp=time.time(),
            data={'flow': flow},
            confidence=1.0,
            processing_time_ms=(time.time() - start) * 1000,
        )

    def _lucas_kanade(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        corners = cv2.goodFeaturesToTrack(prev, maxCorners=100, qualityLevel=0.3, minDistance=7)
        if corners is None:
            return np.zeros((prev.shape[0], prev.shape[1], 2))

        next_corners, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, corners, None)
        flow = np.zeros((prev.shape[0], prev.shape[1], 2))
        for i, (p, n, s) in enumerate(zip(corners, next_corners, status)):
            if s:
                x, y = int(p[0][0]), int(p[0][1])
                dx, dy = n[0] - p[0]
                if 0 <= x < flow.shape[1] and 0 <= y < flow.shape[0]:
                    flow[y, x] = [dx[0], dy[0]]
        return flow

    def process_batch(self, images: List[np.ndarray]) -> List[PerceptionResult]:
        return [self.process(img) for img in images]


class DepthEstimator(PerceptionModel):
    def __init__(
        self,
        model_path: str,
        input_size: tuple = (384, 384),
    ):
        self.model_path = model_path
        self.input_size = input_size
        self._model = None

    def load(self):
        try:
            import onnxruntime as ort
            self._model = ort.InferenceSession(self.model_path)
            return True
        except:
            return False

    def process(self, image: np.ndarray) -> PerceptionResult:
        start = time.time()

        if image.shape[:2] != self.input_size:
            image = cv2.resize(image, self.input_size)

        input_tensor = image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        outputs = self._model.run(None, {'input': input_tensor})
        depth = outputs[0][0, 0]

        return PerceptionResult(
            task=PerceptionTask.DEPTH_ESTIMATION,
            timestamp=time.time(),
            data={'depth': depth},
            confidence=1.0,
            processing_time_ms=(time.time() - start) * 1000,
        )

    def process_batch(self, images: List[np.ndarray]) -> List[PerceptionResult]:
        return [self.process(img) for img in images]


class PerceptionPipeline:
    def __init__(self):
        self.models: Dict[PerceptionTask, PerceptionModel] = {}
        self.preprocessors: List[Callable] = []
        self.postprocessors: List[Callable] = []

    def add_model(self, task: PerceptionTask, model: PerceptionModel):
        self.models[task] = model

    def add_preprocessor(self, preprocessor: Callable):
        self.preprocessors.append(preprocessor)

    def add_postprocessor(self, postprocessor: Callable):
        self.postprocessors.append(postprocessor)

    def process(self, image: np.ndarray, tasks: List[PerceptionTask] = None) -> Dict[PerceptionTask, PerceptionResult]:
        tasks = tasks or list(self.models.keys())
        results = {}

        for preprocessor in self.preprocessors:
            image = preprocessor(image)

        for task in tasks:
            if task in self.models:
                result = self.models[task].process(image)
                results[task] = result

        for postprocessor in self.postprocessors:
            results = postprocessor(results)

        return results

    def process_batch(self, images: List[np.ndarray], tasks: List[PerceptionTask] = None) -> List[Dict[PerceptionTask, PerceptionResult]]:
        return [self.process(img, tasks) for img in images]


class MultiTaskPerception:
    def __init__(
        self,
        shared_backbone: PerceptionModel,
        task_heads: Dict[PerceptionTask, PerceptionModel],
    ):
        self.shared_backbone = shared_backbone
        self.task_heads = task_heads

    def process(self, image: np.ndarray) -> Dict[PerceptionTask, PerceptionResult]:
        backbone_result = self.shared_backbone.process(image)

        results = {}
        for task, head in self.task_heads.items():
            result = head.process(image)
            results[task] = result

        return results


class PerceptionEvaluator:
    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)

    def evaluate_detection(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        iou_threshold: float = 0.5,
    ) -> Dict[str, float]:
        tp = fp = fn = 0
        matched_gt = set()

        for pred in predictions:
            best_iou = 0
            best_gt_idx = -1
            for i, gt in enumerate(ground_truth):
                if i in matched_gt:
                    continue
                iou = self._compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        fn = len(ground_truth) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
        }

    def evaluate_segmentation(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
    ) -> Dict[str, float]:
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = intersection / union if union > 0 else 0

        dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 0

        return {
            'iou': iou,
            'dice': dice,
        }

    def evaluate_pose(
        self,
        pred_keypoints: np.ndarray,
        gt_keypoints: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        if pred_keypoints.shape != gt_keypoints.shape:
            return {'pck': 0.0}

        distances = np.linalg.norm(pred_keypoints[:, :2] - gt_keypoints[:, :2], axis=1)
        visible = gt_keypoints[:, 2] > 0 if gt_keypoints.shape[1] > 2 else np.ones(len(gt_keypoints), dtype=bool)

        if not visible.any():
            return {'pck': 0.0}

        correct = (distances[visible] < threshold).sum()
        pck = correct / visible.sum()

        return {
            'pck': pck,
            'mean_error': distances[visible].mean() if visible.sum() > 0 else 0,
        }

    def _compute_iou(self, box1: List, box2: List) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0


from collections import defaultdict
import time
