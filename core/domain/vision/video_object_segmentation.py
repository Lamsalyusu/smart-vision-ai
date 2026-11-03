from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math
import time


class VOSAlgorithm(Enum):
    OSVOS = "osvos"
    MASKTRACK = "masktrack"
    STCN = "stcn"
    XMEM = "xmem"
    CFBI = "cfbi"
    AOT = "aot"
    DEAOT = "deaot"
    SIMPLEVOS = "simplevos"


@dataclass(frozen=True, slots=True)
class VOSFrame:
    frame_id: int
    image: np.ndarray
    mask: Optional[np.ndarray] = None
    objects: Dict[int, np.ndarray] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass(frozen=True, slots=True)
class VOSResult:
    frame_id: int
    masks: Dict[int, np.ndarray]
    scores: Dict[int, float]
    timestamp: float


class BaseVOS(ABC):
    @abstractmethod
    def initialize(self, frame: np.ndarray, mask: np.ndarray) -> bool:
        pass

    @abstractmethod
    def track(self, frame: np.ndarray) -> Dict[int, np.ndarray]:
        pass

    @abstractmethod
    def add_object(self, frame: np.ndarray, mask: np.ndarray, obj_id: int) -> bool:
        pass

    @abstractmethod
    def remove_object(self, obj_id: int) -> bool:
        pass


class OSVOS(BaseVOS):
    def __init__(
        self,
        model_path: str = "osvos.pth",
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._initialized = False
        self.objects: Dict[int, np.ndarray] = {}

    def _initialize(self):
        try:
            import torch
            self._model = torch.jit.load(self.model_path)
            self._model.eval()
            self._initialized = True
        except ImportError:
            self._initialized = False

    def initialize(self, frame: np.ndarray, mask: np.ndarray) -> bool:
        if not self._initialized:
            self._initialize()

        if not self._initialized:
            return False

        obj_id = len(self.objects)
        self.objects[obj_id] = mask.copy()
        return True

    def track(self, frame: np.ndarray) -> Dict[int, np.ndarray]:
        if not self._initialized:
            return {}

        results = {}
        for obj_id, mask in self.objects.items():
            new_mask = self._propagate_mask(frame, mask)
            results[obj_id] = new_mask
            self.objects[obj_id] = new_mask

        return results

    def _propagate_mask(self, frame: np.ndarray, prev_mask: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(self.prev_frame, cv2.COLOR_RGB2GRAY) if hasattr(self, 'prev_frame') else gray,
            gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        h, w = prev_mask.shape
        new_mask = np.zeros_like(prev_mask)

        for y in range(h):
            for x in range(w):
                if prev_mask[y, x] > 0:
                    flow_vec = flow[y, x]
                    new_y = int(y + flow_vec[1])
                    new_x = int(x + flow_vec[0])
                    if 0 <= new_y < h and 0 <= new_x < w:
                        new_mask[new_y, new_x] = prev_mask[y, x]

        self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
        return new_mask

    def add_object(self, frame: np.ndarray, mask: np.ndarray, obj_id: int) -> bool:
        self.objects[obj_id] = mask.copy()
        return True

    def remove_object(self, obj_id: int) -> bool:
        if obj_id in self.objects:
            del self.objects[obj_id]
            return True
        return False


class MaskTrack(BaseVOS):
    def __init__(
        self,
        model_path: str = "masktrack.pth",
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._initialized = False
        self.objects: Dict[int, Dict] = {}

    def _initialize(self):
        try:
            import torch
            self._model = torch.jit.load(self.model_path)
            self._model.eval()
            self._initialized = True
        except ImportError:
            self._initialized = False

    def initialize(self, frame: np.ndarray, mask: np.ndarray) -> bool:
        if not self._initialized:
            self._initialize()

        if not self._initialized:
            return False

        obj_id = len(self.objects)
        self.objects[obj_id] = {
            'mask': mask.copy(),
            'template': self._extract_template(frame, mask),
            'frame_count': 0,
        }
        return True

    def _extract_template(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        template = frame[y:y+h, x:x+w]
        mask_crop = mask[y:y+h, x:x+w]
        return template * mask_crop[:, :, np.newaxis] if len(template.shape) == 3 else template * mask_crop

    def track(self, frame: np.ndarray) -> Dict[int, np.ndarray]:
        if not self._initialized:
            return {}

        results = {}
        for obj_id, obj_data in self.objects.items():
            template = obj_data['template']
            mask = self._match_template(frame, template)
            self.objects[obj_id]['mask'] = mask
            self.objects[obj_id]['frame_count'] += 1

            if self.objects[obj_id]['frame_count'] % 10 == 0:
                self.objects[obj_id]['template'] = self._extract_template(frame, mask)

            results[obj_id] = mask

        return results

    def _match_template(self, frame: np.ndarray, template: np.ndarray) -> np.ndarray:
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > 0.5:
            h, w = template.shape[:2]
            x, y = max_loc
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            mask[y:y+h, x:x+w] = 255
            return mask
        else:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

    def add_object(self, frame: np.ndarray, mask: np.ndarray, obj_id: int) -> bool:
        self.objects[obj_id] = {
            'mask': mask.copy(),
            'template': self._extract_template(frame, mask),
            'frame_count': 0,
        }
        return True

    def remove_object(self, obj_id: int) -> bool:
        if obj_id in self.objects:
            del self.objects[obj_id]
            return True
        return False


class STCN(BaseVOS):
    def __init__(
        self,
        model_path: str = "stcn.pth",
        device: str = "cpu",
        memory_size: int = 10,
    ):
        self.model_path = model_path
        self.device = device
        self.memory_size = memory_size
        self._model = None
        self._initialized = False
        self.memory: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}

    def _initialize(self):
        try:
            import torch
            self._initialized = True
        except ImportError:
            self._initialized = False

    def initialize(self, frame: np.ndarray, mask: np.ndarray) -> bool:
        if not self._initialized:
            self._initialize()

        if not self._initialized:
            return False

        obj_id = 0
        self.memory[obj_id] = [(frame.copy(), mask.copy())]
        return True

    def track(self, frame: np.ndarray) -> Dict[int, np.ndarray]:
        if not self._initialized:
            return {}

        results = {}
        for obj_id, memory in self.memory.items():
            mask = self._read_memory(frame, memory)
            self.memory[obj_id].append((frame.copy(), mask.copy()))
            if len(self.memory[obj_id]) > self.memory_size:
                self.memory[obj_id].pop(0)
            results[obj_id] = mask

        return results

    def _read_memory(self, frame: np.ndarray, memory: List) -> np.ndarray:
        if not memory:
            return np.zeros((480, 640), dtype=np.uint8)

        ref_frame, ref_mask = memory[-1]
        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(ref_frame, cv2.COLOR_RGB2GRAY) if len(ref_frame.shape) == 3 else ref_frame,
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        mask = memory[-1][1] if isinstance(memory[-1], tuple) else memory[-1]
        h, w = mask.shape
        new_mask = np.zeros_like(mask)

        for y in range(h):
            for x in range(w):
                if memory[-1][1][y, x] > 0:
                    flow_vec = flow[y, x]
                    new_y = int(y + flow_vec[1])
                    new_x = int(x + flow_vec[0])
                    if 0 <= new_y < h and 0 <= new_x < w:
                        new_mask[new_y, new_x] = mask[y, x]

        return new_mask

    def add_object(self, frame: np.ndarray, mask: np.ndarray, obj_id: int) -> bool:
        if obj_id not in self.memory:
            self.memory[obj_id] = []
        self.memory[obj_id].append((frame.copy(), mask.copy()))
        return True

    def remove_object(self, obj_id: int) -> bool:
        if obj_id in self.memory:
            del self.memory[obj_id]
            return True
        return False


class XMem(BaseVOS):
    def __init__(
        self,
        model_path: str = "xmem.pth",
        device: str = "cpu",
        top_k: int = 30,
    ):
        self.model_path = model_path
        self.device = device
        self.top_k = top_k
        self._model = None
        self._initialized = False
        self.memory: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

    def _initialize(self):
        try:
            import torch
            self._initialized = True
        except ImportError:
            self._initialized = False

    def initialize(self, frame: np.ndarray, mask: np.ndarray) -> bool:
        if not self._initialized:
            self._initialize()

        if not self._initialized:
            return False

        obj_id = 0
        self.memory[obj_id] = [(frame.copy(), mask.copy(), self._encode_mask(mask))]
        return True

    def _encode_mask(self, mask: np.ndarray) -> np.ndarray:
        return mask.astype(np.float32)

    def track(self, frame: np.ndarray) -> Dict[int, np.ndarray]:
        if not self._initialized:
            return {}

        results = {}
        for obj_id, memory in self.memory.items():
            mask = self._attend_memory(frame, memory)
            self.memory[obj_id].append((frame.copy(), mask.copy(), self._encode_mask(mask)))
            if len(self.memory[obj_id]) > self.top_k:
                self.memory[obj_id].pop(0)
            results[obj_id] = mask

        return results

    def _attend_memory(self, frame: np.ndarray, memory: List) -> np.ndarray:
        if not memory:
            return np.zeros((480, 640), dtype=np.uint8)

        query_frame = frame
        best_mask = memory[-1][1]

        scores = []
        for ref_frame, ref_mask, ref_feat in memory:
            sim = self._compute_similarity(frame, ref_frame)
            scores.append(sim)

        if scores:
            best_idx = np.argmax(scores)
            best_mask = memory[best_idx][1]

        return best_mask

    def _compute_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        h1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
        h2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
        return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)

    def add_object(self, frame: np.ndarray, mask: np.ndarray, obj_id: int) -> bool:
        if obj_id not in self.memory:
            self.memory[obj_id] = []
        self.memory[obj_id].append((frame.copy(), mask.copy(), self._encode_mask(mask)))
        return True

    def remove_object(self, obj_id: int) -> bool:
        if obj_id in self.memory:
            del self.memory[obj_id]
            return True
        return False


class AOT(BaseVOS):
    def __init__(
        self,
        model_path: str = "aot.pth",
        device: str = "cpu",
        max_obj: int = 10,
    ):
        self.model_path = model_path
        self.device = device
        self.max_obj = max_obj
        self._model = None
        self._initialized = False
        self.objects: Dict[int, Dict] = {}

    def _initialize(self):
        try:
            import torch
            self._initialized = True
        except ImportError:
            self._initialized = False

    def initialize(self, frame: np.ndarray, mask: np.ndarray) -> bool:
        if not self._initialized:
            self._initialize()

        if not self._initialized:
            return False

        obj_id = len(self.objects)
        if obj_id >= self.max_obj:
            return False

        self.objects[obj_id] = {
            'mask': mask.copy(),
            'features': self._extract_features(frame, mask),
        }
        return True

    def _extract_features(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.random.randn(256)

    def track(self, frame: np.ndarray) -> Dict[int, np.ndarray]:
        if not self._initialized:
            return {}

        results = {}
        for obj_id, obj_data in self.objects.items():
            mask = self._match(frame, obj_data['features'])
            obj_data['mask'] = mask
            obj_data['features'] = self._extract_features(frame, mask)
            results[obj_id] = mask

        return results

    def _match(self, frame: np.ndarray, features: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

    def add_object(self, frame: np.ndarray, mask: np.ndarray, obj_id: int) -> bool:
        if obj_id in self.objects or len(self.objects) >= self.max_obj:
            return False

        self.objects[obj_id] = {
            'mask': mask.copy(),
            'features': self._extract_features(frame, mask),
        }
        return True

    def remove_object(self, obj_id: int) -> bool:
        if obj_id in self.objects:
            del self.objects[obj_id]
            return True
        return False


class CFBI(BaseVOS):
    def __init__(
        self,
        model_path: str = "cfbi.pth",
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._initialized = False
        self.prev_frame = None
        self.prev_mask = None

    def _initialize(self):
        try:
            import torch
            self._initialized = True
        except ImportError:
            self._initialized = False

    def initialize(self, frame: np.ndarray, mask: np.ndarray) -> bool:
        if not self._initialized:
            self._initialize()

        if not self._initialized:
            return False

        self.prev_frame = frame.copy()
        self.prev_mask = mask.copy()
        return True

    def track(self, frame: np.ndarray) -> Dict[int, np.ndarray]:
        if not self._initialized or self.prev_frame is None:
            return {}

        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(self.prev_frame, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        mask = self._warp_mask(self.prev_mask, flow)

        self.prev_frame = frame.copy()
        self.prev_mask = mask

        return {0: mask}

    def _warp_mask(self, mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        new_mask = np.zeros_like(mask)

        for y in range(h):
            for x in range(w):
                if mask[y, x] > 0:
                    flow_vec = flow[y, x]
                    new_y = int(y + flow_vec[1])
                    new_x = int(x + flow_vec[0])
                    if 0 <= new_y < h and 0 <= new_x < w:
                        new_mask[new_y, new_x] = mask[y, x]

        return new_mask

    def add_object(self, frame: np.ndarray, mask: np.ndarray, obj_id: int) -> bool:
        self.prev_frame = frame.copy()
        self.prev_mask = mask.copy()
        return True

    def remove_object(self, obj_id: int) -> bool:
        return True


class SimpleVOS(BaseVOS):
    def __init__(self, tracker_type: str = "csrt"):
        self.tracker_type = tracker_type
        self.trackers: Dict[int, Any] = {}
        self.masks: Dict[int, np.ndarray] = {}

    def initialize(self, frame: np.ndarray, mask: np.ndarray) -> bool:
        obj_id = len(self.trackers)
        tracker = self._create_tracker()
        bbox = cv2.boundingRect(mask.astype(np.uint8))
        tracker.init(frame, bbox)
        self.trackers[obj_id] = tracker
        self.masks[obj_id] = mask.copy()
        return True

    def _create_tracker(self):
        if self.tracker_type == "csrt":
            return cv2.TrackerCSRT_create()
        elif self.tracker_type == "kcf":
            return cv2.TrackerKCF_create()
        elif self.tracker_type == "mosse":
            return cv2.TrackerMOSSE_create()
        return cv2.TrackerCSRT_create()

    def track(self, frame: np.ndarray) -> Dict[int, np.ndarray]:
        results = {}
        for obj_id, tracker in self.trackers.items():
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = map(int, bbox)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                mask[y:y+h, x:x+w] = 255
                results[obj_id] = mask
            else:
                results[obj_id] = self.masks.get(obj_id, np.zeros(frame.shape[:2], dtype=np.uint8))
        return results

    def add_object(self, frame: np.ndarray, mask: np.ndarray, obj_id: int) -> bool:
        tracker = self._create_tracker()
        bbox = cv2.boundingRect(mask.astype(np.uint8))
        tracker.init(frame, bbox)
        self.trackers[obj_id] = tracker
        self.masks[obj_id] = mask.copy()
        return True

    def remove_object(self, obj_id: int) -> bool:
        if obj_id in self.trackers:
            del self.trackers[obj_id]
            del self.masks[obj_id]
            return True
        return False


class VOSManager:
    def __init__(
        self,
        algorithm: VOSAlgorithm = VOSAlgorithm.XMEM,
        **kwargs,
    ):
        self.algorithm = algorithm
        self.vos = self._create_vos(algorithm, **kwargs)

    def _create_vos(self, algorithm: VOSAlgorithm, **kwargs) -> BaseVOS:
        if algorithm == VOSAlgorithm.OSVOS:
            return OSVOS(**kwargs)
        elif algorithm == VOSAlgorithm.MASKTRACK:
            return MaskTrack(**kwargs)
        elif algorithm == VOSAlgorithm.STCN:
            return STCN(**kwargs)
        elif algorithm == VOSAlgorithm.XMEM:
            return XMem(**kwargs)
        elif algorithm == VOSAlgorithm.AOT:
            return AOT(**kwargs)
        elif algorithm == VOSAlgorithm.CFBI:
            return CFBI(**kwargs)
        elif algorithm == VOSAlgorithm.SIMPLEVOS:
            return SimpleVOS(**kwargs)
        return XMem()

    def initialize(self, frame: np.ndarray, mask: np.ndarray) -> bool:
        return self.vos.initialize(frame, mask)

    def track(self, frame: np.ndarray) -> Dict[int, np.ndarray]:
        return self.vos.track(frame)

    def add_object(self, frame: np.ndarray, mask: np.ndarray, obj_id: int) -> bool:
        return self.vos.add_object(frame, mask, obj_id)

    def remove_object(self, obj_id: int) -> bool:
        return self.vos.remove_object(obj_id)


class VOSEvaluator:
    @staticmethod
    def compute_jaccard(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def compute_boundary_f(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        pred_boundary = cv2.Canny(pred_mask.astype(np.uint8) * 255, 100, 200)
        gt_boundary = cv2.Canny(gt_mask.astype(np.uint8) * 255, 100, 200)

        precision = np.sum(pred_boundary & gt_boundary) / (np.sum(pred_boundary) + 1e-6)
        recall = np.sum(pred_boundary & gt_boundary) / (np.sum(gt_boundary) + 1e-6)

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def compute_j_and_f(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
        j = VOSEvaluator.compute_jaccard(pred_mask, gt_mask)
        f = VOSEvaluator.compute_boundary_f(pred_mask, gt_mask)
        return j, f

    @staticmethod
    def evaluate_sequence(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> Dict[str, float]:
        j_scores = []
        f_scores = []

        for pred, gt in zip(pred_masks, gt_masks):
            j, f = VOSEvaluator.compute_j_and_f(pred, gt)
            j_scores.append(j)
            f_scores.append(f)

        return {
            'mean_j': np.mean(j_scores),
            'mean_f': np.mean(f_scores),
            'mean_jf': np.mean([(j + f) / 2 for j, f in zip(j_scores, f_scores)]),
        }


class VOSVisualizer:
    @staticmethod
    def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.5) -> np.ndarray:
        overlay = image.copy()
        mask_bool = mask > 0
        overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha
        return overlay

    @staticmethod
    def draw_contours(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        vis = image.copy()
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, thickness)
        return vis

    @staticmethod
    def create_comparison_grid(images: List[np.ndarray], masks: List[np.ndarray], gt_masks: List[np.ndarray] = None) -> np.ndarray:
        rows = len(images)
        cols = 3 if gt_masks else 2
        h, w = images[0].shape[:2]

        grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

        for i, (img, mask) in enumerate(zip(images, masks)):
            row = i
            grid[row*h:(row+1)*h, 0:w] = VOSVisualizer.overlay_mask(img, mask)
            grid[row*h:(row+1)*h, w:2*w] = VOSVisualizer.draw_contours(img, mask)

            if gt_masks:
                grid[row*h:(row+1)*h, 2*w:3*w] = VOSVisualizer.overlay_mask(img, gt_masks[i])

        return grid
