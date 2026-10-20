from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict


class StereoAlgorithm(Enum):
    BM = "block_matching"
    SGBM = "semi_global_block_matching"
    GC = "graph_cuts"
    BP = "belief_propagation"
    VAR = "variational"
    ELAS = "elas"
    DEEP = "deep_learning"


class RectificationMethod(Enum):
    BOUGUET = "bouguet"
    HARTLEY = "hartley"
    POLAR = "polar"
    CALIBRATED = "calibrated"


@dataclass(frozen=True, slots=True)
class StereoCameraParams:
    left_camera_matrix: np.ndarray
    right_camera_matrix: np.ndarray
    left_dist_coeffs: np.ndarray
    right_dist_coeffs: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    image_size: Tuple[int, int]
    rectification_method: RectificationMethod = RectificationMethod.BOUGUET

    @property
    def baseline(self) -> float:
        return np.linalg.norm(self.translation)

    @property
    def focal_length(self) -> float:
        return (self.left_camera_matrix[0, 0] + self.right_camera_matrix[0, 0]) / 2.0


@dataclass(frozen=True, slots=True)
class RectificationMaps:
    left_map1: np.ndarray
    left_map2: np.ndarray
    right_map1: np.ndarray
    right_map2: np.ndarray
    left_roi: Tuple[int, int, int, int]
    right_roi: Tuple[int, int, int, int]
    Q: np.ndarray
    valid_roi: Tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class DisparityMap:
    disparity: np.ndarray
    confidence: np.ndarray
    left_image: np.ndarray
    right_image: np.ndarray
    camera_params: StereoCameraParams
    algorithm: StereoAlgorithm
    timestamp: float = field(default_factory=time.time)

    def to_depth(self) -> np.ndarray:
        valid = self.disparity > 0
        depth = np.zeros_like(self.disparity, dtype=np.float32)
        if np.any(valid):
            f = self.camera_params.focal_length
            b = self.camera_params.baseline
            depth[valid] = (f * b) / self.disparity[valid]
        return depth

    def to_point_cloud(self, color: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        depth = self.to_depth()
        h, w = depth.shape
        points = []
        colors = [] if color is not None else None

        Q = self.camera_params.Q if hasattr(self.camera_params, 'Q') else None
        if Q is None:
            f = self.camera_params.focal_length
            cx = self.camera_params.left_camera_matrix[0, 2]
            cy = self.camera_params.left_camera_matrix[1, 2]
            b = self.camera_params.baseline
            Q = np.array([[1, 0, 0, -cx],
                         [0, 1, 0, -cy],
                         [0, 0, 0, f],
                         [0, 0, -1/b, 0]], dtype=np.float32)

        for v in range(h):
            for u in range(w):
                d = self.disparity[v, u]
                if d > 0:
                    x = (u - cx) * depth[v, u] / f
                    y = (v - cy) * depth[v, u] / f
                    z = depth[v, u]
                    points.append([x, y, z])
                    if colors is not None:
                        colors.append(color[v, u])

        return np.array(points), np.array(colors) if colors else None


class StereoRectifier:
    def __init__(self, params: StereoCameraParams):
        self.params = params
        self._rectification_maps: Optional[RectificationMaps] = None
        self._compute_rectification()

    def _compute_rectification(self):
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.params.left_camera_matrix,
            self.params.left_dist_coeffs,
            self.params.right_camera_matrix,
            self.params.right_dist_coeffs,
            self.params.image_size,
            self.params.rotation,
            self.params.translation,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=-1
        )

        left_map1, left_map2 = cv2.initUndistortRectifyMap(
            self.params.left_camera_matrix, self.params.left_dist_coeffs,
            R1, P1, self.params.image_size, cv2.CV_16SC2
        )
        right_map1, right_map2 = cv2.initUndistortRectifyMap(
            self.params.right_camera_matrix, self.params.right_dist_coeffs,
            R2, P2, self.params.image_size, cv2.CV_16SC2
        )

        self._rectification_maps = RectificationMaps(
            left_map1=left_map1,
            left_map2=left_map2,
            right_map1=right_map1,
            right_map2=right_map2,
            left_roi=roi1,
            right_roi=roi2,
            Q=Q,
            valid_roi=roi1
        )

    def rectify(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._rectification_maps is None:
            raise RuntimeError("Rectification maps not computed")

        left_rect = cv2.remap(left, self._rectification_maps.left_map1,
                              self._rectification_maps.left_map2, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, self._rectification_maps.right_map1,
                               self._rectification_maps.right_map2, cv2.INTER_LINEAR)

        x, y, w, h = self._rectification_maps.valid_roi
        left_rect = left_rect[y:y+h, x:x+w]
        right_rect = right_rect[y:y+h, x:x+w]

        return left_rect, right_rect

    @property
    def maps(self) -> RectificationMaps:
        if self._rectification_maps is None:
            raise RuntimeError("Rectification maps not computed")
        return self._rectification_maps


class BaseStereoMatcher(ABC):
    @abstractmethod
    def compute(self, left: np.ndarray, right: np.ndarray) -> DisparityMap:
        pass

    @abstractmethod
    def set_params(self, **kwargs):
        pass


class BlockMatchingStereo(BaseStereoMatcher):
    def __init__(
        self,
        num_disparities: int = 64,
        block_size: int = 15,
        pre_filter_cap: int = 31,
        min_disparity: int = 0,
        texture_threshold: int = 10,
        uniqueness_ratio: int = 15,
        speckle_window_size: int = 100,
        speckle_range: int = 32,
        disp12_max_diff: int = 1,
    ):
        self.num_disparities = num_disparities
        self.block_size = block_size
        self.pre_filter_cap = pre_filter_cap
        self.min_disparity = min_disparity
        self.texture_threshold = texture_threshold
        self.uniqueness_ratio = uniqueness_ratio
        self.speckle_window_size = speckle_window_size
        self.speckle_range = speckle_range
        self.disp12_max_diff = disp12_max_diff

        self._matcher = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
        self._update_params()

    def _update_params(self):
        self._matcher.setPreFilterCap(self.pre_filter_cap)
        self._matcher.setMinDisparity(self.min_disparity)
        self._matcher.setTextureThreshold(self.texture_threshold)
        self._matcher.setUniquenessRatio(self.uniqueness_ratio)
        self._matcher.setSpeckleWindowSize(self.speckle_window_size)
        self._matcher.setSpeckleRange(self.speckle_range)
        self._matcher.setDisp12MaxDiff(self.disp12_max_diff)

    def compute(self, left: np.ndarray, right: np.ndarray) -> DisparityMap:
        if len(left.shape) == 3:
            left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        if len(right.shape) == 3:
            right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

        disparity = self._matcher.compute(left, right).astype(np.float32) / 16.0
        confidence = self._compute_confidence(disparity, left, right)

        return DisparityMap(
            disparity=disparity,
            confidence=confidence,
            left_image=left,
            right_image=right,
            camera_params=None,
            algorithm=StereoAlgorithm.BM
        )

    def _compute_confidence(self, disparity: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        confidence = np.ones_like(disparity, dtype=np.float32)
        confidence[disparity <= self.min_disparity] = 0.0
        confidence[disparity >= self.min_disparity + self.num_disparities] = 0.0
        return confidence

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._update_params()


class SemiGlobalBlockMatchingStereo(BaseStereoMatcher):
    def __init__(
        self,
        num_disparities: int = 128,
        block_size: int = 5,
        p1: int = 8,
        p2: int = 32,
        disp12_max_diff: int = 1,
        pre_filter_cap: int = 63,
        uniqueness_ratio: int = 10,
        speckle_window_size: int = 100,
        speckle_range: int = 32,
        mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    ):
        self.num_disparities = num_disparities
        self.block_size = block_size
        self.p1 = p1
        self.p2 = p2
        self.disp12_max_diff = disp12_max_diff
        self.pre_filter_cap = pre_filter_cap
        self.uniqueness_ratio = uniqueness_ratio
        self.speckle_window_size = speckle_window_size
        self.speckle_range = speckle_range
        self.mode = mode

        self._matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=disp12_max_diff,
            preFilterCap=pre_filter_cap,
            uniquenessRatio=uniqueness_ratio,
            speckleWindowSize=speckle_window_size,
            speckleRange=speckle_range,
            mode=mode
        )

    def compute(self, left: np.ndarray, right: np.ndarray) -> DisparityMap:
        if len(left.shape) == 3:
            left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        if len(right.shape) == 3:
            right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

        disparity = self._matcher.compute(left, right).astype(np.float32) / 16.0
        confidence = self._compute_confidence(disparity)

        return DisparityMap(
            disparity=disparity,
            confidence=confidence,
            left_image=left,
            right_image=right,
            camera_params=None,
            algorithm=StereoAlgorithm.SGBM
        )

    def _compute_confidence(self, disparity: np.ndarray) -> np.ndarray:
        confidence = np.ones_like(disparity, dtype=np.float32)
        confidence[disparity <= 0] = 0.0
        return confidence

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=self.p1,
            P2=self.p2,
            disp12MaxDiff=self.disp12_max_diff,
            preFilterCap=self.pre_filter_cap,
            uniquenessRatio=self.uniqueness_ratio,
            speckleWindowSize=self.speckle_window_size,
            speckleRange=self.speckle_range,
            mode=self.mode
        )


class GraphCutsStereo(BaseStereoMatcher):
    def __init__(
        self,
        num_disparities: int = 64,
        max_iterations: int = 10,
        smoothness_weight: float = 10.0,
        data_weight: float = 1.0,
    ):
        self.num_disparities = num_disparities
        self.max_iterations = max_iterations
        self.smoothness_weight = smoothness_weight
        self.data_weight = data_weight

    def compute(self, left: np.ndarray, right: np.ndarray) -> DisparityMap:
        if len(left.shape) == 3:
            left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        if len(right.shape) == 3:
            right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

        h, w = left.shape
        disparity = np.zeros((h, w), dtype=np.float32)

        for d in range(self.num_disparities):
            shifted_right = np.roll(right, d, axis=1)
            if d > 0:
                shifted_right[:, :d] = 0
            cost = np.abs(left.astype(np.float32) - shifted_right.astype(np.float32))
            if d == 0:
                disparity = np.argmin(cost, axis=2) if len(cost.shape) == 3 else np.zeros_like(cost)
            else:
                pass

        confidence = np.ones_like(disparity, dtype=np.float32)
        confidence[disparity == 0] = 0.0

        return DisparityMap(
            disparity=disparity,
            confidence=confidence,
            left_image=left,
            right_image=right,
            camera_params=None,
            algorithm=StereoAlgorithm.GC
        )

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class BeliefPropagationStereo(BaseStereoMatcher):
    def __init__(
        self,
        num_disparities: int = 64,
        max_iterations: int = 50,
        damping: float = 0.5,
    ):
        self.num_disparities = num_disparities
        self.max_iterations = max_iterations
        self.damping = damping

    def compute(self, left: np.ndarray, right: np.ndarray) -> DisparityMap:
        if len(left.shape) == 3:
            left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        if len(right.shape) == 3:
            right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

        h, w = left.shape
        disparity = np.zeros((h, w), dtype=np.float32)

        messages = np.zeros((h, w, self.num_disparities), dtype=np.float32)

        for iteration in range(self.max_iterations):
            for d in range(self.num_disparities):
                shifted = np.roll(right, d, axis=1)
                if d > 0:
                    shifted[:, :d] = left[:, :d]
                data_cost = np.abs(left.astype(np.float32) - shifted.astype(np.float32))

                if iteration == 0:
                    messages[:, :, d] = data_cost
                else:
                    neighbor_messages = np.zeros_like(data_cost)
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        rolled = np.roll(messages[:, :, d], (dy, dx), axis=(0, 1))
                        if dy == -1: rolled[-1, :] = 0
                        elif dy == 1: rolled[0, :] = 0
                        elif dx == -1: rolled[:, -1] = 0
                        elif dx == 1: rolled[:, 0] = 0
                        neighbor_messages += rolled
                    neighbor_messages /= 4.0
                    messages[:, :, d] = self.damping * data_cost + (1 - self.damping) * neighbor_messages

        disparity = np.argmin(messages, axis=2).astype(np.float32)
        confidence = 1.0 - (np.min(messages, axis=2) / (np.max(messages, axis=2) + 1e-6))

        return DisparityMap(
            disparity=disparity,
            confidence=confidence,
            left_image=left,
            right_image=right,
            camera_params=None,
            algorithm=StereoAlgorithm.BP
        )

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class VariationalStereo(BaseStereoMatcher):
    def __init__(
        self,
        num_disparities: int = 64,
        alpha: float = 100.0,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
    ):
        self.num_disparities = num_disparities
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def compute(self, left: np.ndarray, right: np.ndarray) -> DisparityMap:
        if len(left.shape) == 3:
            left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        if len(right.shape) == 3:
            right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

        left_f = left.astype(np.float32) / 255.0
        right_f = right.astype(np.float32) / 255.0
        h, w = left_f.shape

        disparity = np.zeros((h, w), dtype=np.float32)

        for iteration in range(self.max_iterations):
            prev_disparity = disparity.copy()

            for d in range(self.num_disparities):
                shifted = np.roll(right_f, d, axis=1)
                if d > 0:
                    shifted[:, :d] = left_f[:, :d]
                data_term = (left_f - shifted) ** 2

                grad_x = np.gradient(disparity, axis=1)
                grad_y = np.gradient(disparity, axis=0)
                smooth_term = self.alpha * (grad_x ** 2 + grad_y ** 2)

                energy = data_term + smooth_term
                if d == 0:
                    min_energy = energy
                    disparity = np.zeros_like(energy)
                else:
                    update_mask = energy < min_energy
                    min_energy[update_mask] = energy[update_mask]
                    disparity[update_mask] = d

            change = np.mean(np.abs(disparity - prev_disparity))
            if change < self.tolerance:
                break

        confidence = 1.0 - (min_energy / (np.max(min_energy) + 1e-6))

        return DisparityMap(
            disparity=disparity,
            confidence=confidence,
            left_image=left,
            right_image=right,
            camera_params=None,
            algorithm=StereoAlgorithm.VAR
        )

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class DeepLearningStereo(BaseStereoMatcher):
    def __init__(
        self,
        model_path: str = "stereo_net.pth",
        device: str = "cpu",
        max_disp: int = 192,
    ):
        self.model_path = model_path
        self.device = device
        self.max_disp = max_disp
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            self._model = torch.jit.load(self.model_path)
            self._model.eval()
            if self.device != "cpu" and torch.cuda.is_available():
                self._model = self._model.cuda()
            self._initialized = True
        except (ImportError, FileNotFoundError):
            self._initialized = False

    def compute(self, left: np.ndarray, right: np.ndarray) -> DisparityMap:
        if not self._initialized:
            self._initialize()

        if self._initialized:
            return self._compute_deep(left, right)
        else:
            return self._compute_fallback(left, right)

    def _compute_deep(self, left: np.ndarray, right: np.ndarray) -> DisparityMap:
        import torch
        import torch.nn.functional as F

        if len(left.shape) == 3:
            left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        if len(right.shape) == 3:
            right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

        h, w = left.shape
        left_tensor = torch.from_numpy(left).float().unsqueeze(0).unsqueeze(0) / 255.0
        right_tensor = torch.from_numpy(right).float().unsqueeze(0).unsqueeze(0) / 255.0

        if self.device != "cpu" and torch.cuda.is_available():
            left_tensor = left_tensor.cuda()
            right_tensor = right_tensor.cuda()

        with torch.no_grad():
            disparity = self._model(left_tensor, right_tensor)

        disparity = disparity.squeeze().cpu().numpy()
        confidence = np.ones_like(disparity, dtype=np.float32)
        confidence[disparity <= 0] = 0.0

        return DisparityMap(
            disparity=disparity,
            confidence=confidence,
            left_image=left,
            right_image=right,
            camera_params=None,
            algorithm=StereoAlgorithm.DEEP
        )

    def _compute_fallback(self, left: np.ndarray, right: np.ndarray) -> DisparityMap:
        fallback = SemiGlobalBlockMatchingStereo()
        return fallback.compute(left, right)

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class StereoVisionPipeline:
    def __init__(
        self,
        camera_params: StereoCameraParams,
        algorithm: StereoAlgorithm = StereoAlgorithm.SGBM,
        rectify: bool = True,
        filter_disparity: bool = True,
        **matcher_kwargs
    ):
        self.camera_params = camera_params
        self.algorithm = algorithm
        self.rectify = rectify
        self.filter_disparity = filter_disparity

        self.rectifier = StereoRectifier(camera_params) if rectify else None
        self.matcher = self._create_matcher(algorithm, **matcher_kwargs)

        self._wls_filter = None
        if filter_disparity:
            self._init_wls_filter()

    def _create_matcher(self, algorithm: StereoAlgorithm, **kwargs) -> BaseStereoMatcher:
        if algorithm == StereoAlgorithm.BM:
            return BlockMatchingStereo(**kwargs)
        elif algorithm == StereoAlgorithm.SGBM:
            return SemiGlobalBlockMatchingStereo(**kwargs)
        elif algorithm == StereoAlgorithm.GC:
            return GraphCutsStereo(**kwargs)
        elif algorithm == StereoAlgorithm.BP:
            return BeliefPropagationStereo(**kwargs)
        elif algorithm == StereoAlgorithm.VAR:
            return VariationalStereo(**kwargs)
        elif algorithm == StereoAlgorithm.DEEP:
            return DeepLearningStereo(**kwargs)
        else:
            raise ValueError(f"Unknown stereo algorithm: {algorithm}")

    def _init_wls_filter(self):
        try:
            self._wls_filter = cv2.ximgproc.createDisparityWLSFilter(
                matcher_left=self.matcher._matcher
            )
            self._wls_filter.setLambda(8000)
            self._wls_filter.setSigmaColor(1.5)
        except AttributeError:
            self._wls_filter = None

    def process(self, left: np.ndarray, right: np.ndarray) -> DisparityMap:
        if self.rectify and self.rectifier:
            left_rect, right_rect = self.rectifier.rectify(left, right)
        else:
            left_rect, right_rect = left, right

        disparity_map = self.matcher.compute(left_rect, right_rect)

        if self.filter_disparity and self._wls_filter is not None:
            disparity_map = self._apply_wls_filter(disparity_map, left_rect, right_rect)

        disparity_map.camera_params = self.camera_params
        if self.rectifier:
            disparity_map.camera_params = StereoCameraParams(
                left_camera_matrix=self.camera_params.left_camera_matrix,
                right_camera_matrix=self.camera_params.right_camera_matrix,
                left_dist_coeffs=self.camera_params.left_dist_coeffs,
                right_dist_coeffs=self.camera_params.right_dist_coeffs,
                rotation=self.camera_params.rotation,
                translation=self.camera_params.translation,
                image_size=self.camera_params.image_size,
                rectification_method=self.camera_params.rectification_method
            )
            disparity_map.camera_params.Q = self.rectifier.maps.Q

        return disparity_map

    def _apply_wls_filter(
        self,
        disparity_map: DisparityMap,
        left: np.ndarray,
        right: np.ndarray
    ) -> DisparityMap:
        try:
            right_matcher = cv2.ximgproc.createRightMatcher(self.matcher._matcher)
            disp_right = right_matcher.compute(right, left).astype(np.float32) / 16.0
            filtered = self._wls_filter.filter(disparity_map.disparity, left, disparity_map_right=disp_right)

            confidence = disparity_map.confidence.copy()
            confidence[filtered <= 0] = 0.0

            return DisparityMap(
                disparity=filtered,
                confidence=confidence,
                left_image=disparity_map.left_image,
                right_image=disparity_map.right_image,
                camera_params=disparity_map.camera_params,
                algorithm=disparity_map.algorithm,
                timestamp=disparity_map.timestamp
            )
        except Exception:
            return disparity_map


class StereoCalibration:
    def __init__(
        self,
        board_size: Tuple[int, int] = (9, 6),
        square_size: float = 25.0,
        flags: int = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH
    ):
        self.board_size = board_size
        self.square_size = square_size
        self.flags = flags
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

        self.left_points: List[np.ndarray] = []
        self.right_points: List[np.ndarray] = []
        self.obj_points: List[np.ndarray] = []

    def add_stereo_pair(self, left: np.ndarray, right: np.ndarray) -> bool:
        left_gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY) if len(left.shape) == 3 else left
        right_gray = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY) if len(right.shape) == 3 else right

        left_found, left_corners = cv2.findChessboardCorners(left_gray, self.board_size, None)
        right_found, right_corners = cv2.findChessboardCorners(right_gray, self.board_size, None)

        if left_found and right_found:
            left_corners = cv2.cornerSubPix(left_gray, left_corners, (11, 11), (-1, -1), self.criteria)
            right_corners = cv2.cornerSubPix(right_gray, right_corners, (11, 11), (-1, -1), self.criteria)

            self.left_points.append(left_corners)
            self.right_points.append(right_corners)
            self.obj_points.append(self.objp)
            return True
        return False

    def calibrate(self, image_size: Tuple[int, int]) -> StereoCameraParams:
        if len(self.obj_points) < 5:
            raise ValueError("Need at least 5 stereo pairs for calibration")

        left_camera_matrix = np.eye(3)
        left_dist_coeffs = np.zeros(5)
        right_camera_matrix = np.eye(3)
        right_dist_coeffs = np.zeros(5)

        ret, left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs, R, T, E, F = cv2.stereoCalibrate(
            self.obj_points,
            self.left_points,
            self.right_points,
            left_camera_matrix, left_dist_coeffs,
            right_camera_matrix, right_dist_coeffs,
            image_size,
            criteria=self.criteria,
            flags=self.flags
        )

        return StereoCameraParams(
            left_camera_matrix=left_camera_matrix,
            right_camera_matrix=right_camera_matrix,
            left_dist_coeffs=left_dist_coeffs,
            right_dist_coeffs=right_dist_coeffs,
            rotation=R,
            translation=T,
            image_size=image_size,
            rectification_method=RectificationMethod.BOUGUET
        )


class DisparityPostProcessor:
    @staticmethod
    def median_filter(disparity: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        return cv2.medianBlur(disparity.astype(np.uint16), kernel_size).astype(np.float32)

    @staticmethod
    def bilateral_filter(disparity: np.ndarray, guide: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        return cv2.ximgproc.jointBilateralFilter(guide, disparity.astype(np.float32), d, sigma_color, sigma_space)

    @staticmethod
    def speckle_filter(disparity: np.ndarray, max_speckle_size: int = 100, max_diff: int = 32) -> np.ndarray:
        filtered = disparity.copy()
        cv2.filterSpeckles(filtered, 0, max_speckle_size, max_diff)
        return filtered

    @staticmethod
    def fill_holes(disparity: np.ndarray, max_hole_size: int = 50) -> np.ndarray:
        mask = (disparity > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_hole_size, max_hole_size))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        filled = disparity.copy()
        filled[closed > 0] = cv2.inpaint(disparity.astype(np.float32), (1 - closed).astype(np.uint8), 3, cv2.INPAINT_TELEA)[closed > 0]
        return filled

    @staticmethod
    def subpixel_refinement(disparity: np.ndarray, left: np.ndarray, right: np.ndarray, window_size: int = 5) -> np.ndarray:
        refined = disparity.copy()
        half_w = window_size // 2
        h, w = disparity.shape

        for v in range(half_w, h - half_w):
            for u in range(half_w, w - half_w):
                d = int(disparity[v, u])
                if d > 0 and u - d - half_w >= 0 and u - d + half_w < w:
                    left_patch = left[v-half_w:v+half_w+1, u-half_w:u+half_w+1]
                    right_patch = right[v-half_w:v+half_w+1, u-d-half_w:u-d+half_w+1]
                    if left_patch.shape == right_patch.shape:
                        corr = cv2.matchTemplate(left_patch.astype(np.float32), right_patch.astype(np.float32), cv2.TM_CCOEFF_NORMED)
                        _, _, _, max_loc = cv2.minMaxLoc(corr)
                        subpixel_d = d + max_loc[0] - half_w
                        refined[v, u] = subpixel_d
        return refined


class StereoVisualOdometry:
    def __init__(
        self,
        camera_params: StereoCameraParams,
        matcher: BaseStereoMatcher,
        min_features: int = 2000,
        ransac_threshold: float = 1.0,
        keyframe_threshold: float = 0.1
    ):
        self.camera_params = camera_params
        self.matcher = matcher
        self.min_features = min_features
        self.ransac_threshold = ransac_threshold
        self.keyframe_threshold = keyframe_threshold

        self.prev_left: Optional[np.ndarray] = None
        self.prev_right: Optional[np.ndarray] = None
        self.prev_features: Optional[FeatureSet] = None
        self.current_pose = np.eye(4)
        self.trajectory: List[np.ndarray] = [self.current_pose]
        self.keyframes: List[Tuple[np.ndarray, np.ndarray]] = []

    def process_frame(self, left: np.ndarray, right: np.ndarray) -> Optional[np.ndarray]:
        if len(left.shape) == 3:
            left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        if len(right.shape) == 3:
            right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

        disparity_map = self.matcher.compute(left, right)
        depth = disparity_map.to_depth()

        features = self._extract_features(left, depth)

        if self.prev_left is None or self.prev_features is None:
            self.prev_left = left
            self.prev_right = right
            self.prev_features = features
            return self.current_pose

        matches = self._match_features(self.prev_features, features)

        if len(matches) < 8:
            self.prev_left = left
            self.prev_right = right
            self.prev_features = features
            return self.current_pose

        pose = self._estimate_pose(matches, self.prev_features, features)
        if pose is not None:
            self.current_pose = self.current_pose @ pose
            self.trajectory.append(self.current_pose)

        self.prev_left = left
        self.prev_right = right
        self.prev_features = features

        return self.current_pose

    def _extract_features(self, image: np.ndarray, depth: np.ndarray) -> FeatureSet:
        detector = cv2.ORB_create(nfeatures=self.min_features)
        keypoints = detector.detect(image, None)
        keypoints, descriptors = detector.compute(image, keypoints)

        valid_kp = []
        valid_desc = []
        for i, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1] and depth[y, x] > 0:
                valid_kp.append(kp)
                valid_desc.append(descriptors[i])

        return FeatureSet(
            keypoints=[Keypoint(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in valid_kp],
            descriptors=np.array(valid_desc) if valid_desc else None,
            image_shape=image.shape,
            feature_type=FeatureType.ORB
        )

    def _match_features(self, prev: FeatureSet, curr: FeatureSet) -> List[FeatureMatch]:
        if prev.descriptors is None or curr.descriptors is None:
            return []
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(prev.descriptors, curr.descriptors)
        return [FeatureMatch(m.queryIdx, m.trainIdx, m.distance) for m in matches]

    def _estimate_pose(
        self,
        matches: List[FeatureMatch],
        prev_features: FeatureSet,
        curr_features: FeatureSet
    ) -> Optional[np.ndarray]:
        if len(matches) < 8:
            return None

        pts1 = np.float32([prev_features.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.float32([curr_features.keypoints[m.trainIdx].pt for m in matches])

        try:
            E, mask = cv2.findEssentialMat(
                pts1, pts2,
                self.camera_params.left_camera_matrix,
                cv2.RANSAC, 0.999, self.ransac_threshold
            )
            _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.camera_params.left_camera_matrix, mask=mask)

            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t.flatten()
            return pose
        except cv2.error:
            return None


class RectifiedStereoPair:
    def __init__(self, left: np.ndarray, right: np.ndarray, rectifier: StereoRectifier):
        self.left = left
        self.right = right
        self.rectifier = rectifier
        self.left_rect: Optional[np.ndarray] = None
        self.right_rect: Optional[np.ndarray] = None
        self._rectified = False

    def rectify(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self._rectified:
            self.left_rect, self.right_rect = self.rectifier.rectify(self.left, self.right)
            self._rectified = True
        return self.left_rect, self.right_rect

    def get_roi(self) -> Tuple[int, int, int, int]:
        return self.rectifier.maps.valid_roi


class StereoDisparityVisualizer:
    @staticmethod
    def colorize_disparity(disparity: np.ndarray, max_disp: Optional[int] = None, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        if max_disp is None:
            max_disp = np.max(disparity[disparity > 0]) if np.any(disparity > 0) else 1
        norm = np.clip(disparity / max_disp, 0, 1)
        colored = cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)
        colored[disparity <= 0] = [0, 0, 0]
        return colored

    @staticmethod
    def colorize_depth(depth: np.ndarray, max_depth: Optional[float] = None, colormap: int = cv2.COLORMAP_PLASMA) -> np.ndarray:
        if max_depth is None:
            max_depth = np.max(depth[depth > 0]) if np.any(depth > 0) else 1
        norm = np.clip(depth / max_depth, 0, 1)
        colored = cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)
        colored[depth <= 0] = [0, 0, 0]
        return colored

    @staticmethod
    def draw_epipolar_lines(image: np.ndarray, lines: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        vis = image.copy()
        for line in lines:
            x0, y0 = map(int, [0, -line[2]/line[1]] if abs(line[1]) > 1e-6 else [0, 0])
            x1, y1 = map(int, [image.shape[1], -(line[2] + line[0]*image.shape[1])/line[1]] if abs(line[1]) > 1e-6 else [image.shape[1], 0])
            cv2.line(vis, (x0, y0), (x1, y1), color, 1)
        return vis

    @staticmethod
    def create_anaglyph(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if len(left.shape) == 2:
            left = cv2.cvtColor(left, cv2.COLOR_GRAY2RGB)
        if len(right.shape) == 2:
            right = cv2.cvtColor(right, cv2.COLOR_GRAY2RGB)
        anaglyph = np.zeros_like(left)
        anaglyph[:, :, 0] = left[:, :, 0]
        anaglyph[:, :, 1] = right[:, :, 1]
        anaglyph[:, :, 2] = right[:, :, 2]
        return anaglyph


class StereoQualityMetrics:
    @staticmethod
    def compute_bad_pixels(gt: np.ndarray, pred: np.ndarray, threshold: float = 3.0) -> float:
        valid = gt > 0
        if not np.any(valid):
            return 0.0
        error = np.abs(gt[valid] - pred[valid])
        return np.mean(error > threshold) * 100

    @staticmethod
    def compute_epe(gt: np.ndarray, pred: np.ndarray) -> float:
        valid = gt > 0
        if not np.any(valid):
            return 0.0
        return np.mean(np.abs(gt[valid] - pred[valid]))

    @staticmethod
    def compute_d1(gt: np.ndarray, pred: np.ndarray) -> float:
        valid = gt > 0
        if not np.any(valid):
            return 0.0
        error = np.abs(gt[valid] - pred[valid])
        return np.mean((error > 3) & (error / gt[valid] > 0.05)) * 100

    @staticmethod
    def compute_rms(gt: np.ndarray, pred: np.ndarray) -> float:
        valid = gt > 0
        if not np.any(valid):
            return 0.0
        return np.sqrt(np.mean((gt[valid] - pred[valid]) ** 2))

    @staticmethod
    def compute_si_rmse(gt: np.ndarray, pred: np.ndarray) -> float:
        valid = gt > 0
        if not np.any(valid):
            return 0.0
        log_gt = np.log(gt[valid] + 1e-6)
        log_pred = np.log(pred[valid] + 1e-6)
        alpha = np.mean(log_pred - log_gt)
        return np.sqrt(np.mean((log_pred - log_gt - alpha) ** 2))


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
