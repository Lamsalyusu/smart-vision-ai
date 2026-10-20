from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import math
import time


class FeatureType(Enum):
    SIFT = "sift"
    SURF = "surf"
    ORB = "orb"
    AKAZE = "akaze"
    BRISK = "brisk"
    KAZE = "kaze"
    SUPERPOINT = "superpoint"
    D2NET = "d2net"
    R2D2 = "r2d2"
    DISK = "disk"


@dataclass(frozen=True, slots=True)
class Keypoint:
    x: float
    y: float
    size: float
    angle: float
    response: float
    octave: int
    class_id: int = -1

    def to_cv2(self) -> cv2.KeyPoint:
        return cv2.KeyPoint(self.x, self.y, self.size, self.angle, self.response, self.octave, self.class_id)


@dataclass(frozen=True, slots=True)
class FeatureMatch:
    query_idx: int
    train_idx: int
    distance: float
    img_idx: int = 0


@dataclass(frozen=True, slots=True)
class FeatureSet:
    keypoints: List[Keypoint]
    descriptors: Optional[np.ndarray]
    image_shape: Tuple[int, int]
    feature_type: FeatureType


class BaseFeatureExtractor(ABC):
    @abstractmethod
    def detect_and_compute(self, image: np.ndarray) -> FeatureSet:
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Keypoint]:
        pass

    @abstractmethod
    def compute(self, image: np.ndarray, keypoints: List[Keypoint]) -> Tuple[List[Keypoint], np.ndarray]:
        pass


class SIFTFeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        nfeatures: int = 0,
        n_octave_layers: int = 3,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10,
        sigma: float = 1.6,
    ):
        self.detector = cv2.SIFT_create(
            nfeatures=nfeatures,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma,
        )

    def detect_and_compute(self, image: np.ndarray) -> FeatureSet:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        kp, desc = self.detector.detectAndCompute(gray, None)
        keypoints = [Keypoint(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp] if kp else []
        return FeatureSet(keypoints, desc, image.shape[:2], FeatureType.SIFT)

    def detect(self, image: np.ndarray) -> List[Keypoint]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        kp = self.detector.detect(gray, None)
        return [Keypoint(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp] if kp else []

    def compute(self, image: np.ndarray, keypoints: List[Keypoint]) -> Tuple[List[Keypoint], np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        cv_kp = [k.to_cv2() for k in keypoints]
        kp, desc = self.detector.compute(gray, cv_kp)
        return [Keypoint(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp], desc


class SURFFeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        hessian_threshold: float = 400,
        n_octaves: int = 4,
        n_octave_layers: int = 3,
        extended: bool = False,
        upright: bool = False,
    ):
        self.detector = cv2.xfeatures2d.SURF_create(
            hessianThreshold=hessian_threshold,
            nOctaves=n_octaves,
            nOctaveLayers=n_octave_layers,
            extended=extended,
            upright=upright,
        )

    def detect_and_compute(self, image: np.ndarray) -> FeatureSet:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        kp, desc = self.detector.detectAndCompute(gray, None)
        keypoints = [Keypoint(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp] if kp else []
        return FeatureSet(keypoints, desc, image.shape[:2], FeatureType.SURF)

    def detect(self, image: np.ndarray) -> List[Keypoint]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        kp = self.detector.detect(gray, None)
        return [Keypoint(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp] if kp else []

    def compute(self, image: np.ndarray, keypoints: List[Keypoint]) -> Tuple[List[Keypoint], np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        cv_kp = [k.to_cv2() for k in keypoints]
        kp, desc = self.detector.compute(image, cv_kp)
        return [Keypoint(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp], desc


class ORBFeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        nfeatures: int = 500,
        scale_factor: float = 1.2,
        nlevels: int = 8,
        edge_threshold: int = 31,
        first_level: int = 0,
        WTA_K: int = 2,
        score_type: int = cv2.ORB_HARRIS_SCORE,
        patch_size: int = 31,
        fast_threshold: int = 20,
    ):
        self.detector = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=scale_factor,
            nlevels=nlevels,
            edgeThreshold=edge_threshold,
            firstLevel=first_level,
            WTA_K=WTA_K,
            scoreType=score_type,
            patchSize=patch_size,
            fastThreshold=fast_threshold,
        )

    def detect_and_compute(self, image: np.ndarray) -> FeatureSet:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        kp, desc = self.detector.detectAndCompute(gray, None)
        keypoints = [Keypoint(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp] if kp else []
        return FeatureSet(keypoints, desc, image.shape[:2], FeatureType.ORB)

    def detect(self, image: np.ndarray) -> List[Keypoint]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        kp = self.detector.detect(gray, None)
        return [Keypoint(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp] if kp else []

    def compute(self, image: np.ndarray, keypoints: List[Keypoint]) -> Tuple[List[Keypoint], np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        cv_kp = [k.to_cv2() for k in keypoints]
        kp, desc = self.detector.compute(image, cv_kp)
        return [Keypoint(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp], desc


class AKAZEFeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        descriptor_type: int = cv2.AKAZE_DESCRIPTOR_MLDB,
        descriptor_size: int = 0,
        descriptor_channels: int = 3,
        threshold: float = 0.001,
        n_octaves: int = 4,
        n_octave_layers: int = 4,
        diffusivity: int = cv2.KAZE_DIFF_PM_G2,
    ):
        self.detector = cv2.AKAZE_create(
            descriptor_type=descriptor_type,
            descriptor_size=descriptor_size,
            descriptor_channels=descriptor_channels,
            threshold=threshold,
            nOctaves=n_octaves,
            nOctaveLayers=n_octave_layers,
            diffusivity=diffusivity,
        )

    def detect_and_compute(self, image: np.ndarray) -> FeatureSet:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        kp, desc = self.detector.detectAndCompute(gray, None)
        keypoints = [Keypoint(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp] if kp else []
        return FeatureSet(keypoints, desc, image.shape[:2], FeatureType.AKAZE)

    def detect(self, image: np.ndarray) -> List[Keypoint]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        kp = self.detector.detect(gray, None)
        return [Keypoint(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp] if kp else []

    def compute(self, image: np.ndarray, keypoints: List[Keypoint]) -> Tuple[List[Keypoint], np.ndarray]:
        cv_kp = [k.to_cv2() for k in keypoints]
        kp, desc = self.detector.compute(image, cv_kp)
        return [Keypoint(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp], desc


class BRISKFeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        thresh: int = 30,
        octaves: int = 4,
        pattern_scale: float = 1.0,
    ):
        self.detector = cv2.BRISK_create(thresh=thresh, octaves=octaves, patternScale=pattern_scale)

    def detect_and_compute(self, image: np.ndarray) -> FeatureSet:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        kp, desc = self.detector.detectAndCompute(gray, None)
        keypoints = [Keypoint(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp] if kp else []
        return FeatureSet(keypoints, desc, image.shape[:2], FeatureType.BRISK)

    def detect(self, image: np.ndarray) -> List[Keypoint]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        kp = self.detector.detect(gray, None)
        return [Keypoint(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp] if kp else []

    def compute(self, image: np.ndarray, keypoints: List[Keypoint]) -> Tuple[List[Keypoint], np.ndarray]:
        cv_kp = [k.to_cv2() for k in keypoints]
        kp, desc = self.detector.compute(image, cv_kp)
        return [Keypoint(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp], desc


class KAZEFeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        extended: bool = False,
        upright: bool = False,
        threshold: float = 0.001,
        n_octaves: int = 4,
        n_octave_layers: int = 4,
        diffusivity: int = cv2.KAZE_DIFF_PM_G2,
    ):
        self.detector = cv2.KAZE_create(
            extended=extended,
            upright=upright,
            threshold=threshold,
            nOctaves=n_octaves,
            nOctaveLayers=n_octave_layers,
            diffusivity=diffusivity,
        )

    def detect_and_compute(self, image: np.ndarray) -> FeatureSet:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        kp, desc = self.detector.detectAndCompute(gray, None)
        keypoints = [Keypoint(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp] if kp else []
        return FeatureSet(keypoints, desc, image.shape[:2], FeatureType.KAZE)

    def detect(self, image: np.ndarray) -> List[Keypoint]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        kp = self.detector.detect(gray, None)
        return [Keypoint(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp] if kp else []

    def compute(self, image: np.ndarray, keypoints: List[Keypoint]) -> Tuple[List[Keypoint], np.ndarray]:
        cv_kp = [k.to_cv2() for k in keypoints]
        kp, desc = self.detector.compute(image, cv_kp)
        return [Keypoint(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in kp], desc


class SuperPointFeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        model_path: str = "superpoint.pth",
        nms_radius: int = 4,
        keypoint_threshold: float = 0.005,
        max_keypoints: int = 1024,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.nms_radius = nms_radius
        self.keypoint_threshold = keypoint_threshold
        self.max_keypoints = max_keypoints
        self.device = device
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            self._model = torch.jit.load(self.model_path)
            self._model.eval()
            self._initialized = True
        except ImportError:
            self._initialized = False

    def detect_and_compute(self, image: np.ndarray) -> FeatureSet:
        if not self._initialized:
            self._initialize()

        if self._initialized:
            return self._detect_torch(image)
        else:
            return self._detect_classical(image)

    def _detect_classical(self, image: np.ndarray) -> FeatureSet:
        extractor = ORBFeatureExtractor()
        return extractor.detect_and_compute(image)

    def _detect_torch(self, image: np.ndarray) -> FeatureSet:
        import torch
        import torch.nn.functional as F

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

        if self.device != "cpu" and torch.cuda.is_available():
            tensor = tensor.cuda()

        with torch.no_grad():
            pred = self._model(tensor)

        keypoints = pred['keypoints'][0].cpu().numpy()
        descriptors = pred['descriptors'][0].cpu().numpy()
        scores = pred['scores'][0].cpu().numpy()

        mask = scores > 0.005
        keypoints = keypoints[mask]
        descriptors = descriptors[mask]

        kps = [Keypoint(k[0], k[1], 1.0, 0.0, 1.0, 0, -1) for k in keypoints]
        return FeatureSet(kps, descriptors, (image.shape[0], image.shape[1]), FeatureType.SUPERPOINT)

    def detect(self, image: np.ndarray) -> List[Keypoint]:
        return self.detect_and_compute(image).keypoints

    def compute(self, image: np.ndarray, keypoints: List[Keypoint]) -> Tuple[List[Keypoint], np.ndarray]:
        return keypoints, np.array([])


class FeatureMatcher:
    def __init__(
        self,
        matcher_type: str = "bf",
        norm_type: int = cv2.NORM_L2,
        cross_check: bool = True,
        ratio_threshold: float = 0.8,
    ):
        self.matcher_type = matcher_type
        self.norm_type = norm_type
        self.cross_check = cross_check
        self.ratio_threshold = ratio_threshold

        if matcher_type == "bf":
            self.matcher = cv2.BFMatcher(norm_type, crossCheck=cross_check)
        elif matcher_type == "flann":
            if norm_type == cv2.NORM_L2:
                index_params = dict(algorithm=1, trees=5)
            else:
                index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, desc1: np.ndarray, desc2: np.ndarray) -> List[FeatureMatch]:
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []

        if self.cross_check:
            matches = self.matcher.match(desc1, desc2)
            return [FeatureMatch(m.queryIdx, m.trainIdx, m.distance) for m in matches]
        else:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            good = []
            for m, n in matches:
                if m.distance < self.ratio_threshold * n.distance:
                    good.append(FeatureMatch(m.queryIdx, m.trainIdx, m.distance))
            return good


class FeatureMatcherBF(FeatureMatcher):
    def __init__(self, norm_type: int = cv2.NORM_L2, cross_check: bool = True):
        super().__init__("bf", norm_type, cross_check)


class FeatureMatcherFLANN(FeatureMatcher):
    def __init__(self, norm_type: int = cv2.NORM_L2):
        super().__init__("flann", norm_type, False)


class FeatureTracker:
    def __init__(
        self,
        extractor: BaseFeatureExtractor,
        matcher: FeatureMatcher,
        max_lost_frames: int = 10,
    ):
        self.extractor = extractor
        self.matcher = matcher
        self.max_lost_frames = max_lost_frames
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 0

    def update(self, image: np.ndarray) -> Dict[int, FeatureSet]:
        features = self.extractor.detect_and_compute(image)
        results = {}

        if not self.tracks:
            for i, (kp, desc) in enumerate(zip(features.keypoints, features.descriptors)):
                self.tracks[self.next_id] = {
                    'keypoint': features.keypoints[i],
                    'descriptor': features.descriptors[i],
                    'age': 0,
                    'lost_frames': 0,
                }
                self.next_id += 1
        else:
            prev_descriptors = np.array([t['descriptor'] for t in self.tracks.values()])
            curr_descriptors = features.descriptors

            matches = self.matcher.match(prev_descriptors, features.descriptors)

            matched_prev = set()
            matched_curr = set()
            for match in self.matcher.match(
                    np.array([t['descriptor'] for t in self.tracks.values()]),
                    features.descriptors):
                track_ids = list(self.tracks.keys())
                track_id = track_ids[match.queryIdx]
                feat_idx = match.trainIdx

                self.tracks[track_id]['keypoint'] = features.keypoints[feat_idx]
                self.tracks[track_id]['descriptor'] = features.descriptors[feat_idx]
                self.tracks[track_id]['age'] += 1
                self.tracks[track_id]['lost_frames'] = 0
                matched_prev.add(match.queryIdx)
                matched_curr.add(feat_idx)

            for tid in list(self.tracks.keys()):
                if tid not in matched_prev:
                    self.tracks[tid]['lost_frames'] += 1
                    if self.tracks[tid]['lost_frames'] > self.max_lost_frames:
                        del self.tracks[tid]

            for i in range(len(features.keypoints)):
                if i not in matched_curr:
                    self.tracks[self.next_id] = {
                        'keypoint': features.keypoints[i],
                        'descriptor': features.descriptors[i],
                        'age': 0,
                        'lost_frames': 0,
                    }
                    self.next_id += 1

        for tid, track in self.tracks.items():
            results[tid] = FeatureSet(
                [track['keypoint']],
                track['descriptor'].reshape(1, -1),
                (1, 1),
                FeatureType.ORB,
            )

        return results


class VisualOdometry:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        extractor: BaseFeatureExtractor,
        matcher: FeatureMatcher,
        min_features: int = 1000,
        ransac_threshold: float = 1.0,
    ):
        self.camera_matrix = camera_matrix
        self.extractor = extractor
        self.matcher = matcher
        self.min_features = min_features
        self.ransac_threshold = ransac_threshold
        self.prev_image: Optional[np.ndarray] = None
        self.prev_kp: Optional[list] = None
        self.prev_desc: Optional[np.ndarray] = None
        self.current_pose = np.eye(4)
        self.trajectory: List[np.ndarray] = [self.current_pose]

    def process_frame(self, image: np.ndarray, timestamp: float = 0.0) -> Optional[np.ndarray]:
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        features = self.extractor.detect_and_compute(image)

        if self.prev_image is None or self.prev_desc is None:
            self.prev_image = image
            self.prev_kp = features.keypoints
            self.prev_desc = features.descriptors
            return self.current_pose

        matches = self.matcher.match(self.prev_desc, features.descriptors)

        if len(matches) < 8:
            self.prev_image = image
            self.prev_kp = features.keypoints
            self.prev_desc = features.descriptors
            return self.current_pose

        pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in self.matcher.match(self.prev_desc, features.descriptors)])
        pts2 = np.float32([features.keypoints[m.trainIdx].pt for m in self.matcher.match(self.prev_desc, features.descriptors)])

        try:
            E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix, cv2.RANSAC, 0.999, 1.0)
            _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.camera_matrix, mask=mask)

            relative_pose = np.eye(4)
            relative_pose[:3, :3] = R
            relative_pose[:3, 3] = t.flatten()
            self.current_pose = self.current_pose @ relative_pose
            self.trajectory.append(self.current_pose)

        except cv2.error:
            pass

        self.prev_image = image
        self.prev_kp = features.keypoints
        self.prev_desc = features.descriptors

        return self.current_pose

    def get_trajectory(self) -> List[np.ndarray]:
        return self.trajectory


class FeatureVisualizer:
    @staticmethod
    def draw_keypoints(image: np.ndarray, keypoints: List[Keypoint], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        vis = image.copy()
        for kp in keypoints:
            cv2.circle(vis, (int(kp.x), int(kp.y)), int(kp.size/2), color, 2)
            if kp.angle != 0:
                end_x = int(kp.x + kp.size * math.cos(kp.angle))
                end_y = int(kp.y + kp.size * math.sin(kp.angle))
                cv2.line(vis, (int(kp.x), int(kp.y)), (end_x, end_y), color, 1)
        return vis

    @staticmethod
    def draw_matches(image1: np.ndarray, kp1: List[Keypoint], image2: np.ndarray, kp2: List[Keypoint], matches: List[FeatureMatch], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        vis[:h1, :w1] = image1
        vis[:h2, w1:w1+w2] = image2

        for match in matches:
            pt1 = (int(kp1[match.queryIdx].x), int(kp1[match.queryIdx].y))
            pt2 = (int(kp2[match.trainIdx].x + w1), int(kp2[match.trainIdx].y))
            cv2.line(vis, pt1, pt2, color, 1)
            cv2.circle(vis, pt1, 3, color, -1)
            cv2.circle(vis, pt2, 3, color, -1)

        return vis

    @staticmethod
    def draw_track(image: np.ndarray, keypoints: List[Keypoint], track_ids: List[int], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        vis = image.copy()
        for kp, tid in zip(keypoints, track_ids):
            cv2.circle(vis, (int(kp.x), int(kp.y)), 5, color, -1)
            cv2.putText(vis, str(tid), (int(kp.x), int(kp.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return vis


class FeatureBank:
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.features: Dict[int, np.ndarray] = {}
        self.labels: Dict[int, Any] = {}
        self.access_counts: Dict[int, int] = defaultdict(int)

    def add(self, feature: np.ndarray, label: Any) -> int:
        if len(self.features) >= self.max_size:
            self._evict_lru()

        idx = len(self.features)
        self.features[idx] = feature
        self.labels[idx] = label
        self.access_counts[idx] = 0
        return idx

    def get(self, idx: int) -> Tuple[np.ndarray, Any]:
        self.access_counts[idx] += 1
        return self.features[idx], self.labels[idx]

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[int, float, Any]]:
        if not self.features:
            return []

        features = np.array(list(self.features.values()))
        dists = np.linalg.norm(features - query, axis=1)
        indices = np.argsort(dists)[:k]
        return [(list(self.features.keys())[i], float(dists[i]), self.labels[list(self.features.keys())[i]]) for i in indices]

    def _evict_lru(self):
        if not self.access_counts:
            return
        lru_idx = min(self.access_counts, key=self.access_counts.get)
        del self.features[lru_idx]
        del self.labels[lru_idx]
        del self.access_counts[lru_idx]


class FeatureBankNN:
    def __init__(self, feature_dim: int, max_size: int = 10000):
        self.feature_dim = feature_dim
        self.max_size = max_size
        self.index = None
        self.labels: Dict[int, Any] = {}
        self._build_index()

    def _build_index(self):
        try:
            import faiss
            self.index = faiss.IndexFlatL2(self.feature_dim)
        except ImportError:
            self.index = None

    def add(self, feature: np.ndarray, label: Any) -> int:
        if self.index is None:
            return -1
        idx = self.index.ntotal
        self.index.add(feature.reshape(1, -1).astype(np.float32))
        self.labels[idx] = label
        return idx

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[int, float, Any]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        distances, indices = self.index.search(query.reshape(1, -1).astype(np.float32), k)
        return [(int(indices[0][i]), float(distances[0][i]), self.labels.get(int(indices[0][i]))) for i in range(len(indices[0]))]


class FeatureVisualizer:
    @staticmethod
    def draw_keypoints(image: np.ndarray, keypoints: List, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        vis = image.copy()
        for kp in keypoints:
            cv2.circle(vis, (int(kp.x), int(kp.y)), int(kp.size/2), color, 2)
            if kp.angle != 0:
                end_x = int(kp.x + kp.size * math.cos(kp.angle))
                end_y = int(kp.y + kp.size * math.sin(kp.angle))
                cv2.line(vis, (int(kp.x), int(kp.y)), (end_x, end_y), color, 1)
        return vis

    @staticmethod
    def draw_matches(image1: np.ndarray, kp1: List, image2: np.ndarray, kp2: List, matches: List, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        vis[:h1, :w1] = image1
        vis[:h2, w1:w1+w2] = image2

        for match in matches:
            pt1 = (int(kp1[match.queryIdx].x), int(kp1[match.queryIdx].y))
            pt2 = (int(kp2[match.trainIdx].x + w1), int(kp2[match.trainIdx].y))
            cv2.line(vis, pt1, pt2, color, 1)
            cv2.circle(vis, pt1, 3, color, -1)
            cv2.circle(vis, pt2, 3, color, -1)

        return vis

    @staticmethod
    def draw_tracks(image: np.ndarray, keypoints: List, track_ids: List[int], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        vis = image.copy()
        for kp, tid in zip(keypoints, track_ids):
            cv2.circle(vis, (int(kp.x), int(kp.y)), 5, color, -1)
            cv2.putText(vis, str(tid), (int(kp.x), int(kp.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return vis
