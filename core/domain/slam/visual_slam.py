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


class VisualSLAMType(Enum):
    MONOCULAR = "monocular"
    STEREO = "stereo"
    RGBD = "rgbd"
    EVENT = "event"
    FISHEYE = "fisheye"
    MULTICAM = "multicam"


class TrackerType(Enum):
    KLT = "klt"
    ORB = "orb"
    KAZE = "kaze"
    AKAZE = "akaze"
    SUPERPOINT = "superpoint"
    D2NET = "d2net"
    R2D2 = "r2d2"
    DISK = "disk"
    LEARNED = "learned"


class MappingType(Enum):
    SPARSE = "sparse"
    DENSE = "dense"
    SEMI_DENSE = "semi_dense"
    SURFEL = "surfel"
    VOXEL = "voxel"
    TSdf = "tsdf"
    MESH = "mesh"


@dataclass(frozen=True, slots=True)
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: np.ndarray = field(default_factory=lambda: np.zeros(5))

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                        [0, self.fy, self.cy],
                        [0, 0, 1]], dtype=np.float32)

    @property
    def inv_K(self) -> np.ndarray:
        return np.linalg.inv(self.K)


@dataclass(frozen=True, slots=True)
class Keyframe:
    id: int
    timestamp: float
    pose: np.ndarray
    image: np.ndarray
    keypoints: List[Any] = field(default_factory=list)
    descriptors: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    covisibility: Dict[int, int] = field(default_factory=dict)
    mn_covisibility: Dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MapPoint:
    id: int
    position: np.ndarray
    normal: np.ndarray
    descriptor: Optional[np.ndarray] = None
    observations: Dict[int, Tuple[int, np.ndarray]] = field(default_factory=dict)
    ref_keyframe: Optional[int] = None
    mn_observations: int = 0
    mb_bad: bool = False


@dataclass(frozen=True, slots=True)
class Frame:
    id: int
    timestamp: float
    image: np.ndarray
    keypoints: List[Any] = field(default_factory=list)
    descriptors: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    is_keyframe: bool = False
    reference_keyframe: Optional[int] = None


class FeatureTracker:
    def __init__(
        self,
        tracker_type: TrackerType = TrackerType.ORB,
        num_features: int = 2000,
        scale_factor: float = 1.2,
        num_levels: int = 8
    ):
        self.tracker_type = tracker_type
        self.num_features = num_features
        self._create_detector()

    def _create_detector(self):
        if self.tracker_type == TrackerType.ORB:
            self.detector = cv2.ORB_create(nfeatures=self.num_features)
        elif self.tracker_type == TrackerType.KAZE:
            self.detector = cv2.KAZE_create()
        elif self.tracker_type == TrackerType.AKAZE:
            self.detector = cv2.AKAZE_create()
        elif self.tracker_type == TrackerType.SIFT:
            self.detector = cv2.SIFT_create(nfeatures=self.num_features)
        else:
            self.detector = cv2.ORB_create(nfeatures=self.num_features)

    def extract(self, image: np.ndarray) -> Tuple[List[Any], Optional[np.ndarray]]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return self.detector.detectAndCompute(gray, None)

    def match(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        if desc1 is None or desc2 is None:
            return []

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(desc1, desc2)
        return sorted(matches, key=lambda x: x.distance)

    def track(
        self,
        prev_kp: List[Any],
        prev_desc: np.ndarray,
        curr_image: np.ndarray
    ) -> Tuple[List[Any], Optional[np.ndarray], List[cv2.DMatch]]:
        curr_kp, curr_desc = self.extract(curr_image)

        if curr_desc is None or prev_desc is None:
            return curr_kp, curr_desc, []

        matches = self.match(prev_desc, curr_desc)

        tracked_kp = []
        tracked_desc = []
        tracked_matches = []

        for i, match in enumerate(matches):
            if i >= 1000:
                break
            tracked_kp.append(curr_kp[match.trainIdx])
            if curr_desc is not None:
                tracked_desc.append(curr_desc[match.trainIdx])
            tracked_matches.append(match)

        return tracked_kp, np.array(tracked_desc) if tracked_desc else None, tracked_matches


class PoseEstimator:
    def __init__(self, camera_model: CameraModel):
        self.camera_model = camera_model

    def estimate_pose_pnp(
        self,
        points_3d: np.ndarray,
        points_2d: np.ndarray,
        ransac: bool = True
    ) -> Tuple[Optional[np.ndarray], int]:
        if len(points_3d) < 4:
            return None, 0

        if ransac:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d.astype(np.float32),
                points_2d.astype(np.float32),
                self.camera_model.K,
                self.camera_model.distortion,
                iterationsCount=100,
                reprojectionError=4.0,
                confidence=0.99,
                flags=cv2.SOLVEPNP_EPNP
            )
            inlier_count = len(inliers) if inliers is not None else 0
        else:
            success, rvec, tvec = cv2.solvePnP(
                points_3d.astype(np.float32),
                points_2d.astype(np.float32),
                self.camera_model.K,
                self.camera_model.distortion,
                flags=cv2.SOLVEPNP_EPNP
            )
            inlier_count = len(points_3d)

        if not success:
            return None, 0

        R = cv2.Rodrigues(rvec)[0]
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = tvec.flatten()

        return pose, inlier_count

    def estimate_pose_epnp(
        self,
        points_3d: np.ndarray,
        points_2d: np.ndarray
    ) -> Tuple[Optional[np.ndarray], int]:
        return self.estimate_pose_pnp(points_3d, points_2d, ransac=False)


class Initializer:
    def __init__(
        self,
        camera_model: CameraModel,
        min_features: int = 100,
        min_parallax: float = 10.0
    ):
        self.camera_model = camera_model
        self.min_features = min_features
        self.min_parallax = min_parallax

    def initialize(
        self,
        frame1: Frame,
        frame2: Frame,
        matches: List[cv2.DMatch]
    ) -> Tuple[Optional[np.ndarray], List[MapPoint], float]:
        if len(matches) < self.min_features:
            return None, [], 0.0

        pts1 = np.float32([frame1.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.float32([frame2.keypoints[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_model.K, cv2.RANSAC, 0.999, 1.0)

        if E is None:
            return None, [], 0.0

        inliers = mask.ravel().astype(bool)
        pts1_in = pts1[inliers]
        pts2_in = pts2[inliers]

        _, R, t, mask = cv2.recoverPose(E, pts1_in, pts2_in, self.camera_model.K)
        inliers = mask.ravel().astype(bool)
        pts1_in = pts1_in[inliers]
        pts2_in = pts2_in[inliers]

        if len(pts1_in) < self.min_features:
            return None, [], 0.0

        parallax = np.mean(np.linalg.norm(pts1_in - pts2_in, axis=1))
        if parallax < self.min_parallax:
            return None, [], 0.0

        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t.flatten()

        points_3d = self._triangulate(pts1_in, pts2_in, pose)

        map_points = []
        for i, pt3d in enumerate(points_3d):
            if pt3d[2] > 0:
                mp = MapPoint(
                    id=i,
                    position=pt3d,
                    normal=np.array([0, 0, 1]),
                    observations={
                        frame1.id: (matches[i].queryIdx, pts1_in[i]),
                        frame2.id: (matches[i].trainIdx, pts2_in[i])
                    }
                )
                map_points.append(mp)

        return pose, map_points, parallax

    def _triangulate(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        pose2: np.ndarray
    ) -> np.ndarray:
        P1 = self.camera_model.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.camera_model.K @ pose2[:3, :]

        points_3d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_3d[:3] / points_3d[3]
        return points_3d.T


class LocalMapper:
    def __init__(
        self,
        camera_model: CameraModel,
        min_obs: int = 3
    ):
        self.camera_model = camera_model
        self.min_obs = min_obs
        self.local_keyframes: List[Keyframe] = []
        self.local_map_points: Dict[int, MapPoint] = {}

    def add_keyframe(self, keyframe: Keyframe):
        self.local_keyframes.append(keyframe)
        if len(self.local_keyframes) > 10:
            self.local_keyframes.pop(0)

    def add_map_point(self, map_point: MapPoint):
        self.local_map_points[map_point.id] = map_point

    def update_map_points(self):
        for mp in self.local_map_points.values():
            if mp.mn_observations < self.min_obs:
                mp.mb_bad = True

    def create_new_map_points(self, current_keyframe: Keyframe, frames: List[Frame]):
        for frame in frames:
            if frame.id == current_keyframe.id:
                continue

            matches = self._match_keyframes(current_keyframe, frame)
            if len(matches) < 10:
                continue

            self._triangulate_points(current_keyframe, frame, matches)

    def _match_keyframes(self, kf1: Keyframe, kf2: Frame) -> List[cv2.DMatch]:
        if kf1.descriptors is None or kf2.descriptors is None:
            return []

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(kf1.descriptors, kf2.descriptors)
        return sorted(matches, key=lambda x: x.distance)[:200]

    def _triangulate_points(self, kf1: Keyframe, kf2: Frame, matches: List[cv2.DMatch]):
        pts1 = np.float32([kf1.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kf2.keypoints[m.trainIdx].pt for m in matches])

        points_3d = cv2.triangulatePoints(
            self.camera_model.K @ np.hstack([np.eye(3), np.zeros((3, 1))]),
            self.camera_model.K @ kf1.pose[:3],
            pts1.T, pts2.T
        )
        points_3d = points_3d[:3] / points_3d[3]

        for i, pt in enumerate(points_3d.T):
            if pt[2] > 0:
                mp = MapPoint(
                    id=len(self.local_map_points),
                    position=pt,
                    normal=np.array([0, 0, 1]),
                    observations={
                        kf1.id: (matches[i].queryIdx, pts1[i]),
                        kf2.id: (matches[i].trainIdx, pts2[i])
                    },
                    ref_keyframe=kf1.id
                )
                self.local_map_points[mp.id] = mp


class LoopCloser:
    def __init__(
        self,
        camera_model: CameraModel,
        min_score: float = 0.15,
        min_inliers: int = 20
    ):
        self.camera_model = camera_model
        self.min_score = min_score
        self.min_inliers = min_inliers
        self.candidate_pairs: List[Tuple[int, int]] = []

    def detect_loops(
        self,
        current_keyframe: Keyframe,
        keyframes: Dict[int, Keyframe]
    ) -> List[Tuple[int, int, np.ndarray]]:
        candidates = []

        for kf_id, kf in keyframes.items():
            if kf_id == current_keyframe.id:
                continue
            if abs(kf.timestamp - current_keyframe.timestamp) < 10:
                continue

            if kf.descriptors is None or current_keyframe.descriptors is None:
                continue

            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(current_keyframe.descriptors, kf.descriptors)

            if len(matches) < self.min_inliers:
                continue

            pts1 = np.float32([current_keyframe.keypoints[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kf.keypoints[m.trainIdx].pt for m in matches])

            E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_model.K, cv2.RANSAC, 0.999, 1.0)

            if E is None:
                continue

            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.camera_model.K, mask=mask)
            inliers = int(np.sum(mask))

            if inliers >= self.min_inliers:
                score = inliers / len(matches)
                if score >= self.min_score:
                    transform = np.eye(4)
                    transform[:3, :3] = R
                    transform[:3, 3] = t.flatten()
                    candidates.append((current_keyframe.id, kf_id, transform))

        return candidates


class VisualSLAM:
    def __init__(
        self,
        camera_model: CameraModel,
        config: Dict = None
    ):
        self.camera_model = camera_model
        self.config = config or {}

        self.tracker = FeatureTracker(
            tracker_type=TrackerType(self.config.get('tracker_type', 'orb')),
            num_features=self.config.get('num_features', 2000)
        )

        self.pose_estimator = PoseEstimator(camera_model)
        self.initializer = Initializer(camera_model)
        self.local_mapper = LocalMapper(camera_model)
        self.loop_closer = LoopCloser(camera_model)

        self.frames: Dict[int, Frame] = {}
        self.keyframes: Dict[int, Keyframe] = {}
        self.map_points: Dict[int, MapPoint] = {}

        self.current_frame: Optional[Frame] = None
        self.last_keyframe: Optional[Keyframe] = None
        self.frame_id = 0
        self.keyframe_id = 0
        self.map_point_id = 0

        self.state = "INITIALIZING"
        self.reference_frame: Optional[Frame] = None

    def track(self, image: np.ndarray, timestamp: float) -> Optional[np.ndarray]:
        self.frame_id += 1
        frame = Frame(
            id=self.frame_id,
            timestamp=timestamp,
            image=image
        )

        frame.keypoints, frame.descriptors = self.tracker.extract(image)

        if self.state == "INITIALIZING":
            return self._initialize(frame)

        if self.state == "TRACKING":
            return self._track_local(frame)

        return None

    def _initialize(self, frame: Frame) -> Optional[np.ndarray]:
        if self.reference_frame is None:
            self.reference_frame = frame
            return None

        matches = self.tracker.match(
            self.reference_frame.descriptors, frame.descriptors
        )

        pose, map_points, parallax = self.initializer.initialize(
            self.reference_frame, frame, matches
        )

        if pose is not None:
            self._create_initial_map(map_points)
            self.state = "TRACKING"
            self.reference_frame = None
            return pose

        return None

    def _track_local(self, frame: Frame) -> Optional[np.ndarray]:
        if self.last_keyframe is None:
            return None

        tracked_kp, tracked_desc, matches = self.tracker.track(
            self.last_keyframe.keypoints,
            self.last_keyframe.descriptors,
            frame.image
        )

        if len(matches) < 50:
            return None

        frame.keypoints = tracked_kp
        frame.descriptors = tracked_desc

        points_3d = []
        points_2d = []

        for match in matches:
            mp_id = self._find_map_point(match.queryIdx)
            if mp_id is not None:
                points_3d.append(self.map_points[mp_id].position)
                points_2d.append(frame.keypoints[match.trainIdx].pt)

        if len(points_3d) < 10:
            return None

        pose, inliers = self.pose_estimator.estimate_pose_pnp(
            np.array(points_3d), np.array(points_2d)
        )

        if pose is not None:
            frame.pose = pose
            frame.is_keyframe = self._is_keyframe(inliers, len(matches))

            if frame.is_keyframe:
                self._create_keyframe(frame)

        return pose

    def _is_keyframe(self, inliers: int, total_matches: int) -> bool:
        if total_matches == 0:
            return False
        return inliers / total_matches < 0.9

    def _create_initial_map(self, map_points: List[MapPoint]):
        for mp in map_points:
            mp.id = self.map_point_id
            self.map_point_id += 1
            self.map_points[mp.id] = mp
            self.local_mapper.add_map_point(mp)

        if self.reference_frame:
            kf = self._frame_to_keyframe(self.reference_frame)
            self.keyframes[kf.id] = kf
            self.last_keyframe = kf
            self.local_mapper.add_keyframe(kf)

    def _create_keyframe(self, frame: Frame):
        self.keyframe_id += 1
        kf = Keyframe(
            id=self.keyframe_id,
            timestamp=frame.timestamp,
            pose=frame.pose,
            image=frame.image,
            keypoints=frame.keypoints,
            descriptors=frame.descriptors
        )
        self.keyframes[kf.id] = kf
        self.last_keyframe = kf
        self.local_mapper.add_keyframe(kf)

        self.local_mapper.create_new_map_points(kf, list(self.frames.values())[-10:])

    def _frame_to_keyframe(self, frame: Frame) -> Keyframe:
        self.keyframe_id += 1
        return Keyframe(
            id=self.keyframe_id,
            timestamp=frame.timestamp,
            pose=frame.pose,
            image=frame.image,
            keypoints=frame.keypoints,
            descriptors=frame.descriptors
        )

    def _find_map_point(self, query_idx: int) -> Optional[int]:
        for mp in self.map_points.values():
            if self.last_keyframe and self.last_keyframe.id in mp.observations:
                if mp.observations[self.last_keyframe.id][0] == query_idx:
                    return mp.id
        return None


class VisualSLAMPipeline:
    def __init__(
        self,
        camera_model: CameraModel,
        config: Dict = None
    ):
        self.slam = VisualSLAM(camera_model, config)
        self.trajectory: List[np.ndarray] = []

    def process_frame(self, image: np.ndarray, timestamp: float) -> Optional[np.ndarray]:
        pose = self.slam.track(image, timestamp)

        if pose is not None:
            self.trajectory.append(pose)

        return pose

    def get_map(self) -> Tuple[Dict[int, Keyframe], Dict[int, MapPoint]]:
        return self.slam.keyframes, self.slam.map_points

    def get_trajectory(self) -> List[np.ndarray]:
        return self.trajectory


class VisualSLAMMetrics:
    @staticmethod
    def ate_rmse(
        estimated: List[np.ndarray],
        ground_truth: List[np.ndarray]
    ) -> float:
        if len(estimated) != len(ground_truth):
            return float('inf')

        errors = []
        for est, gt in zip(estimated, ground_truth):
            trans_error = np.linalg.norm(est[:3, 3] - gt[:3, 3])
            errors.append(trans_error)

        return float(np.sqrt(np.mean(np.array(errors)**2)))

    @staticmethod
    def rpe_rmse(
        estimated: List[np.ndarray],
        ground_truth: List[np.ndarray]
    ) -> Tuple[float, float]:
        if len(estimated) < 2:
            return 0.0, 0.0

        trans_errors = []
        rot_errors = []

        for i in range(1, len(estimated)):
            est_rel = np.linalg.inv(estimated[i-1]) @ estimated[i]
            gt_rel = np.linalg.inv(ground_truth[i-1]) @ ground_truth[i]

            trans_error = np.linalg.norm(est_rel[:3, 3] - gt_rel[:3, 3])
            rot_est = cv2.Rodrigues(est_rel[:3, :3])[0]
            rot_gt = cv2.Rodrigues(gt_rel[:3, :3])[0]
            rot_error = np.linalg.norm(rot_est - rot_gt)

            trans_errors.append(trans_error)
            rot_errors.append(rot_error)

        return float(np.sqrt(np.mean(np.array(trans_errors)**2))), float(np.sqrt(np.mean(np.array(rot_errors)**2)))


class VisualSLAMVisualizer:
    @staticmethod
    def draw_trajectory(
        trajectory: List[np.ndarray],
        map_points: Dict[int, MapPoint],
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if not trajectory:
            return vis

        positions = np.array([pose[:3, 3] for pose in trajectory])
        min_coords = np.min(positions[:, :2], axis=0)
        max_coords = np.max(positions[:, :2], axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1

        scale = min(image_size[0], image_size[1]) * 0.8 / max(range_coords)

        prev_pt = None
        for pose in trajectory:
            x = int((pose[0, 3] - min_coords[0]) * scale + image_size[0] * 0.1)
            y = int(image_size[1] - (pose[1, 3] - min_coords[1]) * scale - image_size[1] * 0.1)

            if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                if prev_pt is not None:
                    cv2.line(vis, prev_pt, (x, y), (0, 255, 0), 1)
                prev_pt = (x, y)

        for mp in map_points.values():
            x = int((mp.position[0] - min_coords[0]) * scale + image_size[0] * 0.1)
            y = int(image_size[1] - (mp.position[1] - min_coords[1]) * scale - image_size[1] * 0.1)

            if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                cv2.circle(vis, (x, y), 1, (255, 0, 0), -1)

        return vis


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet