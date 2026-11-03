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
import heapq


class SLAMAlgorithm(Enum):
    EKF_SLAM = "ekf_slam"
    FAST_SLAM = "fast_slam"
    GRAPH_SLAM = "graph_slam"
    POSE_GRAPH = "pose_graph"
    LOOP_CLOSURE = "loop_closure"
    VISUAL_SLAM = "visual_slam"
    LIDAR_SLAM = "lidar_slam"
    RGBD_SLAM = "rgbd_slam"
    ORB_SLAM = "orb_slam"
    LSD_SLAM = "lsd_slam"
    DSO = "dso"
    SVO = "svo"


@dataclass(frozen=True, slots=True)
class Pose3D:
    rotation: np.ndarray
    translation: np.ndarray
    timestamp: float = 0.0
    covariance: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.rotation.shape != (3, 3):
            raise ValueError("Rotation must be 3x3")
        if self.translation.shape != (3,):
            raise ValueError("Translation must be 3D")

    @property
    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T

    def inverse(self) -> "Pose3D":
        R_inv = self.rotation.T
        t_inv = -R_inv @ self.translation
        return Pose3D(R_inv, t_inv, self.timestamp, self.covariance)

    def compose(self, other: "Pose3D") -> "Pose3D":
        R = self.rotation @ other.rotation
        t = self.rotation @ other.translation + self.translation
        return Pose3D(R, t, other.timestamp)

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        return self.rotation @ point + self.translation

    def distance_to(self, other: "Pose3D") -> float:
        return np.linalg.norm(self.translation - other.translation)

    def angle_to(self, other: "Pose3D") -> float:
        R_rel = self.rotation.T @ other.rotation
        angle = math.acos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
        return angle


@dataclass(frozen=True, slots=True)
class Landmark:
    landmark_id: int
    position: np.ndarray
    covariance: np.ndarray
    observations: int = 0
    descriptors: Optional[np.ndarray] = None
    timestamp: float = 0.0


@dataclass(frozen=True, slots=True)
class Keyframe:
    keyframe_id: str
    timestamp: float
    image: np.ndarray
    pose: Pose3D
    features: np.ndarray
    descriptors: np.ndarray
    depth: Optional[np.ndarray] = None
    map_points: Dict[int, np.ndarray] = field(default_factory=dict)


class SLAMBackend(ABC):
    @abstractmethod
    def add_pose(self, pose: Pose3D) -> int:
        pass

    @abstractmethod
    def add_landmark(self, landmark: Landmark) -> int:
        pass

    @abstractmethod
    def add_constraint(self, pose_idx: int, landmark_idx: int, measurement: np.ndarray, information: np.ndarray):
        pass

    @abstractmethod
    def optimize(self, max_iterations: int = 50) -> bool:
        pass

    @abstractmethod
    def get_poses(self) -> List[Pose3D]:
        pass

    @abstractmethod
    def get_landmarks(self) -> List[Landmark]:
        pass


class PoseGraphSLAM(SLAMBackend):
    def __init__(self):
        self.poses: List[Pose3D] = []
        self.landmarks: List[Landmark] = []
        self.constraints: List[tuple] = []
        self._optimized_poses: Optional[List[Pose3D]] = None

    def add_pose(self, pose: Pose3D) -> int:
        idx = len(self.poses)
        self.poses.append(pose)
        return idx

    def add_landmark(self, landmark: Landmark) -> int:
        idx = len(self.landmarks)
        self.landmarks.append(landmark)
        return idx

    def add_constraint(
        self,
        pose_idx: int,
        landmark_idx: int,
        measurement: np.ndarray,
        information: np.ndarray,
    ):
        self.constraints.append((pose_idx, landmark_idx, measurement, information))

    def add_pose_constraint(
        self,
        from_idx: int,
        to_idx: int,
        relative_pose: Pose3D,
        information: np.ndarray,
    ):
        self.constraints.append(("pose", from_idx, to_idx, relative_pose, information))

    def optimize(self, max_iterations: int = 50) -> bool:
        if len(self.poses) < 2:
            return False

        n = len(self.poses)
        dim = 6

        H = np.zeros((n * dim, n * dim))
        b = np.zeros(n * dim)

        for constraint in self.constraints:
            if constraint[0] == "pose":
                _, from_idx, to_idx, rel_pose, info = constraint
                self._add_pose_constraint(H, b, from_idx, to_idx, rel_pose, info, dim)
            else:
                pose_idx, landmark_idx, measurement, info = constraint
                self._add_landmark_constraint(H, b, pose_idx, landmark_idx, measurement, info, dim)

        try:
            dx = np.linalg.solve(H + 1e-6 * np.eye(n * dim), b)
        except np.linalg.LinAlgError:
            return False

        self._optimized_poses = []
        for i in range(n):
            xi = dx[i*dim:(i+1)*dim]
            pose = self.poses[i]
            new_pose = self._apply_increment(pose, xi)
            self._optimized_poses.append(new_pose)

        return True

    def _add_pose_constraint(
        self,
        H: np.ndarray,
        b: np.ndarray,
        from_idx: int,
        to_idx: int,
        rel_pose: Pose3D,
        info: np.ndarray,
        dim: int,
    ):
        pose_from = self._optimized_poses[from_idx] if self._optimized_poses else self.poses[from_idx]
        pose_to = self._optimized_poses[to_idx] if self._optimized_poses else self.poses[to_idx]

        predicted = pose_from.inverse().compose(pose_to)
        error = self._pose_error(predicted, rel_pose)

        J_from = self._pose_jacobian(pose_from, pose_to)
        J_to = -J_from

        idx_from = from_idx * dim
        idx_to = to_idx * dim

        H[idx_from:idx_from+dim, idx_from:idx_from+dim] += J_from.T @ info @ J_from
        H[idx_to:idx_to+dim, idx_to:idx_to+dim] += J_to.T @ info @ J_to
        H[idx_from:idx_from+dim, idx_to:idx_to+dim] += J_from.T @ info @ J_to
        H[idx_to:idx_to+dim, idx_from:idx_from+dim] += J_to.T @ info @ J_from

        b[idx_from:idx_from+dim] -= J_from.T @ info @ error
        b[idx_to:idx_to+dim] -= J_to.T @ info @ error

    def _add_landmark_constraint(
        self,
        H: np.ndarray,
        b: np.ndarray,
        pose_idx: int,
        landmark_idx: int,
        measurement: np.ndarray,
        info: np.ndarray,
        dim: int,
    ):
        pass

    def _pose_error(self, predicted: Pose3D, measured: Pose3D) -> np.ndarray:
        R_err = predicted.rotation.T @ measured.rotation
        angle = math.acos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        axis = np.array([R_err[2,1]-R_err[1,2], R_err[0,2]-R_err[2,0], R_err[1,0]-R_err[0,1]])
        if np.linalg.norm(axis) > 1e-6:
            axis = axis / np.linalg.norm(axis) * angle
        t_err = predicted.translation - measured.translation
        return np.concatenate([axis, t_err])

    def _pose_jacobian(self, pose_from: Pose3D, pose_to: Pose3D) -> np.ndarray:
        J = np.zeros((6, 6))
        J[:3, :3] = -np.eye(3)
        J[:3, 3:] = self._skew(pose_to.translation - pose_from.translation)
        J[3:, 3:] = -np.eye(3)
        return J

    def _skew(self, v: np.ndarray) -> np.ndarray:
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])

    def _apply_increment(self, pose: Pose3D, xi: np.ndarray) -> Pose3D:
        omega = xi[:3]
        v = xi[3:]

        theta = np.linalg.norm(omega)
        if theta < 1e-6:
            R_delta = np.eye(3) + self._skew(omega)
            t_delta = v
        else:
            axis = omega / theta
            K = self._skew(axis)
            R_delta = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K
            t_delta = (np.eye(3) * theta + (1 - math.cos(theta)) * K + (theta - math.sin(theta)) * K @ K) @ v / theta

        new_R = pose.rotation @ R_delta
        new_t = pose.rotation @ t_delta + pose.translation
        return Pose3D(new_R, new_t, pose.timestamp, pose.covariance)

    def get_poses(self) -> List[Pose3D]:
        return self._optimized_poses if self._optimized_poses is not None else self.poses

    def get_landmarks(self) -> List[Landmark]:
        return self.landmarks


class FastSLAM:
    def __init__(
        self,
        num_particles: int = 50,
        motion_noise: float = 0.1,
        measurement_noise: float = 0.1,
        resample_threshold: float = 0.5,
    ):
        self.num_particles = num_particles
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
        self.resample_threshold = resample_threshold

        self.particles: List[Dict] = []
        self.weights = np.ones(num_particles) / num_particles

        for _ in range(num_particles):
            self.particles.append({
                'pose': Pose3D(np.eye(3), np.zeros(3)),
                'landmarks': {},
                'weight': 1.0 / num_particles,
            })

    def predict(self, motion: Pose3D):
        for i, particle in enumerate(self.particles):
            noise = np.random.randn(6) * self.motion_noise
            noisy_motion = self._add_noise(motion, noise)
            particle['pose'] = particle['pose'].compose(noisy_motion)

    def update(self, measurements: List[Tuple[int, np.ndarray, np.ndarray]]):
        for i, particle in enumerate(self.particles):
            weight = 1.0
            for landmark_id, measurement, info in measurements:
                if landmark_id in particle['landmarks']:
                    landmark = particle['landmarks'][landmark_id]
                    predicted = self._predict_measurement(particle['pose'], landmark)
                    error = measurement - predicted
                    weight *= self._gaussian_likelihood(error, info)
                else:
                    particle['landmarks'][landmark_id] = {
                        'mean': measurement,
                        'cov': np.linalg.inv(info),
                        'observations': 1,
                    }
                    weight *= 1.0
            self.weights[i] = weight

        self.weights = self.weights / (self.weights.sum() + 1e-10)

        if 1.0 / np.sum(self.weights**2) < self.resample_threshold * self.num_particles:
            self._resample()

    def _add_noise(self, pose: Pose3D, noise: np.ndarray) -> Pose3D:
        omega = noise[:3]
        v = noise[3:]

        theta = np.linalg.norm(omega)
        if theta < 1e-6:
            R_delta = np.eye(3) + self._skew(omega)
            t_delta = v
        else:
            axis = omega / theta
            K = self._skew(axis)
            R_delta = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K
            t_delta = v

        new_R = pose.rotation @ R_delta
        new_t = pose.rotation @ t_delta + pose.translation
        return Pose3D(new_R, new_t)

    def _predict_measurement(self, pose: Pose3D, landmark: Dict) -> np.ndarray:
        rel_pos = pose.inverse().transform_point(landmark['mean'])
        return rel_pos

    def _gaussian_likelihood(self, error: np.ndarray, info: np.ndarray) -> float:
        try:
            det = np.linalg.det(info)
            if det <= 0:
                return 1e-10
            return np.exp(-0.5 * error.T @ info @ error) / np.sqrt((2*np.pi)**3 * det)
        except:
            return 1e-10

    def _resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        new_particles = []
        for idx in indices:
            particle = self.particles[idx]
            new_particles.append({
                'pose': particle['pose'],
                'landmarks': {k: v.copy() for k, v in particle['landmarks'].items()},
                'weight': 1.0 / self.num_particles,
            })
        self.particles = new_particles
        self.weights = np.ones(self.num_particles) / self.num_particles

    def _skew(self, v: np.ndarray) -> np.ndarray:
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])

    def get_best_pose(self) -> Pose3D:
        best_idx = np.argmax(self.weights)
        return self.particles[best_idx]['pose']

    def get_best_landmarks(self) -> Dict:
        best_idx = np.argmax(self.weights)
        return self.particles[best_idx]['landmarks']


class GraphSLAM:
    def __init__(self):
        self.nodes: Dict[int, Pose3D] = {}
        self.edges: List[tuple] = []
        self.landmarks: Dict[int, Landmark] = {}
        self.next_node_id = 0
        self.next_landmark_id = 0

    def add_node(self, pose: Pose3D) -> int:
        node_id = self.next_node_id
        self.nodes[node_id] = pose
        self.next_node_id += 1
        return node_id

    def add_landmark(self, position: np.ndarray, covariance: np.ndarray) -> int:
        lm_id = self.next_landmark_id
        self.landmarks[lm_id] = Landmark(lm_id, position, covariance)
        self.next_landmark_id += 1
        return lm_id

    def add_edge(self, from_id: int, to_id: int, relative_pose: Pose3D, information: np.ndarray):
        self.edges.append((from_id, to_id, relative_pose, information))

    def add_landmark_observation(self, node_id: int, landmark_id: int, measurement: np.ndarray, information: np.ndarray):
        self.edges.append(("landmark", node_id, landmark_id, measurement, information))

    def optimize(self, max_iterations: int = 20) -> bool:
        if len(self.nodes) < 2:
            return False

        node_ids = sorted(self.nodes.keys())
        n = len(node_ids)
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        dim = 6

        for _ in range(max_iterations):
            H = np.zeros((n * dim, n * dim))
            b = np.zeros(n * dim)

            for edge in self.edges:
                if edge[0] == "landmark":
                    _, node_id, lm_id, measurement, info = edge
                    if lm_id not in self.landmarks:
                        continue
                    landmark = self.landmarks[lm_id]
                    self._build_landmark_jacobian(H, b, id_to_idx[node_id], landmark, measurement, info, dim)
                else:
                    from_id, to_id, rel_pose, info = edge
                    self._build_pose_jacobian(H, b, id_to_idx[from_id], id_to_idx[to_id], rel_pose, info, dim)

            try:
                dx = np.linalg.solve(H + 1e-6 * np.eye(n * dim), b)
            except np.linalg.LinAlgError:
                return False

            for i, node_id in enumerate(node_ids):
                xi = dx[i*dim:(i+1)*dim]
                self.nodes[node_id] = self._apply_increment(self.nodes[node_id], xi)

        return True

    def _build_pose_jacobian(self, H, b, from_idx, to_idx, rel_pose, info, dim):
        pose_from = self.nodes[list(self.nodes.keys())[from_idx]]
        pose_to = self.nodes[list(self.nodes.keys())[to_idx]]

        predicted = pose_from.inverse().compose(pose_to)
        error = self._pose_error(predicted, rel_pose)

        J_from = self._pose_jacobian(pose_from, pose_to)
        J_to = -J_from

        H[from_idx*dim:(from_idx+1)*dim, from_idx*dim:(from_idx+1)*dim] += J_from.T @ info @ J_from
        H[to_idx*dim:(to_idx+1)*dim, to_idx*dim:(to_idx+1)*dim] += J_to.T @ info @ J_to
        H[from_idx*dim:(from_idx+1)*dim, to_idx*dim:(to_idx+1)*dim] += J_from.T @ info @ J_to
        H[to_idx*dim:(to_idx+1)*dim, from_idx*dim:(from_idx+1)*dim] += J_to.T @ info @ J_from

        b[from_idx*dim:(from_idx+1)*dim] -= J_from.T @ info @ error
        b[to_idx*dim:(to_idx+1)*dim] -= J_to.T @ info @ error

    def _build_landmark_jacobian(self, H, b, node_idx, landmark, measurement, info, dim):
        pose = self.nodes[list(self.nodes.keys())[node_idx]]
        predicted = pose.inverse().transform_point(landmark.position)
        error = measurement - predicted

        J = np.zeros((3, dim))
        J[:3, 3:] = -pose.rotation.T

        H[node_idx*dim:(node_idx+1)*dim, node_idx*dim:(node_idx+1)*dim] += J.T @ info @ J
        b[node_idx*dim:(node_idx+1)*dim] -= J.T @ info @ error

    def _pose_error(self, predicted: Pose3D, measured: Pose3D) -> np.ndarray:
        R_err = predicted.rotation.T @ measured.rotation
        angle = math.acos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        axis = np.array([R_err[2,1]-R_err[1,2], R_err[0,2]-R_err[2,0], R_err[1,0]-R_err[0,1]])
        if np.linalg.norm(axis) > 1e-6:
            axis = axis / np.linalg.norm(axis) * angle
        t_err = predicted.translation - measured.translation
        return np.concatenate([axis, t_err])

    def _pose_jacobian(self, pose_from: Pose3D, pose_to: Pose3D) -> np.ndarray:
        J = np.zeros((6, 6))
        J[:3, :3] = -np.eye(3)
        J[:3, 3:] = self._skew(pose_to.translation - pose_from.translation)
        J[3:, 3:] = -np.eye(3)
        return J

    def _skew(self, v: np.ndarray) -> np.ndarray:
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])

    def _apply_increment(self, pose: Pose3D, xi: np.ndarray) -> Pose3D:
        omega = xi[:3]
        v = xi[3:]

        theta = np.linalg.norm(omega)
        if theta < 1e-6:
            R_delta = np.eye(3) + self._skew(omega)
            t_delta = v
        else:
            axis = omega / theta
            K = self._skew(axis)
            R_delta = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K
            t_delta = (np.eye(3) * theta + (1 - math.cos(theta)) * K + (theta - math.sin(theta)) * K @ K) @ v / theta

        new_R = pose.rotation @ R_delta
        new_t = pose.rotation @ t_delta + pose.translation
        return Pose3D(new_R, new_t, pose.timestamp, pose.covariance)

    def get_trajectory(self) -> List[Pose3D]:
        return [self.nodes[i] for i in sorted(self.nodes.keys())]

    def get_map(self) -> Dict[int, Landmark]:
        return self.landmarks


class LoopClosureDetector:
    def __init__(
        self,
        descriptor_type: str = "orb",
        match_threshold: float = 0.7,
        min_inliers: int = 20,
        ransac_threshold: float = 3.0,
    ):
        self.descriptor_type = descriptor_type
        self.match_threshold = match_threshold
        self.min_inliers = min_inliers
        self.ransac_threshold = ransac_threshold
        self.keyframes: List[Dict] = []
        self._detector = self._create_detector()

    def _create_detector(self):
        if self.descriptor_type == "orb":
            return cv2.ORB_create(nfeatures=1000)
        elif self.descriptor_type == "sift":
            return cv2.SIFT_create()
        elif self.descriptor_type == "surf":
            return cv2.xfeatures2d.SURF_create()
        else:
            return cv2.ORB_create()

    def add_keyframe(self, image: np.ndarray, pose: Pose3D, node_id: int):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        kp, desc = self._detector.detectAndCompute(gray, None)

        self.keyframes.append({
            'node_id': node_id,
            'pose': pose,
            'image': image,
            'keypoints': kp,
            'descriptors': desc,
        })

    def detect_loop(self, current_image: np.ndarray, current_pose: Pose3D, current_node_id: int) -> Optional[Tuple[int, Pose3D]]:
        if current_image.dtype != np.uint8:
            current_image = (current_image * 255).astype(np.uint8)
        gray = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY) if len(current_image.shape) == 3 else current_image
        kp_cur, desc_cur = self._detector.detectAndCompute(gray, None)

        if desc_cur is None or len(self.keyframes) < 10:
            return None

        best_match = None
        best_score = 0

        for i, kf in enumerate(self.keyframes[:-5]):
            if kf['descriptors'] is None:
                continue

            matches = self._match_descriptors(desc_cur, kf['descriptors'])
            if len(matches) < self.min_inliers:
                continue

            inliers, rel_pose = self._verify_geometry(kp_cur, kf['keypoints'], matches)
            if inliers >= self.min_inliers:
                score = inliers / len(matches)
                if score > best_score and score > self.match_threshold:
                    best_score = score
                    best_match = (kf['node_id'], rel_pose)

        return best_match

    def _match_descriptors(self, desc1: np.ndarray, desc2: np.ndarray):
        if self.descriptor_type == "orb":
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        matches = matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches[:100]

    def _verify_geometry(self, kp1, kp2, matches):
        if len(matches) < 8:
            return 0, None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        try:
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, self.ransac_threshold, 0.99)
            inliers = np.sum(mask)
            return int(inliers), None
        except:
            return 0, None


class VisualOdometry:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        detector_type: str = "orb",
        matcher_type: str = "bf",
        min_features: int = 1000,
        ransac_threshold: float = 1.0,
    ):
        self.camera_matrix = camera_matrix
        self.detector_type = detector_type
        self.min_features = min_features
        self.ransac_threshold = ransac_threshold
        self._detector = self._create_detector()
        self._matcher = self._create_matcher(matcher_type)
        self.prev_image: Optional[np.ndarray] = None
        self.prev_kp: Optional[list] = None
        self.prev_desc: Optional[np.ndarray] = None
        self.current_pose = Pose3D(np.eye(3), np.zeros(3))
        self.trajectory: List[Pose3D] = [self.current_pose]

    def _create_detector(self):
        if self.detector_type == "orb":
            return cv2.ORB_create(nfeatures=self.min_features)
        elif self.detector_type == "sift":
            return cv2.SIFT_create(nfeatures=self.min_features)
        else:
            return cv2.ORB_create(nfeatures=self.min_features)

    def _create_matcher(self, matcher_type: str):
        if matcher_type == "bf":
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif matcher_type == "flann":
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        else:
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def process_frame(self, image: np.ndarray, timestamp: float = 0.0) -> Optional[Pose3D]:
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        kp, desc = self._detector.detectAndCompute(gray, None)

        if self.prev_image is None or desc is None or self.prev_desc is None:
            self.prev_image = image
            self.prev_kp = kp
            self.prev_desc = desc
            return self.current_pose

        matches = self._matcher.match(self.prev_desc, desc)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 8:
            self.prev_image = image
            self.prev_kp = kp
            self.prev_desc = desc
            return self.current_pose

        pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in matches[:100]])
        pts2 = np.float32([kp[m.trainIdx].pt for m in matches[:100]])

        try:
            E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix, cv2.RANSAC, 0.999, self.ransac_threshold)
            _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.camera_matrix, mask=mask)

            relative_pose = Pose3D(R, t.flatten(), timestamp)
            self.current_pose = self.current_pose.compose(relative_pose)
            self.trajectory.append(self.current_pose)

        except cv2.error:
            pass

        self.prev_image = image
        self.prev_kp = kp
        self.prev_desc = desc

        return self.current_pose

    def get_trajectory(self) -> List[Pose3D]:
        return self.trajectory


class SLAMSystem:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        config: Dict[str, Any] = None,
    ):
        self.config = config or {}
        self.visual_odometry = VisualOdometry(camera_matrix)
        self.slam_backend = PoseGraphSLAM()
        self.loop_detector = LoopClosureDetector()
        self.loop_closing_slam = LoopClosingSLAM(self.slam_backend, self.loop_detector)
        self.keyframes: List[Keyframe] = []
        self.next_kf_id = 0

    def process_frame(self, image: np.ndarray, timestamp: float = 0.0) -> Optional[Pose3D]:
        pose = self.visual_odometry.process_frame(image, timestamp)

        if pose and (len(self.keyframes) == 0 or self._should_create_keyframe(pose)):
            self._create_keyframe(image, pose, timestamp)

        return pose

    def _should_create_keyframe(self, pose: Pose3D) -> bool:
        if not self.keyframes:
            return True

        last_kf = self.keyframes[-1]
        dist = pose.distance_to(last_kf.pose)
        angle = pose.angle_to(last_kf.pose)

        return dist > 1.0 or angle > 0.2

    def _create_keyframe(self, image: np.ndarray, pose: Pose3D, timestamp: float):
        kf_id = f"kf_{self.next_kf_id}"
        self.next_kf_id += 1

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        kp, desc = cv2.ORB_create().detectAndCompute(gray, None)

        keyframe = Keyframe(
            keyframe_id=kf_id,
            timestamp=timestamp,
            image=image,
            pose=pose,
            intrinsics=np.eye(3),
            features=np.array([p.pt for p in kp]) if kp else np.array([]),
            descriptors=desc if desc is not None else np.array([]),
        )
        self.keyframes.append(keyframe)

        self.loop_closing_slam.process_keyframe(image, pose, len(self.keyframes) - 1)

    def get_trajectory(self) -> List[Pose3D]:
        return self.visual_odometry.get_trajectory()

    def get_optimized_trajectory(self) -> List[Pose3D]:
        return self.slam_backend.get_poses()

    def optimize(self):
        self.slam_backend.optimize()
        return self.slam_backend.get_poses()


class SLAMBenchmark:
    @staticmethod
    def evaluate_trajectory(estimated: List[Pose3D], ground_truth: List[Pose3D]) -> Dict[str, float]:
        if len(estimated) != len(ground_truth):
            return {'ate_rmse': float('inf'), 'rpe_rmse': float('inf')}

        ate_errors = [e.distance_to(g) for e, g in zip(estimated, ground_truth)]
        ate_rmse = math.sqrt(np.mean(np.array(ate_errors)**2))

        rpe_errors = []
        for i in range(1, len(estimated)):
            rel_est = estimated[i-1].inverse().compose(estimated[i])
            rel_gt = ground_truth[i-1].inverse().compose(ground_truth[i])
            rpe_errors.append(rel_est.distance_to(rel_gt))

        rpe_rmse = math.sqrt(np.mean(np.array(rpe_errors)**2)) if rpe_errors else 0

        return {
            'ate_rmse': ate_rmse,
            'rpe_rmse': rpe_rmse,
            'ate_mean': np.mean(ate_errors),
            'ate_max': np.max(ate_errors),
        }

    @staticmethod
    def compute_absolute_trajectory_error(estimated: List[Pose3D], ground_truth: List[Pose3D]) -> np.ndarray:
        return np.array([e.distance_to(g) for e, g in zip(estimated, ground_truth)])

    @staticmethod
    def compute_relative_pose_error(estimated: List[Pose3D], ground_truth: List[Pose3D]) -> np.ndarray:
        errors = []
        for i in range(1, len(estimated)):
            rel_est = estimated[i-1].inverse().compose(estimated[i])
            rel_gt = ground_truth[i-1].inverse().compose(ground_truth[i])
            errors.append(rel_est.distance_to(rel_gt))
        return np.array(errors)

    @staticmethod
    def align_trajectories(estimated: List[Pose3D], ground_truth: List[Pose3D]) -> Tuple[List[Pose3D], np.ndarray, np.ndarray]:
        est_pts = np.array([p.translation for p in estimated])
        gt_pts = np.array([p.translation for p in ground_truth])

        centroid_est = np.mean(est_pts, axis=0)
        centroid_gt = np.mean(gt_pts, axis=0)

        centered_est = est_pts - centroid_est
        centered_gt = gt_pts - centroid_gt

        H = centered_est.T @ centered_gt
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_gt - R @ centroid_est

        aligned = []
        for p in estimated:
            new_t = R @ p.translation + t
            aligned.append(Pose3D(p.rotation, new_t, p.timestamp, p.covariance))

        return aligned, R, t
