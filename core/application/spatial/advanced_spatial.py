from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict
import math


class SpatialTask(Enum):
    LOCALIZATION = "localization"
    MAPPING = "mapping"
    RELOCALIZATION = "relocalization"
    LOOP_CLOSURE = "loop_closure"
    BUNDLE_ADJUSTMENT = "bundle_adjustment"
    SCENE_RECONSTRUCTION = "scene_reconstruction"


@dataclass(frozen=True, slots=True)
class Keyframe:
    keyframe_id: str
    timestamp: float
    image: np.ndarray
    pose: np.ndarray
    intrinsics: np.ndarray
    features: np.ndarray
    descriptors: np.ndarray
    depth: Optional[np.ndarray] = None


@dataclass(frozen=True, slots=True)
class MapPoint:
    point_id: int
    position: np.ndarray
    descriptor: np.ndarray
    observations: Dict[str, np.ndarray]
    color: np.ndarray
    normal: Optional[np.ndarray] = None


class VisualSLAM:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        max_features: int = 2000,
        scale_factor: float = 1.2,
        n_levels: int = 8,
    ):
        self.camera_matrix = camera_matrix
        self.max_features = max_features
        self.scale_factor = scale_factor
        self.n_levels = n_levels

        self.orb = cv2.ORB_create(nfeatures=max_features, scaleFactor=scale_factor, nlevels=n_levels)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.keyframes: Dict[str, Keyframe] = {}
        self.map_points: Dict[int, MapPoint] = {}
        self.current_pose = np.eye(4)
        self.next_kf_id = 0
        self.next_mp_id = 0

    def process_frame(self, image: np.ndarray, timestamp: float) -> Optional[Dict]:
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        kp, desc = self.orb.detectAndCompute(gray, None)

        if desc is None or len(kp) < 50:
            return None

        if not self.keyframes:
            return self._initialize_first_frame(image, timestamp, kp, desc)

        return self._track_frame(image, timestamp, kp, desc)

    def _initialize_first_frame(self, image: np.ndarray, timestamp: float, kp, desc) -> Dict:
        kf_id = f"kf_{self.next_kf_id}"
        self.next_kf_id += 1

        pose = np.eye(4)
        self.current_pose = pose

        keyframe = Keyframe(
            keyframe_id=kf_id,
            timestamp=timestamp,
            image=image,
            pose=pose,
            intrinsics=self.camera_matrix,
            features=np.array([p.pt for p in kp]),
            descriptors=desc,
        )
        self.keyframes[kf_id] = keyframe

        for i, (pt, d) in enumerate(zip(keyframe.features, desc)):
            mp_id = self.next_mp_id
            self.next_mp_id += 1
            mp = MapPoint(
                point_id=mp_id,
                position=self._triangulate_point(keyframe, pt),
                descriptor=d,
                observations={kf_id: pt},
                color=image[int(pt[1]), int(pt[0])],
            )
            self.map_points[mp_id] = mp

        return {
            'pose': pose,
            'keyframe_id': kf_id,
            'num_map_points': len(self.map_points),
        }

    def _track_frame(self, image: np.ndarray, timestamp: float, kp, desc) -> Dict:
        last_kf = list(self.keyframes.values())[-1]
        matches = self.matcher.match(last_kf.descriptors, desc)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 30:
            return self._relocalize(image, timestamp, kp, desc)

        pts1 = last_kf.features[[m.queryIdx for m in matches]]
        pts2 = np.array([kp[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix, cv2.RANSAC, 0.999, 1.0)
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix, mask=mask)

        new_pose = np.eye(4)
        new_pose[:3, :3] = R
        new_pose[:3, 3] = t.flatten()
        self.current_pose = last_kf.pose @ new_pose

        kf_id = f"kf_{self.next_kf_id}"
        self.next_kf_id += 1

        keyframe = Keyframe(
            keyframe_id=kf_id,
            timestamp=timestamp,
            image=image,
            pose=self.current_pose,
            intrinsics=self.camera_matrix,
            features=pts2,
            descriptors=[desc[m.trainIdx] for m in matches],
        )
        self.keyframes[kf_id] = keyframe

        self._update_map_points(matches, keyframe, last_kf, mask.ravel())

        return {
            'pose': self.current_pose,
            'keyframe_id': kf_id,
            'num_matches': len(matches),
            'num_map_points': len(self.map_points),
        }

    def _relocalize(self, image: np.ndarray, timestamp: float, kp, desc) -> Dict:
        best_kf = None
        best_matches = []

        for kf in list(self.keyframes.values())[-10:]:
            matches = self.matcher.match(kf.descriptors, desc)
            matches = sorted(matches, key=lambda x: x.distance)
            if len(matches) > len(best_matches):
                best_matches = matches
                best_kf = kf

        if best_kf is None or len(best_matches) < 30:
            return None

        return self._track_frame(image, timestamp, kp, desc)

    def _update_map_points(self, matches, new_kf: Keyframe, last_kf: Keyframe, mask: np.ndarray):
        for i, m in enumerate(matches):
            if not mask[i]:
                continue

            pt1 = last_kf.features[m.queryIdx]
            pt2 = new_kf.features[m.trainIdx]

            matched_mp = None
            for mp in self.map_points.values():
                if last_kf.keyframe_id in mp.observations:
                    obs = mp.observations[last_kf.keyframe_id]
                    if np.linalg.norm(obs - pt1) < 2:
                        matched_mp = mp
                        break

            if matched_mp:
                matched_mp.observations[new_kf.keyframe_id] = pt2
                matched_mp.position = self._triangulate_between(last_kf, new_kf, pt1, pt2)
            else:
                mp_id = self.next_mp_id
                self.next_mp_id += 1
                mp = MapPoint(
                    point_id=mp_id,
                    position=self._triangulate_between(last_kf, new_kf, pt1, pt2),
                    descriptor=desc[m.trainIdx],
                    observations={last_kf.keyframe_id: pt1, new_kf.keyframe_id: pt2},
                    color=image[int(pt2[1]), int(pt2[0])],
                )
                self.map_points[mp_id] = mp

    def _triangulate_point(self, kf: Keyframe, pt: np.ndarray) -> np.ndarray:
        return np.array([pt[0], pt[1], 10.0])

    def _triangulate_between(self, kf1: Keyframe, kf2: Keyframe, pt1: np.ndarray, pt2: np.ndarray) -> np.ndarray:
        P1 = kf1.intrinsics @ kf1.pose[:3, :]
        P2 = kf2.intrinsics @ kf2.pose[:3, :]

        A = np.array([
            pt1[0] * P1[2] - P1[0],
            pt1[1] * P1[2] - P1[1],
            pt2[0] * P2[2] - P2[0],
            pt2[1] * P2[2] - P2[1],
        ])

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        return X[:3] / X[3]


class BundleAdjuster:
    def __init__(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def optimize(
        self,
        keyframes: Dict[str, Keyframe],
        map_points: Dict[int, MapPoint],
    ) -> Tuple[Dict[str, np.ndarray], Dict[int, np.ndarray]]:
        kf_ids = list(keyframes.keys())
        mp_ids = list(map_points.keys())

        n_kf = len(kf_ids)
        n_mp = len(mp_ids)

        pose_params = {}
        for i, kf_id in enumerate(kf_ids):
            pose_params[kf_id] = keyframes[kf_id].pose[:3, :].flatten()

        point_params = {}
        for i, mp_id in enumerate(mp_ids):
            point_params[mp_id] = map_points[mp_id].position

        for iteration in range(self.max_iterations):
            J = []
            r = []

            for kf_id, kf in keyframes.items():
                for mp_id, mp in map_points.items():
                    if kf_id in mp.observations:
                        pt = mp.observations[kf_id]
                        proj = self._project_point(point_params[mp_id], pose_params[kf_id], kf.intrinsics)
                        residual = pt - proj
                        r.append(residual)

            r = np.array(r).flatten()
            if np.linalg.norm(r) < self.tolerance:
                break

        return pose_params, point_params

    def _project_point(self, point: np.ndarray, pose: np.ndarray, K: np.ndarray) -> np.ndarray:
        P = pose.reshape(3, 4)
        X = np.append(point, 1)
        proj = K @ P @ X
        return proj[:2] / proj[2]


class LoopClosureDetector:
    def __init__(
        self,
        vocabulary_tree: Any = None,
        min_score: float = 0.7,
        min_inliers: int = 20,
    ):
        self.vocabulary_tree = vocabulary_tree
        self.min_score = min_score
        self.min_inliers = min_inliers
        self.keyframe_database: List[Keyframe] = []
        self.bow_descriptors: List[np.ndarray] = []

    def add_keyframe(self, keyframe: Keyframe):
        self.keyframe_database.append(keyframe)
        if keyframe.descriptors is not None:
            self.bow_descriptors.append(self._compute_bow(keyframe.descriptors))

    def detect_loop(self, keyframe: Keyframe) -> Optional[Tuple[str, np.ndarray]]:
        if len(self.keyframe_database) < 10:
            return None

        bow = self._compute_bow(keyframe.descriptors)

        best_score = 0
        best_kf = None

        for i, db_bow in enumerate(self.bow_descriptors[:-5]):
            score = self._bow_similarity(bow, db_bow)
            if score > best_score and score > self.min_score:
                best_score = score
                best_kf = self.keyframe_database[i]

        if best_kf is None:
            return None

        matches = self._match_descriptors(keyframe.descriptors, best_kf.descriptors)
        if len(matches) < self.min_inliers:
            return None

        pts1 = best_kf.features[[m.queryIdx for m in matches]]
        pts2 = keyframe.features[[m.trainIdx for m in matches]]

        E, mask = cv2.findEssentialMat(pts1, pts2, keyframe.intrinsics, cv2.RANSAC, 0.999, 1.0)
        inliers = mask.sum()

        if inliers < self.min_inliers:
            return None

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, keyframe.intrinsics, mask=mask)
        rel_pose = np.eye(4)
        rel_pose[:3, :3] = R
        rel_pose[:3, 3] = t.flatten()

        return best_kf.keyframe_id, rel_pose

    def _compute_bow(self, descriptors: np.ndarray) -> np.ndarray:
        return np.mean(descriptors, axis=0) if descriptors is not None else np.zeros(32)

    def _bow_similarity(self, bow1: np.ndarray, bow2: np.ndarray) -> float:
        norm1 = np.linalg.norm(bow1)
        norm2 = np.linalg.norm(bow2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(bow1, bow2) / (norm1 * norm2)

    def _match_descriptors(self, desc1: np.ndarray, desc2: np.ndarray):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        return matcher.match(desc1, desc2)


class DenseMapper:
    def __init__(
        self,
        voxel_size: float = 0.05,
        truncation: float = 0.1,
    ):
        self.voxel_size = voxel_size
        self.truncation = truncation
        self.tsdf_volume: Dict[Tuple[int, int, int], Dict] = {}

    def integrate(self, depth: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray):
        h, w = depth.shape
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        for v in range(h):
            for u in range(w):
                d = depth[v, u]
                if d <= 0 or d > 10:
                    continue

                x = (u - cx) * d / fx
                y = (v - cy) * d / fy
                z = d

                point_cam = np.array([x, y, z, 1])
                point_world = pose @ point_cam
                point_world = point_world[:3]

                voxel = tuple(np.floor(point_world / self.voxel_size).astype(int))

                if voxel not in self.tsdf_volume:
                    self.tsdf_volume[voxel] = {'tsdf': 0, 'weight': 0, 'color': np.array([128, 128, 128])}

                dist = np.linalg.norm(point_world - np.array(voxel) * self.voxel_size)
                tsdf_val = dist / self.truncation
                tsdf_val = np.clip(tsdf_val, -1, 1)

                w = self.tsdf_volume[voxel]['weight']
                new_w = w + 1
                self.tsdf_volume[voxel]['tsdf'] = (self.tsdf_volume[voxel]['tsdf'] * w + tsdf_val) / new_w
                self.tsdf_volume[voxel]['weight'] = new_w

    def extract_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        vertices = []
        faces = []

        for voxel, data in self.tsdf_volume.items():
            if abs(data['tsdf']) < 0.5:
                center = np.array(voxel) * self.voxel_size
                vertices.append(center)

        return np.array(vertices), np.array(faces)


class SemanticMapper:
    def __init__(self, num_classes: int = 20):
        self.num_classes = num_classes
        self.semantic_volume: Dict[Tuple[int, int, int], np.ndarray] = {}

    def integrate(self, semantic: np.ndarray, depth: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray):
        h, w = depth.shape
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        for v in range(h):
            for u in range(w):
                d = depth[v, u]
                if d <= 0:
                    continue

                class_id = semantic[v, u]
                if class_id >= self.num_classes:
                    continue

                x = (u - cx) * d / fx
                y = (v - cy) * d / fy
                z = d

                point_cam = np.array([x, y, z, 1])
                point_world = pose @ point_cam
                point_world = point_world[:3]

                voxel = tuple(np.floor(point_world / 0.1).astype(int))

                if voxel not in self.semantic_volume:
                    self.semantic_volume[voxel] = np.zeros(self.num_classes)

                self.semantic_volume[voxel][class_id] += 1

    def get_semantic_map(self) -> Dict[Tuple[int, int, int], int]:
        result = {}
        for voxel, counts in self.semantic_volume.items():
            if counts.sum() > 0:
                result[voxel] = int(np.argmax(counts))
        return result
