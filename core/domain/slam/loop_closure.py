from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict, deque
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize, least_squares
from scipy.linalg import pinv, svd, cholesky, block_diag
from scipy.sparse import csr_matrix, lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, lsqr, cg
import warnings

class LoopClosureMethod(Enum):
    BOW = "bag_of_words"
    GEOMETRIC = "geometric"
    SEMANTIC = "semantic"
    APPEARANCE = "appearance"
    SCAN_CONTEXT = "scan_context"
    MULTI_MODAL = "multi_modal"

class VocabularyType(Enum):
    ORB = "orb"
    SIFT = "sift"
    BRISK = "brisk"
    AKAZE = "akaze"
    SUPERPOINT = "superpoint"
    D2NET = "d2net"
    R2D2 = "r2d2"
    DISK = "disk"
    ALIKED = "aliked"

@dataclass(frozen=True, slots=True)
class KeyFrame:
    id: int
    timestamp: float
    pose: np.ndarray
    descriptors: np.ndarray
    keypoints: np.ndarray
    scores: np.ndarray
    bow_vector: np.ndarray
    feature_vector: np.ndarray
    scan: Optional[np.ndarray] = None
    semantic_labels: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class LoopCandidate:
    query_id: int
    match_id: int
    score: float
    transform: np.ndarray
    inliers: int
    method: LoopClosureMethod
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class LoopClosureResult:
    candidates: List[LoopCandidate]
    verified_loops: List[LoopCandidate]
    rejected_loops: List[LoopCandidate]
    processing_time: float
    statistics: Dict[str, Any] = field(default_factory=dict)

class VocabularyTree:
    def __init__(self, branching_factor: int = 10, depth: int = 6, descriptor_size: int = 32):
        self.k = branching_factor
        self.L = depth
        self.descriptor_size = descriptor_size
        self.nodes: List[VocabularyNode] = []
        self.word_id_counter = 0
        self.root: Optional[VocabularyNode] = None
        self.word_to_node: Dict[int, VocabularyNode] = {}
        self.trained = False

    def train(self, descriptors: np.ndarray, max_iterations: int = 100):
        if len(descriptors) == 0:
            return

        self.root = VocabularyNode(0, np.mean(descriptors, axis=0), 0)
        self.nodes = [self.root]
        self._hierarchical_kmeans(self.root, descriptors, 0, max_iterations)
        self._assign_word_ids(self.root)
        self.trained = True

    def _hierarchical_kmeans(self, node: VocabularyNode, descriptors: np.ndarray,
                             level: int, max_iterations: int):
        if level >= self.L or len(descriptors) < self.k:
            node.is_leaf = True
            return

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(self.k, len(descriptors)),
                        max_iter=max_iterations, n_init=1, random_state=42)
        labels = kmeans.fit_predict(descriptors)
        centroids = kmeans.cluster_centers_

        for i, centroid in enumerate(centroids):
            child_descriptors = descriptors[labels == i]
            if len(child_descriptors) == 0:
                continue

            child = VocabularyNode(len(self.nodes), centroid, level + 1)
            self.nodes.append(child)
            node.children.append(child)
            self._hierarchical_kmeans(child, child_descriptors, level + 1, max_iterations)

    def _assign_word_ids(self, node: VocabularyNode):
        if node.is_leaf:
            node.word_id = self.word_id_counter
            self.word_to_node[self.word_id_counter] = node
            self.word_id_counter += 1
        else:
            for child in node.children:
                self._assign_word_ids(child)

    def transform(self, descriptors: np.ndarray) -> np.ndarray:
        bow = np.zeros(self.word_id_counter)
        for desc in descriptors:
            node = self.root
            while not node.is_leaf and node.children:
                distances = [np.linalg.norm(desc - child.centroid) for child in node.children]
                node = node.children[np.argmin(distances)]
            if node.word_id is not None:
                bow[node.word_id] += 1

        if np.sum(bow) > 0:
            bow = bow / np.sum(bow)
        return bow

    def score(self, bow1: np.ndarray, bow2: np.ndarray) -> float:
        intersection = np.sum(np.minimum(bow1, bow2))
        union = np.sum(np.maximum(bow1, bow2))
        return intersection / union if union > 0 else 0.0

    def save(self, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'VocabularyTree':
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

@dataclass
class VocabularyNode:
    id: int
    centroid: np.ndarray
    level: int
    children: List['VocabularyNode'] = field(default_factory=list)
    is_leaf: bool = False
    word_id: Optional[int] = None

class DBoW2:
    def __init__(self, vocabulary: VocabularyTree = None, k: int = 10, L: int = 6):
        self.vocabulary = vocabulary or VocabularyTree(k, L)
        self.database: Dict[int, KeyFrame] = {}
        self.direct_index: Dict[int, List[int]] = defaultdict(list)
        self.inverted_index: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.max_database_size = 10000

    def add_keyframe(self, kf: KeyFrame):
        self.database[kf.id] = kf

        for word_id, weight in enumerate(kf.bow_vector):
            if weight > 0:
                self.inverted_index[word_id].append((kf.id, weight))
                self.direct_index[kf.id].append(word_id)

        if len(self.database) > self.max_database_size:
            oldest_id = min(self.database.keys())
            self.remove_keyframe(oldest_id)

    def remove_keyframe(self, kf_id: int):
        if kf_id in self.database:
            for word_id in self.direct_index[kf_id]:
                self.inverted_index[word_id] = [(id_, w) for id_, w in self.inverted_index[word_id] if id_ != kf_id]
            del self.direct_index[kf_id]
            del self.database[kf_id]

    def query(self, bow_vector: np.ndarray, num_results: int = 10,
              min_score: float = 0.01) -> List[Tuple[int, float]]:
        candidates: Dict[int, float] = defaultdict(float)

        for word_id, weight in enumerate(bow_vector):
            if weight > 0:
                for db_id, db_weight in self.inverted_index.get(word_id, []):
                    candidates[db_id] += weight * db_weight

        results = [(id_, score) for id_, score in candidates.items() if score >= min_score]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:num_results]

    def score(self, id1: int, id2: int) -> float:
        if id1 not in self.database or id2 not in self.database:
            return 0.0
        return self.vocabulary.score(self.database[id1].bow_vector, self.database[id2].bow_vector)

class ScanContext:
    def __init__(self, num_sectors: int = 60, num_rings: int = 20,
                 max_range: float = 80.0, min_range: float = 0.5):
        self.num_sectors = num_sectors
        self.num_rings = num_rings
        self.max_range = max_range
        self.min_range = min_range
        self.descriptors: Dict[int, np.ndarray] = {}
        self.keyframes: Dict[int, KeyFrame] = {}

    def compute_descriptor(self, scan: np.ndarray) -> np.ndarray:
        if len(scan) == 0:
            return np.zeros((self.num_rings, self.num_sectors))

        ranges = np.linalg.norm(scan[:, :2], axis=1)
        angles = np.arctan2(scan[:, 1], scan[:, 0])

        valid = (ranges >= self.min_range) & (ranges <= self.max_range)
        ranges = ranges[valid]
        angles = angles[valid]
        heights = scan[valid, 2]

        descriptor = np.zeros((self.num_rings, self.num_sectors))
        ring_size = (self.max_range - self.min_range) / self.num_rings
        sector_size = 2 * math.pi / self.num_sectors

        for r, a, h in zip(ranges, angles, heights):
            ring_idx = int((r - self.min_range) / ring_size)
            sector_idx = int((a + math.pi) / sector_size)

            ring_idx = np.clip(ring_idx, 0, self.num_rings - 1)
            sector_idx = np.clip(sector_idx, 0, self.num_sectors - 1)

            descriptor[ring_idx, sector_idx] = max(descriptor[ring_idx, sector_idx], h)

        return descriptor

    def add_keyframe(self, kf: KeyFrame):
        if kf.scan is not None:
            descriptor = self.compute_descriptor(kf.scan)
            self.descriptors[kf.id] = descriptor
            self.keyframes[kf.id] = kf

    def match(self, query_descriptor: np.ndarray, num_candidates: int = 10,
              exclude_recent: int = 50) -> List[Tuple[int, float, int]]:
        scores = []

        for kf_id, desc in self.descriptors.items():
            if kf_id in self.keyframes:
                kf = self.keyframes[kf_id]
                if abs(kf.id - list(self.keyframes.keys())[-1]) < exclude_recent:
                    continue

            score, yaw_diff = self._distance(query_descriptor, desc)
            scores.append((kf_id, score, yaw_diff))

        scores.sort(key=lambda x: x[1])
        return scores[:num_candidates]

    def _distance(self, desc1: np.ndarray, desc2: np.ndarray) -> Tuple[float, int]:
        min_dist = float('inf')
        best_yaw = 0

        for yaw_shift in range(self.num_sectors):
            shifted = np.roll(desc2, yaw_shift, axis=1)
            diff = np.abs(desc1 - shifted)
            dist = np.sum(diff) / (self.num_rings * self.num_sectors)

            if dist < min_dist:
                min_dist = dist
                best_yaw = yaw_shift

        return min_dist, best_yaw

    def get_yaw_angle(self, yaw_idx: int) -> float:
        return yaw_idx * 2 * math.pi / self.num_sectors

class GeometricVerifier:
    def __init__(self, max_distance: float = 0.1, max_angle: float = 0.1,
                 min_inliers: int = 10, ransac_iterations: int = 1000):
        self.max_distance = max_distance
        self.max_angle = max_angle
        self.min_inliers = min_inliers
        self.ransac_iterations = ransac_iterations

    def verify(self, kf1: KeyFrame, kf2: KeyFrame) -> Optional[LoopCandidate]:
        matches = self._match_descriptors(kf1.descriptors, kf2.descriptors)

        if len(matches) < self.min_inliers * 2:
            return None

        best_transform = None
        best_inliers = 0
        best_inlier_mask = None

        for _ in range(self.ransac_iterations):
            if len(matches) < 3:
                break

            idx = np.random.choice(len(matches), 3, replace=False)
            sample_matches = matches[idx]

            transform = self._estimate_transform(kf1, kf2, sample_matches)
            if transform is None:
                continue

            inlier_mask = self._compute_inliers(kf1, kf2, matches, transform)
            num_inliers = np.sum(inlier_mask)

            if num_inliers > best_inliers:
                best_inliers = num_inliers
                best_transform = transform
                best_inlier_mask = inlier_mask

        if best_inliers >= self.min_inliers and best_transform is not None:
            transform = self._refine_transform(kf1, kf2, matches, best_inlier_mask)
            return LoopCandidate(
                query_id=kf1.id,
                match_id=kf2.id,
                score=best_inliers / len(matches),
                transform=transform,
                inliers=best_inliers,
                method=LoopClosureMethod.GEOMETRIC
            )

        return None

    def _match_descriptors(self, desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
        if len(desc1) == 0 or len(desc2) == 0:
            return np.empty((0, 2), dtype=int)

        if desc1.shape[1] != desc2.shape[1]:
            return np.empty((0, 2), dtype=int)

        distances = np.linalg.norm(desc1[:, None] - desc2[None, :], axis=2)
        matches = []

        for i in range(len(desc1)):
            sorted_idx = np.argsort(distances[i])
            if len(sorted_idx) >= 2:
                best = sorted_idx[0]
                second_best = sorted_idx[1]
                if distances[i, best] < 0.75 * distances[i, second_best]:
                    matches.append([i, best])

        return np.array(matches, dtype=int)

    def _estimate_transform(self, kf1: KeyFrame, kf2: KeyFrame,
                            matches: np.ndarray) -> Optional[np.ndarray]:
        if len(matches) < 3:
            return None

        pts1 = kf1.keypoints[matches[:, 0]]
        pts2 = kf2.keypoints[matches[:, 1]]

        if pts1.shape[1] == 2:
            pts1 = np.hstack([pts1, np.ones((len(pts1), 1))])
            pts2 = np.hstack([pts2, np.ones((len(pts2), 1))])

        A = []
        for p1, p2 in zip(pts1, pts2):
            A.append([p2[0]*p1[0], p2[0]*p1[1], p2[0]*p1[2],
                      p2[1]*p1[0], p2[1]*p1[1], p2[1]*p1[2],
                      p2[2]*p1[0], p2[2]*p1[1], p2[2]*p1[2]])

        A = np.array(A)
        _, _, Vt = svd(A)
        F = Vt[-1].reshape(3, 3)

        U, _, Vt = svd(F)
        F = U @ np.diag([1, 1, 0]) @ Vt

        return F

    def _compute_inliers(self, kf1: KeyFrame, kf2: KeyFrame,
                         matches: np.ndarray, transform: np.ndarray) -> np.ndarray:
        pts1 = kf1.keypoints[matches[:, 0]]
        pts2 = kf2.keypoints[matches[:, 1]]

        if pts1.shape[1] == 2:
            pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
            pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])
        else:
            pts1_h = pts1
            pts2_h = pts2

        pts1_trans = (transform @ pts1_h.T).T
        pts1_trans = pts1_trans[:, :2] / (pts1_trans[:, 2:3] + 1e-8)

        distances = np.linalg.norm(pts1_trans - pts2[:, :2], axis=1)
        return distances < self.max_distance

    def _refine_transform(self, kf1: KeyFrame, kf2: KeyFrame,
                          matches: np.ndarray, inlier_mask: np.ndarray) -> np.ndarray:
        inlier_matches = matches[inlier_mask]
        if len(inlier_matches) < 3:
            return np.eye(3)

        pts1 = kf1.keypoints[inlier_matches[:, 0]]
        pts2 = kf2.keypoints[inlier_matches[:, 1]]

        if pts1.shape[1] == 2:
            pts1 = np.hstack([pts1, np.ones((len(pts1), 1))])
            pts2 = np.hstack([pts2, np.ones((len(pts2), 1))])

        centroid1 = np.mean(pts1, axis=0)
        centroid2 = np.mean(pts2, axis=0)

        pts1_centered = pts1 - centroid1
        pts2_centered = pts2 - centroid2

        H = pts2_centered.T @ pts1_centered
        U, _, Vt = svd(H)
        R_opt = Vt.T @ U.T

        if np.linalg.det(R_opt) < 0:
            Vt[-1, :] *= -1
            R_opt = Vt.T @ U.T

        t_opt = centroid2 - R_opt @ centroid1

        transform = np.eye(4)
        transform[:3, :3] = R_opt
        transform[:3, 3] = t_opt

        return transform

class SemanticVerifier:
    def __init__(self, min_overlap: float = 0.3, iou_threshold: float = 0.5):
        self.min_overlap = min_overlap
        self.iou_threshold = iou_threshold

    def verify(self, kf1: KeyFrame, kf2: KeyFrame) -> Optional[LoopCandidate]:
        if kf1.semantic_labels is None or kf2.semantic_labels is None:
            return None

        labels1 = kf1.semantic_labels
        labels2 = kf2.semantic_labels

        unique_labels = np.unique(np.concatenate([labels1, labels2]))
        overlap_scores = []

        for label in unique_labels:
            mask1 = labels1 == label
            mask2 = labels2 == label

            if np.sum(mask1) == 0 or np.sum(mask2) == 0:
                continue

            intersection = np.sum(mask1 & mask2)
            union = np.sum(mask1 | mask2)
            iou = intersection / union if union > 0 else 0
            overlap_scores.append(iou)

        if not overlap_scores:
            return None

        avg_overlap = np.mean(overlap_scores)
        if avg_overlap < self.min_overlap:
            return None

        return LoopCandidate(
            query_id=kf1.id,
            match_id=kf2.id,
            score=avg_overlap,
            transform=np.eye(4),
            inliers=int(avg_overlap * 100),
            method=LoopClosureMethod.SEMANTIC
        )

class LoopClosureDetector:
    def __init__(self, vocabulary: VocabularyTree = None,
                 geometric_verifier: GeometricVerifier = None,
                 semantic_verifier: SemanticVerifier = None,
                 scan_context: ScanContext = None,
                 min_score: float = 0.1,
                 min_geometric_inliers: int = 10,
                 temporal_window: int = 100,
                 num_candidates: int = 20):
        self.dbow = DBoW2(vocabulary)
        self.geometric_verifier = geometric_verifier or GeometricVerifier(min_inliers=min_geometric_inliers)
        self.semantic_verifier = semantic_verifier or SemanticVerifier()
        self.scan_context = scan_context or ScanContext()
        self.min_score = min_score
        self.temporal_window = temporal_window
        self.num_candidates = num_candidates
        self.keyframes: Dict[int, KeyFrame] = {}
        self.last_keyframe_id = -1

    def add_keyframe(self, kf: KeyFrame):
        self.keyframes[kf.id] = kf
        self.dbow.add_keyframe(kf)
        self.scan_context.add_keyframe(kf)
        self.last_keyframe_id = kf.id

    def detect_loops(self, query_kf: KeyFrame) -> LoopClosureResult:
        start_time = time.time()

        bow_candidates = self.dbow.query(query_kf.bow_vector, self.num_candidates, self.min_score)
        scan_candidates = self.scan_context.match(self.scan_context.compute_descriptor(query_kf.scan) if query_kf.scan is not None else np.zeros((20, 60)),
                                                   self.num_candidates)

        combined_candidates = self._combine_candidates(bow_candidates, scan_candidates)

        verified = []
        rejected = []

        for cand_id, score in combined_candidates:
            if cand_id == query_kf.id:
                continue
            if abs(cand_id - query_kf.id) < self.temporal_window:
                continue

            cand_kf = self.keyframes.get(cand_id)
            if cand_kf is None:
                continue

            geometric_result = self.geometric_verifier.verify(query_kf, cand_kf)
            if geometric_result:
                semantic_result = self.semantic_verifier.verify(query_kf, cand_kf)
                if semantic_result:
                    geometric_result.score = (geometric_result.score + semantic_result.score) / 2
                    geometric_result.method = LoopClosureMethod.MULTI_MODAL
                verified.append(geometric_result)
            else:
                rejected.append(LoopCandidate(
                    query_id=query_kf.id,
                    match_id=cand_id,
                    score=score,
                    transform=np.eye(4),
                    inliers=0,
                    method=LoopClosureMethod.BOW
                ))

        processing_time = time.time() - start_time

        return LoopClosureResult(
            candidates=[LoopCandidate(query_id=query_kf.id, match_id=c[0], score=c[1],
                                      transform=np.eye(4), inliers=0, method=LoopClosureMethod.BOW)
                        for c in combined_candidates],
            verified_loops=verified,
            rejected_loops=rejected,
            processing_time=processing_time,
            statistics={
                "num_bow_candidates": len(bow_candidates),
                "num_scan_candidates": len(scan_candidates),
                "num_verified": len(verified),
                "num_rejected": len(rejected)
            }
        )

    def _combine_candidates(self, bow_candidates: List[Tuple[int, float]],
                            scan_candidates: List[Tuple[int, float, int]]) -> List[Tuple[int, float]]:
        combined: Dict[int, float] = {}

        for id_, score in bow_candidates:
            combined[id_] = combined.get(id_, 0) + score * 0.6

        for id_, score, _ in scan_candidates:
            combined[id_] = combined.get(id_, 0) + (1 - score) * 0.4

        result = [(id_, score) for id_, score in combined.items()]
        result.sort(key=lambda x: x[1], reverse=True)
        return result[:self.num_candidates]

class PoseGraphOptimizer:
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6,
                 use_sparse: bool = True, robust_kernel: str = "huber",
                 kernel_threshold: float = 1.0):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.use_sparse = use_sparse
        self.robust_kernel = robust_kernel
        self.kernel_threshold = kernel_threshold

    def optimize(self, poses: Dict[int, np.ndarray],
                 edges: List[Tuple[int, int, np.ndarray, np.ndarray]],
                 fixed_nodes: Set[int] = None) -> Dict[int, np.ndarray]:
        node_ids = sorted(poses.keys())
        id_to_idx = {id_: i for i, id_ in enumerate(node_ids)}
        n = len(node_ids)

        if fixed_nodes is None:
            fixed_nodes = {node_ids[0]}

        for iter_num in range(self.max_iterations):
            H = lil_matrix((6 * n, 6 * n))
            b = np.zeros(6 * n)
            total_error = 0.0

            for from_id, to_id, measurement, info in edges:
                i = id_to_idx[from_id]
                j = id_to_idx[to_id]

                pose_i = poses[from_id]
                pose_j = poses[to_id]

                error = self._compute_error(pose_i, pose_j, measurement)
                J_i, J_j = self._compute_jacobians(pose_i, pose_j, measurement)

                weight = self._robust_weight(error, info)

                H_i = J_i.T @ info @ J_i * weight
                H_j = J_j.T @ info @ J_j * weight
                H_ij = J_i.T @ info @ J_j * weight

                b_i = J_i.T @ info @ error * weight
                b_j = J_j.T @ info @ error * weight

                H[6*i:6*i+6, 6*i:6*i+6] += H_i
                H[6*j:6*j+6, 6*j:6*j+6] += H_j
                H[6*i:6*i+6, 6*j:6*j+6] += H_ij
                H[6*j:6*j+6, 6*i:6*i+6] += H_ij.T

                b[6*i:6*i+6] += b_i
                b[6*j:6*j+6] += b_j

                total_error += error.T @ info @ error * weight

            for fixed_id in fixed_nodes:
                if fixed_id in id_to_idx:
                    idx = id_to_idx[fixed_id]
                    H[6*idx:6*idx+6, 6*idx:6*idx+6] += np.eye(6) * 1e6

            if self.use_sparse:
                H_csr = H.tocsr()
                dx = spsolve(H_csr, -b)
            else:
                H_dense = H.toarray()
                dx = np.linalg.solve(H_dense + 1e-6 * np.eye(6*n), -b)

            max_update = 0.0
            for id_ in node_ids:
                idx = id_to_idx[id_]
                update = dx[6*idx:6*idx+6]
                max_update = max(max_update, np.linalg.norm(update))
                poses[id_] = self._apply_update(poses[id_], update)

            if max_update < self.tolerance:
                break

        return poses

    def _compute_error(self, pose_i: np.ndarray, pose_j: np.ndarray,
                       measurement: np.ndarray) -> np.ndarray:
        rel_measured = measurement
        rel_estimated = np.linalg.inv(pose_i) @ pose_j
        error_transform = np.linalg.inv(rel_measured) @ rel_estimated
        return self._transform_to_vector(error_transform)

    def _compute_jacobians(self, pose_i: np.ndarray, pose_j: np.ndarray,
                           measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = 6
        J_i = np.zeros((6, 6))
        J_j = np.zeros((6, 6))

        eps = 1e-6
        for k in range(6):
            perturb = np.zeros(6)
            perturb[k] = eps

            pose_i_plus = self._apply_update(pose_i, perturb)
            pose_i_minus = self._apply_update(pose_i, -perturb)
            error_plus = self._compute_error(pose_i_plus, pose_j, measurement)
            error_minus = self._compute_error(pose_i_minus, pose_j, measurement)
            J_i[:, k] = (error_plus - error_minus) / (2 * eps)

            pose_j_plus = self._apply_update(pose_j, perturb)
            pose_j_minus = self._apply_update(pose_j, -perturb)
            error_plus = self._compute_error(pose_i, pose_j_plus, measurement)
            error_minus = self._compute_error(pose_i, pose_j_minus, measurement)
            J_j[:, k] = (error_plus - error_minus) / (2 * eps)

        return J_i, J_j

    def _transform_to_vector(self, T: np.ndarray) -> np.ndarray:
        trans = T[:3, 3]
        rot = R.from_matrix(T[:3, :3]).as_rotvec()
        return np.concatenate([trans, rot])

    def _apply_update(self, pose: np.ndarray, update: np.ndarray) -> np.ndarray:
        trans_update = update[:3]
        rot_update = update[3:]

        new_pose = pose.copy()
        new_pose[:3, 3] += trans_update

        if np.linalg.norm(rot_update) > 1e-10:
            rot_mat = R.from_rotvec(rot_update).as_matrix()
            new_pose[:3, :3] = rot_mat @ pose[:3, :3]

        return new_pose

    def _robust_weight(self, error: np.ndarray, info: np.ndarray) -> float:
        if self.robust_kernel == "none":
            return 1.0

        mahalanobis = error.T @ info @ error

        if self.robust_kernel == "huber":
            if mahalanobis <= self.kernel_threshold**2:
                return 1.0
            else:
                return self.kernel_threshold / math.sqrt(mahalanobis)
        elif self.robust_kernel == "cauchy":
            return 1.0 / (1.0 + mahalanobis / self.kernel_threshold**2)
        elif self.robust_kernel == "tukey":
            if mahalanobis <= self.kernel_threshold**2:
                return (1.0 - (mahalanobis / self.kernel_threshold**2))**2
            else:
                return 0.0

        return 1.0

class LoopClosurePipeline:
    def __init__(self, detector: LoopClosureDetector = None,
                 optimizer: PoseGraphOptimizer = None):
        self.detector = detector or LoopClosureDetector()
        self.optimizer = optimizer or PoseGraphOptimizer()
        self.poses: Dict[int, np.ndarray] = {}
        self.edges: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
        self.keyframes: Dict[int, KeyFrame] = {}
        self.loop_closures: List[LoopCandidate] = []

    def add_keyframe(self, kf: KeyFrame, odometry: np.ndarray = None):
        self.keyframes[kf.id] = kf
        self.detector.add_keyframe(kf)

        if kf.id == 0:
            self.poses[kf.id] = kf.pose
        else:
            if odometry is not None:
                self.poses[kf.id] = self.poses[kf.id - 1] @ odometry
            else:
                self.poses[kf.id] = kf.pose

            self.edges.append((
                kf.id - 1, kf.id,
                np.linalg.inv(self.poses[kf.id - 1]) @ self.poses[kf.id],
                np.eye(6) * 100
            ))

        result = self.detector.detect_loops(kf)
        for loop in result.verified_loops:
            self.add_loop_closure(loop)

    def add_loop_closure(self, loop: LoopCandidate):
        self.loop_closures.append(loop)
        self.edges.append((
            loop.query_id, loop.match_id,
            loop.transform,
            np.eye(6) * 1000
        ))

    def optimize(self) -> Dict[int, np.ndarray]:
        self.poses = self.optimizer.optimize(self.poses, self.edges)
        return self.poses

    def get_trajectory(self) -> List[np.ndarray]:
        return [self.poses[i] for i in sorted(self.poses.keys())]

    def get_loop_closures(self) -> List[LoopCandidate]:
        return self.loop_closures