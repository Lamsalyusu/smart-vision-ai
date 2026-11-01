from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True, slots=True)
class DetectionCandidate:
    detection_id: str
    class_name: str
    bbox: tuple[float, float, float, float]  # [x1, y1, x2, y2]
    feature_embedding: np.ndarray | None = None
    velocity_2d: tuple[float, float] | None = None


@dataclass
class TrackTarget:
    track_id: int
    class_name: str
    state_mean: np.ndarray  # 8-state UKF [x, y, s, r, vx, vy, vs, vr]
    state_covariance: np.ndarray
    feature_gallery: list[np.ndarray]
    missed_frames: int = 0


class HungarianFusionMatcher:
    """Optimal Bipartite Matching using Multi-Modal Cost Matrix Fusion.

    Fuses:
    1. Mahalanobis distance from UKF state covariance
    2. Spatial Generalized IoU
    3. Re-ID cosine feature distance
    4. Directional velocity cosine similarity
    """

    def __init__(
        self,
        weight_iou: float = 0.40,
        weight_reid: float = 0.35,
        weight_mahalanobis: float = 0.15,
        weight_velocity: float = 0.10,
        max_distance_threshold: float = 0.65,
    ) -> None:
        self.w_iou = weight_iou
        self.w_reid = weight_reid
        self.w_maha = weight_mahalanobis
        self.w_vel = weight_velocity
        self.max_thresh = max_distance_threshold

    @staticmethod
    def compute_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
        x_left = max(box_a[0], box_b[0])
        y_top = max(box_a[1], box_b[1])
        x_right = min(box_a[2], box_b[2])
        y_bottom = min(box_a[3], box_b[3])

        intersection = max(0.0, x_right - x_left) * max(0.0, y_bottom - y_top)
        area_a = max(1e-6, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        area_b = max(1e-6, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
        union = area_a + area_b - intersection
        return float(intersection / union) if union > 0 else 0.0

    @staticmethod
    def compute_cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 1.0
        similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        return max(0.0, min(2.0, 1.0 - similarity))

    @staticmethod
    def compute_mahalanobis(track: TrackTarget, det_pos: np.ndarray) -> float:
        """Normalized Mahalanobis distance between 2D detection position and UKF state."""
        pos_mean = track.state_mean[:2]
        pos_cov = track.state_covariance[:2, :2] + np.eye(2) * 1e-4
        delta = det_pos - pos_mean
        try:
            inv_cov = np.linalg.inv(pos_cov)
            dist = float(np.sqrt(np.dot(np.dot(delta.T, inv_cov), delta)))
            # Normalize via sigmoid-like transformation
            return float(1.0 - math.exp(-0.5 * dist))
        except np.linalg.LinAlgError:
            return 1.0

    def match(
        self,
        tracks: list[TrackTarget],
        detections: list[DetectionCandidate],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Compute optimal assignment matrix using Hungarian Algorithm.

        Returns:
            matched_pairs: List of (track_idx, detection_idx)
            unmatched_tracks: List of track_idx
            unmatched_detections: List of detection_idx
        """
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        n_t = len(tracks)
        n_d = len(detections)
        cost_matrix = np.ones((n_t, n_d), dtype=np.float64)

        for i, track in enumerate(tracks):
            # Calculate bbox from state: [x, y, s, r]
            cx, cy, s, r = track.state_mean[0], track.state_mean[1], track.state_mean[2], track.state_mean[3]
            w = math.sqrt(max(1.0, s * r))
            h = max(1.0, s / max(1e-4, w))
            track_box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

            for j, det in enumerate(detections):
                # Class gating: different classes cannot match
                if track.class_name != det.class_name:
                    cost_matrix[i, j] = 10.0
                    continue

                # 1. IoU distance
                iou = self.compute_iou(track_box, det.bbox)
                cost_iou = 1.0 - iou

                # 2. Re-ID distance against gallery
                cost_reid = 1.0
                if det.feature_embedding is not None and track.feature_gallery:
                    dists = [
                        self.compute_cosine_distance(det.feature_embedding, g)
                        for g in track.feature_gallery
                    ]
                    cost_reid = min(dists)

                # 3. Mahalanobis distance
                det_center = np.array([(det.bbox[0] + det.bbox[2]) / 2, (det.bbox[1] + det.bbox[3]) / 2])
                cost_maha = self.compute_mahalanobis(track, det_center)

                # 4. Velocity agreement
                cost_vel = 0.5
                if det.velocity_2d is not None and len(track.state_mean) >= 6:
                    track_vel = track.state_mean[4:6]
                    cost_vel = self.compute_cosine_distance(track_vel, np.array(det.velocity_2d))

                fused_cost = (
                    self.w_iou * cost_iou
                    + self.w_reid * cost_reid
                    + self.w_maha * cost_maha
                    + self.w_vel * cost_vel
                )
                cost_matrix[i, j] = fused_cost

        # Execute Hungarian Assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pairs: list[tuple[int, int]] = []
        unmatched_tracks = set(range(n_t))
        unmatched_detections = set(range(n_d))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= self.max_thresh:
                matched_pairs.append((int(r), int(c)))
                unmatched_tracks.discard(int(r))
                unmatched_detections.discard(int(c))

        return matched_pairs, sorted(unmatched_tracks), sorted(unmatched_detections)
