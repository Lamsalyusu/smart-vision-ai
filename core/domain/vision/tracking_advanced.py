from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable
from enum import Enum
from collections import deque
import math
import uuid


class TrackState(Enum):
    NEW = "new"
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


class MotionModel(Enum):
    CONSTANT_VELOCITY = "constant_velocity"
    CONSTANT_ACCELERATION = "constant_acceleration"
    KALMAN = "kalman"
    EXTENDED_KALMAN = "extended_kalman"
    UNScented_KALMAN = "unscented_kalman"


@dataclass(frozen=True, slots=True)
class Track:
    track_id: str
    bbox: np.ndarray
    confidence: float
    class_id: int
    class_name: str
    state: TrackState = TrackState.NEW
    age: int = 0
    hits: int = 0
    hit_streak: int = 0
    time_since_update: int = 0
    velocity: np.ndarray = None
    feature: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    keypoints: Optional[np.ndarray] = None
    attributes: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.velocity is None:
            object.__setattr__(self, 'velocity', np.zeros(2))


class KalmanTracker:
    def __init__(
        self,
        dt: float = 1.0,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        state_dim: int = 8,
        meas_dim: int = 4,
    ):
        self.dt = dt
        self.state_dim = state_dim
        self.meas_dim = meas_dim

        self.F = np.eye(state_dim)
        for i in range(4):
            self.F[i, i + 4] = dt

        self.H = np.zeros((meas_dim, state_dim))
        for i in range(meas_dim):
            self.H[i, i] = 1.0

        self.Q = np.eye(state_dim) * process_noise
        for i in range(4):
            self.Q[i, i] = process_noise * dt**4 / 4
            self.Q[i + 4, i + 4] = process_noise * dt**2
            self.Q[i, i + 4] = self.Q[i + 4, i] = process_noise * dt**3 / 2

        self.R = np.eye(meas_dim) * measurement_noise
        self.P = np.eye(state_dim) * 10.0
        self.x = np.zeros(state_dim)

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4]

    def update(self, measurement: np.ndarray) -> np.ndarray:
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P
        return self.x[:4]

    def initiate(self, measurement: np.ndarray):
        self.x[:4] = measurement
        self.x[4:] = 0
        self.P = np.eye(self.state_dim) * 10.0

    def get_state(self) -> np.ndarray:
        return self.x[:4]

    def get_velocity(self) -> np.ndarray:
        return self.x[4:6]


class ExtendedKalmanTracker:
    def __init__(
        self,
        dt: float = 1.0,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
    ):
        self.dt = dt
        self.state_dim = 6
        self.meas_dim = 4

        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 10.0
        self.Q = np.eye(self.state_dim) * process_noise
        self.R = np.eye(self.meas_dim) * measurement_noise

    def f(self, x: np.ndarray) -> np.ndarray:
        x_new = np.zeros_like(x)
        x_new[0] = x[0] + x[2] * self.dt
        x_new[1] = x[1] + x[3] * self.dt
        x_new[2] = x[2]
        x_new[3] = x[3]
        x_new[4] = x[4]
        x_new[5] = x[5]
        return x_new

    def F_jacobian(self, x: np.ndarray) -> np.ndarray:
        F = np.eye(self.state_dim)
        F[0, 2] = self.dt
        F[1, 3] = self.dt
        return F

    def h(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[0] - x[4]/2, x[1] - x[5]/2, x[0] + x[4]/2, x[1] + x[5]/2])

    def H_jacobian(self, x: np.ndarray) -> np.ndarray:
        H = np.zeros((self.meas_dim, self.state_dim))
        H[0, 0] = 1
        H[0, 4] = -0.5
        H[1, 1] = 1
        H[1, 5] = -0.5
        H[2, 0] = 1
        H[2, 4] = 0.5
        H[3, 1] = 1
        H[3, 5] = 0.5
        return H

    def predict(self) -> np.ndarray:
        self.x = self.f(self.x)
        F = self.F_jacobian(self.x)
        self.P = F @ self.P @ F.T + self.Q
        return self.h(self.x)

    def update(self, measurement: np.ndarray) -> np.ndarray:
        h = self.h(self.x)
        H = self.H_jacobian(self.x)

        y = measurement - h
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P
        return self.h(self.x)

    def initiate(self, measurement: np.ndarray):
        x1, y1, x2, y2 = measurement
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        self.x = np.array([cx, cy, 0, 0, w, h])
        self.P = np.eye(self.state_dim) * 10.0


class UnscentedKalmanTracker:
    def __init__(
        self,
        dt: float = 1.0,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        self.dt = dt
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.state_dim = 8
        self.meas_dim = 4

        self.lambd = alpha**2 * (self.state_dim + kappa) - self.state_dim
        self.n_sigma = 2 * self.state_dim + 1

        self.Wm = np.full(self.n_sigma, 1 / (2 * (self.state_dim + self.lambd)))
        self.Wc = np.full(self.n_sigma, 1 / (2 * (self.state_dim + self.lambd)))
        self.Wm[0] = self.lambd / (self.state_dim + self.lambd)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)

        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 10.0
        self.Q = np.eye(self.state_dim) * process_noise
        self.R = np.eye(self.meas_dim) * measurement_noise

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        n = self.state_dim
        sqrt_P = np.linalg.cholesky((n + self.lambd) * P)
        sigmas = np.zeros((self.n_sigma, n))
        sigmas[0] = x
        for i in range(n):
            sigmas[i + 1] = x + sqrt_P[i]
            sigmas[n + i + 1] = x - sqrt_P[i]
        return sigmas

    def f(self, x: np.ndarray) -> np.ndarray:
        x_new = np.zeros_like(x)
        x_new[0] = x[0] + x[4] * self.dt
        x_new[1] = x[1] + x[5] * self.dt
        x_new[2] = x[2] + x[6] * self.dt
        x_new[3] = x[3] + x[7] * self.dt
        x_new[4:] = x[4:]
        return x_new

    def h(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[0] - x[2]/2, x[1] - x[3]/2, x[0] + x[2]/2, x[1] + x[3]/2])

    def predict(self) -> np.ndarray:
        sigmas = self._sigma_points(self.x, self.P)
        sigmas_f = np.array([self.f(s) for s in sigmas])

        self.x = np.sum(self.Wm[:, None] * sigmas_f, axis=0)
        diff = sigmas_f - self.x
        self.P = diff.T @ np.diag(self.Wc) @ diff + self.Q

        return self.h(self.x)

    def update(self, measurement: np.ndarray) -> np.ndarray:
        sigmas = self._sigma_points(self.x, self.P)
        sigmas_h = np.array([self.h(s) for s in sigmas])

        z_pred = np.sum(self.Wm[:, None] * sigmas_h, axis=0)
        diff_z = sigmas_h - z_pred
        diff_x = sigmas - self.x

        P_zz = diff_z.T @ np.diag(self.Wc) @ diff_z + self.R
        P_xz = diff_x.T @ np.diag(self.Wc) @ diff_z

        K = P_xz @ np.linalg.inv(P_zz)
        self.x = self.x + K @ (measurement - z_pred)
        self.P = self.P - K @ P_zz @ K.T

        return self.h(self.x)

    def initiate(self, measurement: np.ndarray):
        x1, y1, x2, y2 = measurement
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0])
        self.P = np.eye(self.state_dim) * 10.0


class DeepSORTTracker:
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_cosine_distance: float = 0.2,
        nn_budget: int = 100,
        feature_extractor: Optional[Callable] = None,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.feature_extractor = feature_extractor

        self.tracks: dict[str, Track] = {}
        self.kalman_filters: dict[str, KalmanTracker] = {}
        self.feature_bank: dict[str, deque] = {}
        self._next_id = 0

    def update(self, detections: list[Detection], image: np.ndarray = None) -> list[Track]:
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            kf = self.kalman_filters[track_id]

            predicted_bbox = kf.predict()
            track = Track(
                track_id=track.track_id,
                bbox=predicted_bbox,
                confidence=track.confidence,
                class_id=track.class_id,
                class_name=track.class_name,
                state=TrackState.LOST if track.time_since_update > 0 else TrackState.TRACKED,
                age=track.age + 1,
                hits=track.hits,
                hit_streak=0,
                time_since_update=track.time_since_update + 1,
                velocity=kf.get_velocity(),
                feature=track.feature,
                mask=track.mask,
                keypoints=track.keypoints,
                attributes=track.attributes,
            )
            self.tracks[track_id] = track

        matched_tracks, unmatched_dets, unmatched_tracks = self._match(detections)

        for track_id, det_idx in matched_tracks:
            det = detections[det_idx]
            track = self.tracks[track_id]
            kf = self.kalman_filters[track_id]

            updated_bbox = kf.update(det.xyxy)
            feature = self._extract_feature(image, det) if image is not None and self.feature_extractor else None

            if feature is not None:
                self._update_feature_bank(track_id, feature)
                track_feature = np.mean(list(self.feature_bank[track_id]), axis=0)
            else:
                track_feature = track.feature

            track = Track(
                track_id=track.track_id,
                bbox=updated_bbox,
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
                state=TrackState.TRACKED,
                age=track.age + 1,
                hits=track.hits + 1,
                hit_streak=track.hit_streak + 1,
                time_since_update=0,
                velocity=kf.get_velocity(),
                feature=track_feature,
                mask=det.mask,
                keypoints=det.keypoints,
                attributes=det.attributes,
            )
            self.tracks[track_id] = track

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            feature = self._extract_feature(image, det) if image is not None and self.feature_extractor else None

            track_id = str(uuid.uuid4())[:8]
            kf = KalmanTracker()
            kf.initiate(det.xyxy)

            track = Track(
                track_id=track_id,
                bbox=det.xyxy,
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
                state=TrackState.NEW,
                age=1,
                hits=1,
                hit_streak=1,
                time_since_update=0,
                velocity=kf.get_velocity(),
                feature=feature,
                mask=det.mask,
                keypoints=det.keypoints,
                attributes=det.attributes,
            )
            self.tracks[track_id] = track
            self.kalman_filters[track_id] = kf
            if feature is not None:
                self.feature_bank[track_id] = deque([feature], maxlen=self.nn_budget)

        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            if track.time_since_update > self.max_age:
                self._remove_track(track_id)

        active_tracks = [t for t in self.tracks.values() if t.state != TrackState.REMOVED]
        return active_tracks

    def _match(self, detections: list[Detection]) -> tuple[list, list, list]:
        if not detections:
            return [], [], list(self.tracks.keys())

        track_ids = list(self.tracks.keys())
        if not track_ids:
            return [], list(range(len(detections))), []

        track_bboxes = np.array([self.tracks[tid].bbox for tid in track_ids])
        det_bboxes = np.array([d.xyxy for d in detections])

        iou_matrix = self._iou_matrix(track_bboxes, det_bboxes)

        cost_matrix = 1 - iou_matrix

        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            if track.feature is not None:
                for j, det in enumerate(detections):
                    if det.feature is not None:
                        cosine_dist = 1 - np.dot(track.feature, det.feature) / (
                            np.linalg.norm(track.feature) * np.linalg.norm(det.feature) + 1e-6
                        )
                        if cosine_dist > self.max_cosine_distance:
                            cost_matrix[i, j] = 1e5

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

        return matched_tracks, unmatched_dets, [track_ids[i] for i in unmatched_tracks]

    def _hungarian(self, cost_matrix: np.ndarray) -> list[tuple[int, int]]:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return list(zip(row_ind, col_ind))

    def _iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        n1, n2 = len(boxes1), len(boxes2)
        iou = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                iou[i, j] = self._iou(boxes1[i], boxes2[j])

        return iou

    def _iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter / (area1 + area2 - inter + 1e-6)

    def _extract_feature(self, image: np.ndarray, det: Detection) -> Optional[np.ndarray]:
        if self.feature_extractor is None or image is None:
            return None

        x1, y1, x2, y2 = map(int, det.xyxy)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        try:
            features = self.feature_extractor.extract(crop)
            if features:
                return features[0].data
        except (ValueError, TypeError, AttributeError):
            return None

    def _update_feature_bank(self, track_id: str, feature: np.ndarray):
        if track_id not in self.feature_bank:
            self.feature_bank[track_id] = deque(maxlen=self.nn_budget)
        self.feature_bank[track_id].append(feature)

    def _remove_track(self, track_id: str):
        if track_id in self.tracks:
            self.tracks[track_id] = Track(
                track_id=self.tracks[track_id].track_id,
                bbox=self.tracks[track_id].bbox,
                confidence=self.tracks[track_id].confidence,
                class_id=self.tracks[track_id].class_id,
                class_name=self.tracks[track_id].class_name,
                state=TrackState.REMOVED,
                age=self.tracks[track_id].age,
                hits=self.tracks[track_id].hits,
                hit_streak=self.tracks[track_id].hit_streak,
                time_since_update=self.tracks[track_id].time_since_update,
                velocity=self.tracks[track_id].velocity,
                feature=self.tracks[track_id].feature,
                mask=self.tracks[track_id].mask,
                keypoints=self.tracks[track_id].keypoints,
                attributes=self.tracks[track_id].attributes,
            )
        if track_id in self.kalman_filters:
            del self.kalman_filters[track_id]
        if track_id in self.feature_bank:
            del self.feature_bank[track_id]


class ByteTracker:
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        min_box_area: float = 10,
        mot20: bool = False,
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area
        self.mot20 = mot20

        self.tracked_tracks: list[Track] = []
        self.lost_tracks: list[Track] = []
        self.removed_tracks: list[Track] = []
        self.frame_id = 0
        self.kalman_filters: dict[str, KalmanTracker] = {}

    def update(self, detections: list[Detection], image: np.ndarray = None) -> list[Track]:
        self.frame_id += 1

        high_conf_dets = [d for d in detections if d.confidence >= self.track_thresh and d.area >= self.min_box_area]
        low_conf_dets = [d for d in detections if d.confidence < self.track_thresh and d.area >= self.min_box_area]

        for track in self.tracked_tracks:
            kf = self.kalman_filters[track.track_id]
            predicted_bbox = kf.predict()
            track = Track(
                track_id=track.track_id,
                bbox=predicted_bbox,
                confidence=track.confidence,
                class_id=track.class_id,
                class_name=track.class_name,
                state=TrackState.LOST,
                age=track.age + 1,
                hits=track.hits,
                hit_streak=0,
                time_since_update=track.time_since_update + 1,
                velocity=kf.get_velocity(),
                feature=track.feature,
                mask=track.mask,
                keypoints=track.keypoints,
                attributes=track.attributes,
            )
            self.tracked_tracks = [t for t in self.tracked_tracks if t.track_id != track.track_id]
            self.tracked_tracks.append(track)

        strack_pool = self.tracked_tracks + self.lost_tracks
        strack_pool = self._joint_stracks(strack_pool)

        dists = self._iou_distance(strack_pool, high_conf_dets)
        matches, u_track, u_det = self._linear_assignment(dists, self.match_thresh)

        for track_idx, det_idx in matches:
            track = strack_pool[track_idx]
            det = high_conf_dets[det_idx]
            kf = self.kalman_filters[track.track_id]
            updated_bbox = kf.update(det.xyxy)

            track = Track(
                track_id=track.track_id,
                bbox=updated_bbox,
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
                state=TrackState.TRACKED,
                age=track.age + 1,
                hits=track.hits + 1,
                hit_streak=track.hit_streak + 1,
                time_since_update=0,
                velocity=kf.get_velocity(),
                feature=track.feature,
                mask=det.mask,
                keypoints=det.keypoints,
                attributes=det.attributes,
            )
            self.tracked_tracks = [t for t in self.tracked_tracks if t.track_id != track.track_id]
            self.tracked_tracks.append(track)

        for idx in u_track:
            track = strack_pool[idx]
            if track.state == TrackState.TRACKED:
                track = Track(
                    track_id=track.track_id,
                    bbox=track.bbox,
                    confidence=track.confidence,
                    class_id=track.class_id,
                    class_name=track.class_name,
                    state=TrackState.LOST,
                    age=track.age,
                    hits=track.hits,
                    hit_streak=track.hit_streak,
                    time_since_update=track.time_since_update,
                    velocity=track.velocity,
                    feature=track.feature,
                    mask=track.mask,
                    keypoints=track.keypoints,
                    attributes=track.attributes,
                )
                self.lost_tracks = [t for t in self.lost_tracks if t.track_id != track.track_id]
                self.lost_tracks.append(track)

        if len(u_det) > 0:
            det_dets = [high_conf_dets[i] for i in u_det]
            r_tracked, r_lost = [], []
            for track in self.tracked_tracks:
                if track.state == TrackState.TRACKED:
                    r_tracked.append(track)
                else:
                    r_lost.append(track)

            dists = self._iou_distance(r_tracked + r_lost, det_dets)
            matches, u_track2, u_det2 = self._linear_assignment(dists, 0.5)

            for track_idx, det_idx in matches:
                track = (r_tracked + r_lost)[track_idx]
                det = det_dets[det_idx]
                kf = self.kalman_filters[track.track_id]
                updated_bbox = kf.update(det.xyxy)

                track = Track(
                    track_id=track.track_id,
                    bbox=updated_bbox,
                    confidence=det.confidence,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    state=TrackState.TRACKED,
                    age=track.age + 1,
                    hits=track.hits + 1,
                    hit_streak=track.hit_streak + 1,
                    time_since_update=0,
                    velocity=kf.get_velocity(),
                    feature=track.feature,
                    mask=det.mask,
                    keypoints=det.keypoints,
                    attributes=det.attributes,
                )
                if track.track_id in [t.track_id for t in self.tracked_tracks]:
                    self.tracked_tracks = [t for t in self.tracked_tracks if t.track_id != track.track_id]
                    self.tracked_tracks.append(track)
                else:
                    self.lost_tracks = [t for t in self.lost_tracks if t.track_id != track.track_id]
                    self.tracked_tracks.append(track)

            for idx in u_det2:
                det = det_dets[idx]
                self._init_track(det)

        for track in self.lost_tracks:
            if self.frame_id - track.time_since_update > self.track_buffer:
                self.removed_tracks.append(track)
                self.lost_tracks = [t for t in self.lost_tracks if t.track_id != track.track_id]
                if track.track_id in self.kalman_filters:
                    del self.kalman_filters[track.track_id]

        return self.tracked_tracks

    def _init_track(self, det: Detection):
        track_id = str(uuid.uuid4())[:8]
        kf = KalmanTracker()
        kf.initiate(det.xyxy)

        track = Track(
            track_id=track_id,
            bbox=det.xyxy,
            confidence=det.confidence,
            class_id=det.class_id,
            class_name=det.class_name,
            state=TrackState.TRACKED,
            age=1,
            hits=1,
            hit_streak=1,
            time_since_update=0,
            velocity=kf.get_velocity(),
            feature=det.feature,
            mask=det.mask,
            keypoints=det.keypoints,
            attributes=det.attributes,
        )
        self.tracked_tracks.append(track)
        self.kalman_filters[track_id] = kf

    def _joint_stracks(self, tracks: list[Track]) -> list[Track]:
        return tracks

    def _iou_distance(self, tracks: list[Track], detections: list[Detection]) -> np.ndarray:
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)))

        track_bboxes = np.array([t.bbox for t in tracks])
        det_bboxes = np.array([d.xyxy for d in detections])
        return 1 - self._iou_matrix(track_bboxes, det_bboxes)

    def _iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        n1, n2 = len(boxes1), len(boxes2)
        iou = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                iou[i, j] = self._iou(boxes1[i], boxes2[j])
        return iou

    def _iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-6)

    def _linear_assignment(self, cost_matrix: np.ndarray, thresh: float) -> tuple:
        from scipy.optimize import linear_sum_assignment
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = [(r, c) for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] < thresh]
        unmatched_rows = [r for r in range(cost_matrix.shape[0]) if r not in [m[0] for m in matches]]
        unmatched_cols = [c for c in range(cost_matrix.shape[1]) if c not in [m[1] for m in matches]]
        return matches, unmatched_rows, unmatched_cols


class TrackManager:
    def __init__(
        self,
        tracker_type: str = "deepsort",
        **tracker_kwargs,
    ):
        self.tracker_type = tracker_type
        self.tracker = self._create_tracker(tracker_type, tracker_kwargs)
        self.track_history: dict[str, list[Track]] = {}

    def _create_tracker(self, tracker_type: str, kwargs: dict):
        if tracker_type == "deepsort":
            return DeepSORTTracker(**kwargs)
        elif tracker_type == "bytetrack":
            return ByteTracker(**kwargs)
        elif tracker_type == "kalman":
            return KalmanMultiTracker(**kwargs)
        else:
            raise ValueError(f"Unknown tracker type: {tracker_type}")

    def update(self, detections: list[Detection], image: np.ndarray = None) -> list[Track]:
        tracks = self.tracker.update(detections, image)

        for track in tracks:
            if track.track_id not in self.track_history:
                self.track_history[track.track_id] = []
            self.track_history[track.track_id].append(track)

        return tracks

    def get_trajectory(self, track_id: str) -> list[Track]:
        return self.track_history.get(track_id, [])

    def get_active_tracks(self) -> list[Track]:
        if hasattr(self.tracker, 'tracked_tracks'):
            return self.tracker.tracked_tracks
        elif hasattr(self.tracker, 'tracks'):
            return [t for t in self.tracker.tracks.values() if t.state != TrackState.REMOVED]
        return []


class KalmanMultiTracker:
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        motion_model: MotionModel = MotionModel.KALMAN,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.motion_model = motion_model

        self.tracks: dict[str, Track] = {}
        self.filters: dict[str, any] = {}
        self._next_id = 0

    def _create_filter(self):
        if self.motion_model == MotionModel.KALMAN:
            return KalmanTracker()
        elif self.motion_model == MotionModel.EXTENDED_KALMAN:
            return ExtendedKalmanTracker()
        elif self.motion_model == MotionModel.UNScented_KALMAN:
            return UnscentedKalmanTracker()
        else:
            return KalmanTracker()

    def update(self, detections: list[Detection], image: np.ndarray = None) -> list[Track]:
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            filt = self.filters[track_id]

            predicted_bbox = filt.predict()
            track = Track(
                track_id=track.track_id,
                bbox=predicted_bbox,
                confidence=track.confidence,
                class_id=track.class_id,
                class_name=track.class_name,
                state=TrackState.LOST if track.time_since_update > 0 else TrackState.TRACKED,
                age=track.age + 1,
                hits=track.hits,
                hit_streak=0,
                time_since_update=track.time_since_update + 1,
                velocity=filt.get_velocity() if hasattr(filt, 'get_velocity') else np.zeros(2),
                feature=track.feature,
                mask=track.mask,
                keypoints=track.keypoints,
                attributes=track.attributes,
            )
            self.tracks[track_id] = track

        if not detections:
            for track_id in list(self.tracks.keys()):
                if self.tracks[track_id].time_since_update > self.max_age:
                    del self.tracks[track_id]
                    del self.filters[track_id]
            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())
        track_bboxes = np.array([self.tracks[tid].bbox for tid in track_ids])
        det_bboxes = np.array([d.xyxy for d in detections])

        iou_matrix = self._iou_matrix(track_bboxes, det_bboxes)
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
            filt = self.filters[track_id]

            updated_bbox = filt.update(det.xyxy)

            track = Track(
                track_id=track.track_id,
                bbox=updated_bbox,
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
                state=TrackState.TRACKED,
                age=track.age + 1,
                hits=track.hits + 1,
                hit_streak=track.hit_streak + 1,
                time_since_update=0,
                velocity=filt.get_velocity() if hasattr(filt, 'get_velocity') else np.zeros(2),
                feature=track.feature,
                mask=det.mask,
                keypoints=det.keypoints,
                attributes=det.attributes,
            )
            self.tracks[track_id] = track

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            track_id = str(uuid.uuid4())[:8]
            filt = self._create_filter()
            filt.initiate(det.xyxy)

            track = Track(
                track_id=track_id,
                bbox=det.xyxy,
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
                state=TrackState.NEW,
                age=1,
                hits=1,
                hit_streak=1,
                time_since_update=0,
                velocity=filt.get_velocity() if hasattr(filt, 'get_velocity') else np.zeros(2),
                feature=det.feature,
                mask=det.mask,
                keypoints=det.keypoints,
                attributes=det.attributes,
            )
            self.tracks[track_id] = track
            self.filters[track_id] = filt

        for track_idx in unmatched_tracks:
            track_id = track_ids[track_idx]
            track = self.tracks[track_id]
            if track.time_since_update > self.max_age:
                del self.tracks[track_id]
                del self.filters[track_id]

        return [t for t in self.tracks.values() if t.hits >= self.min_hits or t.time_since_update == 0]

    def _hungarian(self, cost_matrix: np.ndarray) -> list[tuple[int, int]]:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return list(zip(row_ind, col_ind))

    def _iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        n1, n2 = len(boxes1), len(boxes2)
        iou = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                iou[i, j] = self._iou(boxes1[i], boxes2[j])
        return iou

    def _iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-6)
