from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math
import time
import hashlib


class TrackingAlgorithm(Enum):
    KALMAN = "kalman"
    EXTENDED_KALMAN = "extended_kalman"
    UNSCENTED_KALMAN = "unscented_kalman"
    PARTICLE_FILTER = "particle_filter"
    DEEP_SORT = "deep_sort"
    BYTE_TRACK = "byte_track"
    STRONG_SORT = "strong_sort"
    OC_SORT = "oc_sort"
    MOTION_TRACKER = "motion_tracker"
    MULTI_HYPOTHESIS = "multi_hypothesis"


@dataclass(frozen=True, slots=True)
class TrackState:
    position: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray
    timestamp: float


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: np.ndarray
    confidence: float
    class_id: int
    class_name: str
    features: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    keypoints: Optional[np.ndarray] = None


@dataclass(frozen=True, slots=True)
class Track:
    track_id: str
    state: TrackState
    detections: List[Detection]
    age: int
    hits: int
    hit_streak: int
    time_since_update: int
    confirmed: bool = False
    features: Optional[np.ndarray] = None


class KalmanFilterTracker:
    def __init__(
        self,
        dt: float = 1.0 / 30.0,
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

    def get_state(self) -> TrackState:
        return TrackState(
            position=self.x[:2],
            velocity=self.x[4:6],
            covariance=self.P[:4, :4],
            timestamp=time.time(),
        )


class ExtendedKalmanTracker:
    def __init__(
        self,
        dt: float = 1.0 / 30.0,
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
        dt: float = 1.0 / 30.0,
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
        sqrt_P = np.linalg.cholesky((self.state_dim + self.lambd) * P)
        sigmas = np.zeros((self.n_sigma, self.state_dim))
        sigmas[0] = x
        for i in range(self.state_dim):
            sigmas[i + 1] = x + sqrt_P[i]
            sigmas[self.state_dim + i + 1] = x - sqrt_P[i]
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


class ParticleFilterTracker:
    def __init__(
        self,
        num_particles: int = 100,
        process_noise: float = 1.0,
        measurement_noise: float = 10.0,
        resample_threshold: float = 0.5,
    ):
        self.num_particles = num_particles
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.resample_threshold = resample_threshold

        self.particles = np.zeros((num_particles, 4))
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, dt: float = 1.0 / 30.0):
        for i in range(self.num_particles):
            x, y, vx, vy = self.particles[i]
            x_new = x + vx * dt + np.random.randn() * self.process_noise * dt
            y_new = y + vy * dt + np.random.randn() * self.process_noise * dt
            vx_new = vx + np.random.randn() * self.process_noise
            vy_new = vy + np.random.randn() * self.process_noise
            self.particles[i] = [x_new, y_new, vx_new, vy_new]

    def update(self, measurement: np.ndarray):
        for i in range(self.num_particles):
            x, y = self.particles[i][:2]
            dx = x - measurement[0]
            dy = y - measurement[1]
            dist = np.sqrt(dx**2 + dy**2)
            self.weights[i] = np.exp(-dist**2 / (2 * self.measurement_noise**2))

        self.weights /= np.sum(self.weights)

        if 1.0 / np.sum(self.weights**2) < self.resample_threshold * self.num_particles:
            self._resample()

    def _resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_state(self) -> np.ndarray:
        return np.average(self.particles, axis=0, weights=self.weights)

    def initiate(self, measurement: np.ndarray):
        self.particles = np.tile(measurement, (self.num_particles, 1))
        self.particles += np.random.randn(self.num_particles, 4) * 5
        self.weights = np.ones(self.num_particles) / self.num_particles


class MultiHypothesisTracker:
    def __init__(
        self,
        max_hypotheses: int = 100,
        hypothesis_threshold: float = 0.01,
        merge_threshold: float = 0.5,
    ):
        self.max_hypotheses = max_hypotheses
        self.hypothesis_threshold = hypothesis_threshold
        self.merge_threshold = merge_threshold
        self.hypotheses: List[Dict] = []

    def predict(self):
        for hyp in self.hypotheses:
            hyp['tracker'].predict()

    def update(self, detections: List[Detection]):
        new_hypotheses = []

        for hyp in self.hypotheses:
            for det in detections:
                new_hyp = hyp.copy()
                new_hyp['tracker'].update(det.bbox)
                new_hyp['score'] *= self._compute_likelihood(hyp['tracker'].get_state(), det)
                new_hyp['detections'].append(det)
                new_hypotheses.append(new_hyp)

        self.hypotheses = sorted(new_hypotheses, key=lambda h: h['score'], reverse=True)
        self.hypotheses = [h for h in self.hypotheses if h['score'] > self.hypothesis_threshold]
        self.hypotheses = self.hypotheses[:self.max_hypotheses]

        self._merge_hypotheses()

    def _compute_likelihood(self, state: TrackState, det: Detection) -> float:
        pred_pos = state.position
        det_pos = det.bbox[:2]
        dist = np.linalg.norm(pred_pos - det_pos)
        return np.exp(-dist**2 / 50.0)

    def _merge_hypotheses(self):
        merged = []
        for h1 in self.hypotheses:
            merged_flag = False
            for h2 in merged:
                if self._hypotheses_similar(h1, h2):
                    self._merge(h1, h2)
                    merged_flag = True
                    break
            if not merged_flag:
                merged.append(h1)
        self.hypotheses = merged

    def _hypotheses_similar(self, h1: Dict, h2: Dict) -> bool:
        pos1 = h1['tracker'].get_state().position
        pos2 = h2['tracker'].get_state().position
        return np.linalg.norm(pos1 - pos2) < self.merge_threshold

    def _merge(self, h1: Dict, h2: Dict):
        w1, w2 = h1['score'], h2['score']
        total = w1 + w2
        if total > 0:
            for key in ['position', 'velocity']:
                h1['tracker'].x[key] = (w1 * h1['tracker'].x[key] + w2 * h2['tracker'].x[key]) / total
            h1['score'] = max(w1, w2)


class MotionTracker:
    def __init__(
        self,
        tracker_type: TrackingAlgorithm = TrackingAlgorithm.KALMAN,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        self.tracker_type = tracker_type
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: Dict[str, Track] = {}
        self.trackers: Dict[str, Any] = {}
        self.next_id = 0

    def _create_tracker(self):
        if self.tracker_type == TrackingAlgorithm.KALMAN:
            return KalmanFilterTracker()
        elif self.tracker_type == TrackingAlgorithm.EXTENDED_KALMAN:
            return ExtendedKalmanTracker()
        elif self.tracker_type == TrackingAlgorithm.UNSCENTED_KALMAN:
            return UnscentedKalmanTracker()
        elif self.tracker_type == TrackingAlgorithm.PARTICLE_FILTER:
            return ParticleFilterTracker()
        return KalmanFilterTracker()

    def update(self, detections: List[Detection]) -> List[Track]:
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            tracker = self.trackers[track_id]
            predicted = tracker.predict()
            track = Track(
                track_id=track.track_id,
                state=tracker.get_state(),
                detections=track.detections,
                age=track.age + 1,
                hits=track.hits,
                hit_streak=0,
                time_since_update=track.time_since_update + 1,
                confirmed=track.confirmed,
                features=track.features,
            )
            self.tracks[track_id] = track

        if not detections:
            for track_id in list(self.tracks.keys()):
                if self.tracks[track_id].time_since_update > self.max_age:
                    del self.tracks[track_id]
                    del self.trackers[track_id]
            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())
        track_bboxes = np.array([self.tracks[tid].state.position for tid in track_ids])
        det_bboxes = np.array([d.bbox for d in detections])

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
            tracker = self.trackers[track_id]

            updated = tracker.update(det.bbox)

            track = Track(
                track_id=track.track_id,
                state=tracker.get_state(),
                detections=track.detections + [det],
                age=track.age + 1,
                hits=track.hits + 1,
                hit_streak=track.hit_streak + 1,
                time_since_update=0,
                confirmed=track.confirmed or track.hits >= self.min_hits,
                features=track.features,
            )
            self.tracks[track_id] = track

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            track_id = f"track_{self.next_id}"
            self.next_id += 1

            tracker = self._create_tracker()
            tracker.initiate(det.bbox)

            track = Track(
                track_id=track_id,
                state=tracker.get_state(),
                detections=[det],
                age=1,
                hits=1,
                hit_streak=1,
                time_since_update=0,
                confirmed=False,
                features=det.features,
            )
            self.tracks[track_id] = track
            self.trackers[track_id] = tracker

        for track_idx in unmatched_tracks:
            track_id = track_ids[track_idx]
            track = self.tracks[track_id]
            if track.time_since_update > self.max_age:
                del self.tracks[track_id]
                del self.trackers[track_id]

        return [t for t in self.tracks.values() if t.confirmed or t.time_since_update == 0]

    def _hungarian(self, cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
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


class TrackManager:
    def __init__(
        self,
        tracker: MotionTracker,
        feature_extractor: Optional[Callable] = None,
    ):
        self.tracker = tracker
        self.feature_extractor = feature_extractor
        self.track_history: Dict[str, List[Track]] = {}

    def update(self, detections: List[Detection], image: np.ndarray = None) -> List[Track]:
        if self.feature_extractor and image is not None:
            for det in detections:
                if det.features is None:
                    crop = self._extract_crop(image, det.bbox)
                    if crop is not None:
                        det.features = self.feature_extractor(crop)

        tracks = self.tracker.update(detections)

        for track in tracks:
            if track.track_id not in self.track_history:
                self.track_history[track.track_id] = []
            self.track_history[track.track_id].append(track)

        return tracks

    def _extract_crop(self, image: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        if x2 > x1 and y2 > y1:
            return image[y1:y2, x1:x2]
        return None

    def get_trajectory(self, track_id: str) -> List[Track]:
        return self.track_history.get(track_id, [])

    def get_active_tracks(self) -> List[Track]:
        return list(self.tracker.tracks.values())


class TrackFusion:
    def __init__(self, trackers: List[MotionTracker], weights: List[float] = None):
        self.trackers = trackers
        self.weights = weights or [1.0] * len(trackers)

    def update(self, detections: List[Detection]) -> List[Track]:
        all_tracks = []
        for tracker, weight in zip(self.trackers, self.weights):
            tracks = tracker.update(detections)
            for track in tracks:
                track = Track(
                    track_id=track.track_id,
                    state=track.state,
                    detections=track.detections,
                    age=track.age,
                    hits=track.hits,
                    hit_streak=track.hit_streak,
                    time_since_update=track.time_since_update,
                    confirmed=track.confirmed,
                    features=track.features,
                )
                all_tracks.append(track)

        return self._fuse_tracks(all_tracks)

    def _fuse_tracks(self, tracks: List[Track]) -> List[Track]:
        if not tracks:
            return []

        fused = []
        used = [False] * len(tracks)

        for i, track1 in enumerate(tracks):
            if used[i]:
                continue

            cluster = [track1]
            used[i] = True

            for j, track2 in enumerate(tracks):
                if used[j]:
                    continue
                if self._tracks_match(track1, track2):
                    cluster.append(track2)
                    used[j] = True

            fused_track = self._merge_cluster(cluster)
            fused.append(fused_track)

        return fused

    def _tracks_match(self, t1: Track, t2: Track) -> bool:
        dist = np.linalg.norm(t1.state.position - t2.state.position)
        return dist < 50

    def _merge_cluster(self, cluster: List[Track]) -> Track:
        weights = np.array([t.hits for t in cluster])
        weights = weights / weights.sum()

        position = np.sum([t.state.position * w for t, w in zip(cluster, weights)], axis=0)
        velocity = np.sum([t.state.velocity * w for t, w in zip(cluster, weights)], axis=0)

        merged_state = TrackState(
            position=position,
            velocity=velocity,
            covariance=np.mean([t.state.covariance for t in cluster], axis=0),
            timestamp=time.time(),
        )

        return Track(
            track_id=cluster[0].track_id,
            state=merged_state,
            detections=[d for t in cluster for d in t.detections],
            age=max(t.age for t in cluster),
            hits=sum(t.hits for t in cluster),
            hit_streak=max(t.hit_streak for t in cluster),
            time_since_update=min(t.time_since_update for t in cluster),
            confirmed=all(t.confirmed for t in cluster),
            features=cluster[0].features,
        )
