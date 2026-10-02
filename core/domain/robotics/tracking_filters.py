from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math
import time


class MotionModel(Enum):
    CONSTANT_VELOCITY = "constant_velocity"
    CONSTANT_ACCELERATION = "constant_acceleration"
    COORDINATED_TURN = "coordinated_turn"
    CYCLIC = "cyclic"
    SINGER = "singer"
    CURVED = "curved"


@dataclass(frozen=True, slots=True)
class Track:
    track_id: str
    state: np.ndarray
    covariance: np.ndarray
    timestamp: float
    label: Optional[int] = None
    score: float = 1.0
    age: int = 0
    hits: int = 1
    hit_streak: int = 1


class KalmanFilter:
    def __init__(
        self,
        state_dim: int,
        meas_dim: int,
        dt: float = 1.0,
    ):
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.dt = dt

        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 1000

        self.F = np.eye(state_dim)
        self.Q = np.eye(state_dim) * 0.01
        self.H = np.zeros((meas_dim, state_dim))
        self.R = np.eye(meas_dim) * 0.1

    def set_model(self, F: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray):
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R

    def predict(self, dt: float = None):
        dt = dt if dt is not None else self.dt
        self.F[:3, 3:6] = np.eye(3) * dt
        self.F[3:6, 6:9] = np.eye(3) * dt

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray, H: np.ndarray = None, R: np.ndarray = None):
        if H is not None:
            self.H = H
        if R is not None:
            self.R = R

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x.copy(), self.P.copy()


class ExtendedKalmanFilter:
    def __init__(self, state_dim: int):
        self.state_dim = state_dim
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 1000

    def predict(self, f_func, F_func, Q: np.ndarray, dt: float):
        self.x = f_func(self.x, dt)
        F = F_func(self.x, dt)
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, h_func, H_func, R: np.ndarray):
        h = h_func(self.x)
        H = H_func(self.x)

        y = z - h
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P


class UnscentedKalmanFilter:
    def __init__(
        self,
        state_dim: int,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        self.state_dim = state_dim
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambd = alpha**2 * (state_dim + kappa) - state_dim
        self.n_sigma = 2 * state_dim + 1

        self.Wm = np.full(self.n_sigma, 1 / (2 * (state_dim + self.lambd)))
        self.Wc = np.full(self.n_sigma, 1 / (2 * (state_dim + self.lambd)))
        self.Wm[0] = self.lambd / (state_dim + self.lambd)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)

        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 1000

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        sqrt_P = np.linalg.cholesky((self.state_dim + self.lambd) * P)
        sigmas = np.zeros((self.n_sigma, self.state_dim))
        sigmas[0] = x
        for i in range(self.state_dim):
            sigmas[i + 1] = x + sqrt_P[i]
            sigmas[self.state_dim + i + 1] = x - sqrt_P[i]
        return sigmas

    def predict(self, f_func, Q: np.ndarray, dt: float):
        sigmas = self._sigma_points(self.x, self.P)
        sigmas_f = np.array([f_func(s, dt) for s in sigmas])

        self.x = np.sum(self.Wm[:, None] * sigmas_f, axis=0)
        diff = sigmas_f - self.x
        self.P = diff.T @ np.diag(self.Wc) @ diff + Q

    def update(self, z: np.ndarray, h_func, R: np.ndarray):
        sigmas = self._sigma_points(self.x, self.P)
        sigmas_h = np.array([h_func(s) for s in sigmas])

        z_pred = np.sum(self.Wm[:, None] * sigmas_h, axis=0)
        diff_z = sigmas_h - z_pred
        diff_x = sigmas - self.x

        P_zz = diff_z.T @ np.diag(self.Wc) @ diff_z + R
        P_xz = diff_x.T @ np.diag(self.Wc) @ diff_z

        K = P_xz @ np.linalg.inv(P_zz)
        self.x = self.x + K @ (z - z_pred)
        self.P = self.P - K @ P_zz @ K.T


class IMMFilter:
    def __init__(self, filters: List, mode_transition_matrix: np.ndarray):
        self.filters = filters
        self.mode_probs = np.ones(len(filters)) / len(filters)
        self.transition_matrix = mode_transition_matrix

    def predict(self, dt: float):
        for f in self.filters:
            f.predict(dt)

    def update(self, z: np.ndarray, H: np.ndarray = None, R: np.ndarray = None):
        likelihoods = []
        for f in self.filters:
            y = z - f.H @ f.x
            S = f.H @ f.P @ f.H.T + f.R
            likelihood = np.exp(-0.5 * y.T @ np.linalg.inv(S) @ y) / np.sqrt(np.linalg.det(2 * np.pi * S))
            likelihoods.append(likelihood)

        likelihoods = np.array(likelihoods)
        self.mode_probs = self.mode_probs * likelihoods
        self.mode_probs /= self.mode_probs.sum()

        for i, f in enumerate(self.filters):
            f.update(z, f.H, f.R)

        self.mode_probs = self.transition_matrix.T @ self.mode_probs

    def get_combined_state(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.sum(self.mode_probs[:, None] * np.array([f.x for f in self.filters]), axis=0)
        P = np.zeros_like(self.filters[0].P)
        for i, f in enumerate(self.filters):
            diff = f.x - self.get_combined_state()[0]
            P += self.mode_probs[i] * (f.P + np.outer(diff, diff))
        return self.get_combined_state()


class ParticleFilter:
    def __init__(
        self,
        num_particles: int = 1000,
        process_noise: float = 0.1,
        measurement_noise: float = 0.1,
    ):
        self.num_particles = num_particles
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.particles = np.random.randn(num_particles, 4)
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, dt: float = 1.0):
        for i in range(self.num_particles):
            self.particles[i, :2] += self.particles[i, 2:] * dt + np.random.randn(2) * self.process_noise * dt
            self.particles[i, 2:] += np.random.randn(2) * self.process_noise * dt

    def update(self, measurement: np.ndarray, R: np.ndarray):
        for i in range(self.num_particles):
            pred = self.particles[i, :2]
            diff = measurement - pred
            self.weights[i] = np.exp(-0.5 * diff.T @ np.linalg.inv(R) @ diff)

        self.weights /= self.weights.sum()

        if 1.0 / np.sum(self.weights**2) < self.num_particles / 2:
            self._resample()

    def _resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.average(self.particles, axis=0, weights=self.weights)
        P = np.cov(self.particles.T, aweights=self.weights)
        return x, P


class PDAFilter:
    def __init__(
        self,
        state_dim: int,
        meas_dim: int,
        pd: float = 0.9,
        pg: float = 0.1,
        clutter_density: float = 1e-6,
    ):
        self.kf = KalmanFilter(state_dim, meas_dim)
        self.pd = pd
        self.pg = pg
        self.clutter_density = clutter_density

    def update(self, measurements: List[np.ndarray], H: np.ndarray = None, R: np.ndarray = None):
        if H is not None:
            self.kf.H = H
        if R is not None:
            self.kf.R = R

        self.kf.predict()

        if not measurements:
            self.kf.x = self.kf.F @ self.kf.x
            self.kf.P = self.kf.F @ self.kf.P @ self.kf.F.T + self.kf.Q
            return

        S = self.kf.H @ self.kf.P @ self.kf.H.T + self.kf.R
        S_inv = np.linalg.inv(S)

        associations = []
        for z in measurements:
            y = z - self.kf.H @ self.kf.x
            beta = np.exp(-0.5 * z.T @ np.linalg.inv(self.kf.H @ self.kf.P @ self.kf.H.T + self.kf.R) @ z)
            associations.append(beta)

        beta_sum = sum(associations)
        beta_0 = (1 - self.pd) / (1 - self.pg)

        for i, z in enumerate(measurements):
            if associations[i] > 0:
                self.kf.update(z)

        self.kf.x = self.kf.F @ self.kf.x
        self.kf.P = self.kf.F @ self.kf.P @ self.kf.F.T + self.kf.Q


class JPDAFilter:
    def __init__(
        self,
        state_dim: int,
        meas_dim: int,
        pd: float = 0.9,
        pg: float = 0.1,
        clutter_density: float = 1e-6,
        max_tracks: int = 100,
    ):
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.pd = pd
        self.pg = pg
        self.clutter_density = clutter_density
        self.max_tracks = max_tracks
        self.tracks: Dict[str, KalmanFilter] = {}

    def predict(self, dt: float):
        for track in self.tracks.values():
            track.predict(dt)

    def update(self, measurements: List[np.ndarray]):
        if not measurements:
            return

        track_ids = list(self.tracks.keys())
        num_tracks = len(track_ids)
        num_meas = len(measurements)

        if num_tracks == 0:
            for z in measurements:
                self._init_track(z)
            return

        S = np.zeros((num_meas, num_meas))
        for i in range(num_meas):
            for j in range(num_meas):
                S[i, j] = self._association_probability(i, j)

        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            if track_id not in self.tracks:
                continue

            associations = []
            for z in measurements:
                beta = self._association_probability(track, z)
                associations.append(beta)

            beta_sum = sum(associations)
            if beta_sum > 0:
                for i, z in enumerate(measurements):
                    beta = associations[i] / beta_sum
                    self._weighted_update(track, measurements[i], associations[i])

    def _init_track(self, z: np.ndarray):
        track_id = str(len(self.tracks))
        kf = KalmanFilter(6, 3)
        kf.x[:3] = z
        self.tracks[track_id] = kf

    def _association_probability(self, track_idx: int, meas_idx: int) -> float:
        return 1.0

    def _weighted_update(self, track: KalmanFilter, measurement: np.ndarray, beta: float):
        track.update(measurement)


class MHTFilter:
    def __init__(
        self,
        max_hypotheses: int = 100,
        hypothesis_threshold: float = 0.01,
    ):
        self.max_hypotheses = max_hypotheses
        self.hypothesis_threshold = hypothesis_threshold
        self.hypotheses: List[Dict] = []

    def predict(self, dt: float):
        for hyp in self.hypotheses:
            for track in hyp['tracks'].values():
                track.predict(dt)

    def update(self, measurements: List[np.ndarray]):
        new_hypotheses = []

        for hyp in self.hypotheses:
            for z in measurements:
                new_hyp = hyp.copy()
                new_hyp['tracks'] = {tid: track.copy() for tid, track in hyp['tracks'].items()}
                new_hyp['tracks']['new'] = KalmanFilter(6, 3)
                new_hyp['tracks']['new'].x[:3] = z
                new_hyp['score'] *= 0.9
                new_hypotheses.append(new_hyp)

        self.hypotheses = sorted(new_hypotheses, key=lambda h: h['score'], reverse=True)[:self.max_hypotheses]


class TrackManager:
    def __init__(
        self,
        filter_type: str = "kalman",
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        self.filter_type = filter_type
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks: Dict[str, Track] = {}
        self.filters: Dict[str, Any] = {}
        self._next_id = 0

    def _create_filter(self):
        if self.filter_type == "kalman":
            return KalmanFilter(8, 4)
        elif self.filter_type == "ekf":
            return ExtendedKalmanFilter(8)
        elif self.filter_type == "ukf":
            return UnscentedKalmanFilter(8)
        return KalmanFilter(8, 4)

    def update(self, detections: List[np.ndarray]) -> List[Track]:
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            filt = self.filters[track_id]

            predicted = filt.predict()
            track.state = predicted
            track.age += 1
            track.hit_streak = 0
            track.time_since_update += 1

        if not detections:
            for track_id in list(self.tracks.keys()):
                if self.tracks[track_id].time_since_update > self.max_age:
                    del self.tracks[track_id]
                    del self.filters[track_id]
            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())
        track_states = np.array([self.tracks[tid].state[:4] for tid in track_ids])
        det_states = np.array([d[:4] for d in detections])

        iou_matrix = self._iou_matrix(track_states, det_states)
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

            updated = filt.update(det[:4])

            track.state = updated
            track.hits += 1
            track.hit_streak += 1
            track.time_since_update = 0

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            track_id = str(self._next_id)
            self._next_id += 1

            filt = KalmanFilter(8, 4)
            filt.initiate(det[:4])

            track = Track(
                track_id=track_id,
                state=filt.x[:4],
                covariance=filt.P[:4, :4],
                timestamp=time.time(),
                age=1,
                hits=1,
                hit_streak=1,
                time_since_update=0,
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


class MultiTargetTracker:
    def __init__(
        self,
        tracker_type: str = "kalman",
        max_tracks: int = 100,
    ):
        self.tracker_type = tracker_type
        self.max_tracks = max_tracks
        self.tracks: Dict[str, Track] = {}
        self.filters: Dict[str, Any] = {}
        self._next_id = 0

    def _create_filter(self):
        if self.tracker_type == "kalman":
            return KalmanFilter(8, 4)
        elif self.tracker_type == "ekf":
            return ExtendedKalmanFilter(8)
        elif self.tracker_type == "ukf":
            return UnscentedKalmanFilter(8)
        return KalmanFilter(8, 4)

    def update(self, detections: List[np.ndarray]) -> List[Track]:
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            filt = self.filters[track_id]

            predicted = filt.predict()
            track.state = predicted
            track.age += 1
            track.hit_streak = 0
            track.time_since_update += 1

        if not detections:
            for track_id in list(self.tracks.keys()):
                if self.tracks[track_id].time_since_update > 30:
                    del self.tracks[track_id]
                    del self.filters[track_id]
            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())
        track_bboxes = np.array([self.tracks[tid].state[:4] for tid in track_ids])
        det_bboxes = np.array([d[:4] for d in detections])

        iou_matrix = self._iou_matrix(track_bboxes, det_bboxes)
        cost_matrix = 1 - iou_matrix

        matches = self._hungarian(cost_matrix)

        matched_tracks = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(track_ids)))

        for track_idx, det_idx in matches:
            if cost_matrix[track_idx, det_idx] > 0.7:
                continue
            matched_tracks.append((track_ids[track_idx], det_idx))
            unmatched_dets.remove(det_idx)
            unmatched_tracks.remove(track_idx)

        for track_id, det_idx in matched_tracks:
            det = detections[det_idx]
            track = self.tracks[track_id]
            filt = self.filters[track_id]

            updated = filt.update(det[:4])

            track.state = updated
            track.hits += 1
            track.hit_streak += 1
            track.time_since_update = 0

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            track_id = str(self._next_id)
            self._next_id += 1

            filt = KalmanFilter(8, 4)
            filt.initiate(det[:4])

            track = Track(
                track_id=track_id,
                state=filt.x[:4],
                covariance=filt.P[:4, :4],
                timestamp=time.time(),
                age=1,
                hits=1,
                hit_streak=1,
                time_since_update=0,
            )
            self.tracks[track_id] = track
            self.filters[track_id] = filt

        for track_idx in unmatched_tracks:
            track_id = track_ids[track_idx]
            track = self.tracks[track_id]
            if track.time_since_update > 30:
                del self.tracks[track_id]
                del self.filters[track_id]

        return [t for t in self.tracks.values() if t.hits >= 3 or t.time_since_update == 0]

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


class TrackingEvaluator:
    @staticmethod
    def compute_mota(tracks: List, gt_tracks: List) -> float:
        return 0.0

    @staticmethod
    def compute_motp(tracks: List, gt_tracks: List) -> float:
        return 0.0

    @staticmethod
    def compute_idf1(tracks: List, gt_tracks: List) -> float:
        return 0.0

    @staticmethod
    def compute_tracklet_fragmentation(tracks: List) -> float:
        return 0.0


class TrackVisualizer:
    @staticmethod
    def draw_tracks(image: np.ndarray, tracks: List, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        vis = image.copy()
        for track in tracks:
            if track.age < 3:
                continue
            x, y, w, h = track.state[:4]
            cv2.rectangle(vis, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
            cv2.putText(vis, f"ID: {track.track_id}", (int(x), int(y)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return vis

    @staticmethod
    def draw_trajectories(image: np.ndarray, tracks: List, history_length: int = 30) -> np.ndarray:
        return image