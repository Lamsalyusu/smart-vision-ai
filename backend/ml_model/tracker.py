import numpy as np
from collections import deque


def calculate_iou(bbox1, bbox2):
    """Compute Intersection over Union between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, bbox1[2] - bbox1[0]) * max(0.0, bbox1[3] - bbox1[1])
    area2 = max(0.0, bbox2[2] - bbox2[0]) * max(0.0, bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0.0 else 0.0


def bbox_to_z(bbox):
    """Convert bounding box [x1, y1, x2, y2] to measurement state [u, v, s, r]."""
    w = max(1e-4, bbox[2] - bbox[0])
    h = max(1e-4, bbox[3] - bbox[1])
    u = bbox[0] + w / 2.0
    v = bbox[1] + h / 2.0
    s = w * h
    r = w / h
    return np.array([u, v, s, r], dtype=np.float32).reshape((4, 1))


def x_to_bbox(x):
    """Convert state vector [u, v, s, r, ...] to bounding box [x1, y1, x2, y2]."""
    u, v, s, r = float(x[0, 0]), float(x[1, 0]), float(x[2, 0]), float(x[3, 0])
    if s <= 0 or r <= 0:
        return [0.0, 0.0, 0.0, 0.0]

    w = np.sqrt(s * r)
    h = s / max(1e-4, w)

    return [
        float(u - w / 2.0),
        float(v - h / 2.0),
        float(u + w / 2.0),
        float(v + h / 2.0),
    ]


class KalmanBoxTracker:
    """Kalman filter state estimator for tracking a single bounding box."""

    _count = 0

    def __init__(self, bbox, class_name, confidence, class_id=0):
        KalmanBoxTracker._count += 1
        self.id = KalmanBoxTracker._count
        self.class_name = class_name
        self.class_id = class_id
        self.confidence = float(confidence)

        # State: [u, v, s, r, u_dot, v_dot, s_dot]^T
        self.x = np.zeros((7, 1), dtype=np.float32)
        z = bbox_to_z(bbox)
        self.x[:4] = z

        # State transition model F (constant velocity)
        self.F = np.eye(7, dtype=np.float32)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0

        # Measurement matrix H
        self.H = np.zeros((4, 7), dtype=np.float32)
        self.H[:4, :4] = np.eye(4, dtype=np.float32)

        # State covariance P
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1000.0, 1000.0, 1000.0]).astype(np.float32)

        # Process noise covariance Q
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001]).astype(np.float32)

        # Measurement noise covariance R
        self.R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float32)

        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.history = deque(maxlen=30)
        self.history.append((float(self.x[0, 0]), float(self.x[1, 0])))

    def predict(self):
        """Advance the state vector using the kinematic motion model."""
        if self.x[2, 0] + self.x[6, 0] <= 0:
            self.x[6, 0] = 0.0

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return x_to_bbox(self.x)

    def update(self, bbox, confidence=None):
        """Incorporate new detection measurement into the state estimate."""
        z = bbox_to_z(bbox)
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)
        I = np.eye(7, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        if confidence is not None:
            self.confidence = float(confidence)

        self.history.append((float(self.x[0, 0]), float(self.x[1, 0])))

    def get_state(self):
        """Current smoothed bounding box estimate."""
        return x_to_bbox(self.x)

    def get_velocity(self):
        """Instantaneous velocity vector (vx, vy) in pixels/frame."""
        return float(self.x[4, 0]), float(self.x[5, 0])


class SortTracker:
    """Multi-object tracker associating detections across frames using IoU & Kalman filters."""

    def __init__(self, max_age=15, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        """
        Update tracker with new frame detections.
        detections: list of dicts with 'bbox', 'class', 'confidence', 'class_id'
        Returns: list of tracked objects with 'track_id', smoothed 'bbox', 'velocity'
        """
        self.frame_count += 1

        # 1. Predict new locations of existing tracks
        predicted_boxes = []
        to_delete = []
        for i, trk in enumerate(self.trackers):
            pos = trk.predict()
            if any(np.isnan(pos)):
                to_delete.append(i)
            else:
                predicted_boxes.append(pos)

        for i in reversed(to_delete):
            self.trackers.pop(i)

        # 2. Compute IoU matrix and match detections to existing tracks
        num_tracks = len(self.trackers)
        num_dets = len(detections)

        matched_tracks, matched_dets = set(), set()
        matches = []

        if num_tracks > 0 and num_dets > 0:
            iou_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
            for t_idx, trk in enumerate(self.trackers):
                for d_idx, det in enumerate(detections):
                    # Prefer matching the same class, else penalize
                    same_class = (trk.class_name == det.get("class"))
                    class_factor = 1.0 if same_class else 0.4
                    iou = calculate_iou(predicted_boxes[t_idx], det["bbox"])
                    iou_matrix[t_idx, d_idx] = iou * class_factor

            # Greedy bipartite matching
            flat_indices = np.argsort(-iou_matrix, axis=None)
            for flat_idx in flat_indices:
                t_idx, d_idx = np.unravel_index(flat_idx, iou_matrix.shape)
                if iou_matrix[t_idx, d_idx] < self.iou_threshold:
                    break
                if t_idx in matched_tracks or d_idx in matched_dets:
                    continue
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)
                matches.append((t_idx, d_idx))

        # 3. Update matched trackers
        for t_idx, d_idx in matches:
            det = detections[d_idx]
            self.trackers[t_idx].update(det["bbox"], det.get("confidence"))
            self.trackers[t_idx].class_name = det.get("class", self.trackers[t_idx].class_name)
            self.trackers[t_idx].class_id = det.get("class_id", self.trackers[t_idx].class_id)

        # 4. Create new trackers for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                trk = KalmanBoxTracker(
                    det["bbox"],
                    class_name=det.get("class", "object"),
                    confidence=det.get("confidence", 0.5),
                    class_id=det.get("class_id", 0),
                )
                self.trackers.append(trk)

        # 5. Extract active tracks for return
        results = []
        surviving_trackers = []
        for trk in self.trackers:
            # Active if recently updated and hits exceed minimum threshold
            if trk.time_since_update <= self.max_age:
                surviving_trackers.append(trk)
                if trk.time_since_update == 0 and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                    bbox = trk.get_state()
                    vx, vy = trk.get_velocity()
                    results.append({
                        "track_id": trk.id,
                        "class": trk.class_name,
                        "class_id": trk.class_id,
                        "confidence": trk.confidence,
                        "bbox": [round(c, 1) for c in bbox],
                        "velocity": [round(vx, 2), round(vy, 2)],
                        "hits": trk.hits,
                        "age": trk.age,
                    })

        self.trackers = surviving_trackers
        return results
