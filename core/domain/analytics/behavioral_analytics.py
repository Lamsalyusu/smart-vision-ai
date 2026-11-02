from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict, Counter
import copy


class BehaviorType(Enum):
    MOVEMENT = "movement"
    INTERACTION = "interaction"
    DWELL = "dwell"
    TRANSITION = "transition"
    GESTURE = "gesture"
    GAZE = "gaze"
    SOCIAL = "social"
    ANOMALOUS = "anomalous"


@dataclass(frozen=True, slots=True)
class BehaviorEvent:
    event_id: str
    entity_id: str
    behavior_type: BehaviorType
    timestamp: float
    position: Optional[np.ndarray] = None
    features: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class BehaviorSequence:
    sequence_id: str
    entity_id: str
    events: List[BehaviorEvent]
    start_time: float
    end_time: float
    pattern: Optional[str] = None
    anomaly_score: float = 0.0


@dataclass(frozen=True, slots=True)
class BehaviorProfile:
    entity_id: str
    behavior_counts: Dict[BehaviorType, int]
    transition_matrix: np.ndarray
    average_durations: Dict[BehaviorType, float]
    preferred_locations: List[np.ndarray]
    activity_patterns: Dict[str, float]
    last_updated: float


class BehaviorTracker:
    def __init__(
        self,
        window_size: float = 3600.0,
        min_sequence_length: int = 3
    ):
        self.window_size = window_size
        self.min_sequence_length = min_sequence_length
        self.events: Dict[str, List[BehaviorEvent]] = defaultdict(list)
        self.sequences: Dict[str, List[BehaviorSequence]] = defaultdict(list)
        self.profiles: Dict[str, BehaviorProfile] = {}

    def add_event(self, event: BehaviorEvent):
        self.events[event.entity_id].append(event)
        self._cleanup_old_events(event.entity_id)
        self._update_profile(event.entity_id)
        self._detect_sequences(event.entity_id)

    def _cleanup_old_events(self, entity_id: str):
        current_time = time.time()
        cutoff = current_time - self.window_size
        self.events[entity_id] = [
            e for e in self.events[entity_id] if e.timestamp > cutoff
        ]

    def _update_profile(self, entity_id: str):
        events = self.events[entity_id]
        if not events:
            return

        behavior_counts = Counter(e.behavior_type for e in events)

        transition_matrix = np.zeros((len(BehaviorType), len(BehaviorType)))
        for i in range(len(events) - 1):
            from_idx = list(BehaviorType).index(events[i].behavior_type)
            to_idx = list(BehaviorType).index(events[i + 1].behavior_type)
            transition_matrix[from_idx, to_idx] += 1

        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)

        avg_durations = {}
        for bt in BehaviorType:
            durations = [e.duration for e in events if e.behavior_type == bt]
            if durations:
                avg_durations[bt] = np.mean(durations)

        locations = [e.position for e in events if e.position is not None]
        preferred_locations = []
        if locations:
            from sklearn.cluster import KMeans
            try:
                kmeans = KMeans(n_clusters=min(5, len(locations)))
                kmeans.fit(np.array(locations))
                preferred_locations = kmeans.cluster_centers_.tolist()
            except:
                pass

        hourly_activity = defaultdict(int)
        for e in events:
            hour = int((e.timestamp % 86400) / 3600)
            hourly_activity[hour] += 1

        self.profiles[entity_id] = BehaviorProfile(
            entity_id=entity_id,
            behavior_counts=dict(behavior_counts),
            transition_matrix=transition_matrix,
            average_durations=avg_durations,
            preferred_locations=preferred_locations,
            activity_patterns=dict(hourly_activity),
            last_updated=time.time()
        )

    def _detect_sequences(self, entity_id: str):
        events = self.events[entity_id]
        if len(events) < self.min_sequence_length:
            return

        sequences = []
        for i in range(len(events) - self.min_sequence_length + 1):
            seq_events = events[i:i + self.min_sequence_length]
            pattern = "->".join(e.behavior_type.value for e in seq_events)

            anomaly_score = self._calculate_anomaly_score(seq_events)

            sequences.append(BehaviorSequence(
                sequence_id=f"{entity_id}_seq_{int(time.time())}_{i}",
                entity_id=entity_id,
                events=seq_events,
                start_time=seq_events[0].timestamp,
                end_time=seq_events[-1].timestamp,
                pattern=pattern,
                anomaly_score=anomaly_score
            ))

        self.sequences[entity_id] = sequences[-100:]

    def _calculate_anomaly_score(self, events: List[BehaviorEvent]) -> float:
        if len(events) < 2:
            return 0.0

        durations = [e.duration for e in events]
        if durations:
            mean_dur = np.mean(durations)
            std_dur = np.std(durations)
            anomaly = np.mean([abs(d - mean_dur) / (std_dur + 1e-6) for d in durations])
        else:
            anomaly = 0.0

        return float(min(anomaly, 10.0))

    def get_profile(self, entity_id: str) -> Optional[BehaviorProfile]:
        return self.profiles.get(entity_id)

    def get_recent_sequences(self, entity_id: str, limit: int = 10) -> List[BehaviorSequence]:
        return self.sequences.get(entity_id, [])[-limit:]


class BehaviorAnalyzer:
    def __init__(self, tracker: BehaviorTracker):
        self.tracker = tracker

    def analyze_entity(self, entity_id: str) -> Dict[str, Any]:
        profile = self.tracker.get_profile(entity_id)
        sequences = self.tracker.get_recent_sequences(entity_id)

        if not profile:
            return {}

        return {
            'entity_id': entity_id,
            'total_events': sum(profile.behavior_counts.values()),
            'behavior_distribution': profile.behavior_counts,
            'dominant_behavior': max(profile.behavior_counts, key=profile.behavior_counts.get).value if profile.behavior_counts else None,
            'transition_matrix': profile.transition_matrix.tolist(),
            'avg_durations': profile.average_durations,
            'preferred_locations': profile.preferred_locations,
            'activity_heatmap': profile.activity_patterns,
            'sequence_count': len(sequences),
            'anomalous_sequences': [s for s in sequences if s.anomaly_score > 2.0],
            'regularity_score': self._calculate_regularity(profile)
        }

    def _calculate_regularity(self, profile: BehaviorProfile) -> float:
        if not profile.behavior_counts:
            return 0.0

        total = sum(profile.behavior_counts.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in profile.behavior_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        max_entropy = math.log2(len(BehaviorType))
        return 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0

    def compare_entities(self, entity_id_1: str, entity_id_2: str) -> float:
        profile1 = self.tracker.get_profile(entity_id_1)
        profile2 = self.tracker.get_profile(entity_id_2)

        if not profile1 or not profile2:
            return 0.0

        behavior_vec1 = np.array([profile1.behavior_counts.get(bt, 0) for bt in BehaviorType])
        behavior_vec2 = np.array([profile2.behavior_counts.get(bt, 0) for bt in BehaviorType])

        norm1 = np.linalg.norm(behavior_vec1)
        norm2 = np.linalg.norm(behavior_vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(behavior_vec1, behavior_vec2) / (norm1 * norm2)
        return float(similarity)


class AnomalousBehaviorDetector:
    def __init__(
        self,
        tracker: BehaviorTracker,
        anomaly_threshold: float = 2.0,
        window_size: int = 50
    ):
        self.tracker = tracker
        self.anomaly_threshold = anomaly_threshold
        self.window_size = window_size
        self.baseline_profiles: Dict[str, BehaviorProfile] = {}

    def train_baseline(self, entity_id: str):
        profile = self.tracker.get_profile(entity_id)
        if profile:
            self.baseline_profiles[entity_id] = profile

    def detect_anomalies(self, entity_id: str) -> List[BehaviorEvent]:
        if entity_id not in self.baseline_profiles:
            return []

        baseline = self.baseline_profiles[entity_id]
        current = self.tracker.get_profile(entity_id)

        if not current:
            return []

        anomalies = []

        for bt in BehaviorType:
            baseline_count = baseline.behavior_counts.get(bt, 0)
            current_count = current.behavior_counts.get(bt, 0)

            if baseline_count > 0:
                ratio = current_count / baseline_count
                if ratio > self.anomaly_threshold or ratio < 1.0 / self.anomaly_threshold:
                    anomalies.append(BehaviorEvent(
                        event_id=f"anomaly_{bt.value}_{int(time.time())}",
                        entity_id=entity_id,
                        behavior_type=bt,
                        timestamp=time.time(),
                        features={
                            'baseline_count': baseline_count,
                            'current_count': current_count,
                            'ratio': ratio
                        },
                        confidence=min(abs(math.log(ratio)), 1.0) if ratio > 0 else 0.0
                    ))

        return anomalies

    def detect_sequence_anomalies(self, entity_id: str) -> List[BehaviorSequence]:
        sequences = self.tracker.get_recent_sequences(entity_id)
        anomalous = [s for s in sequences if s.anomaly_score > self.anomaly_threshold]
        return anomalous


class SocialBehaviorAnalyzer:
    def __init__(self, tracker: BehaviorTracker):
        self.tracker = tracker

    def detect_interactions(
        self,
        entity_ids: List[str],
        proximity_threshold: float = 2.0,
        time_window: float = 60.0
    ) -> List[Dict[str, Any]]:
        interactions = []

        for i, id1 in enumerate(entity_ids):
            for id2 in entity_ids[i + 1:]:
                interaction = self._analyze_pair(id1, id2, proximity_threshold, time_window)
                if interaction:
                    interactions.append(interaction)

        return interactions

    def _analyze_pair(
        self,
        id1: str,
        id2: str,
        proximity_threshold: float,
        time_window: float
    ) -> Optional[Dict[str, Any]]:
        events1 = self.tracker.events.get(id1, [])
        events2 = self.tracker.events.get(id2, [])

        if not events1 or not events2:
            return None

        interactions = []

        for e1 in events1:
            for e2 in events2:
                if abs(e1.timestamp - e2.timestamp) > time_window:
                    continue

                if e1.position is not None and e2.position is not None:
                    dist = np.linalg.norm(e1.position - e2.position)
                    if dist <= proximity_threshold:
                        interactions.append({
                            'time': e1.timestamp,
                            'distance': dist,
                            'behavior_1': e1.behavior_type.value,
                            'behavior_2': e2.behavior_type.value
                        })

        if not interactions:
            return None

        return {
            'entity_1': id1,
            'entity_2': id2,
            'interaction_count': len(interactions),
            'avg_distance': np.mean([i['distance'] for i in interactions]),
            'duration': interactions[-1]['time'] - interactions[0]['time'] if interactions else 0,
            'behaviors': list(set(i['behavior_1'] + '_' + i['behavior_2'] for i in interactions))
        }

    def detect_groups(self, entity_ids: List[str], proximity_threshold: float = 3.0) -> List[List[str]]:
        graph = defaultdict(set)

        for i, id1 in enumerate(entity_ids):
            for id2 in entity_ids[i + 1:]:
                if self._are_proximate(id1, id2, proximity_threshold):
                    graph[id1].add(id2)
                    graph[id2].add(id1)

        visited = set()
        groups = []

        for entity_id in entity_ids:
            if entity_id in visited:
                continue

            group = []
            queue = [entity_id]
            visited.add(entity_id)

            while queue:
                current = queue.pop(0)
                group.append(current)

                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(group) > 1:
                groups.append(group)

        return groups

    def _are_proximate(self, id1: str, id2: str, threshold: float) -> bool:
        events1 = self.tracker.events.get(id1, [])
        events2 = self.tracker.events.get(id2, [])

        for e1 in events1[-10:]:
            for e2 in events2[-10:]:
                if e1.position is not None and e2.position is not None:
                    dist = np.linalg.norm(e1.position - e2.position)
                    if dist <= threshold:
                        return True
        return False


class BehaviorPredictor:
    def __init__(self, tracker: BehaviorTracker):
        self.tracker = tracker
        self.transition_models: Dict[str, np.ndarray] = {}

    def train(self, entity_id: str):
        profile = self.tracker.get_profile(entity_id)
        if profile:
            self.transition_models[entity_id] = profile.transition_matrix

    def predict_next_behavior(
        self,
        entity_id: str,
        current_behavior: BehaviorType,
        steps: int = 3
    ) -> List[Tuple[BehaviorType, float]]:
        if entity_id not in self.transition_models:
            return []

        transition_matrix = self.transition_models[entity_id]
        current_idx = list(BehaviorType).index(current_behavior)

        predictions = []
        current_probs = transition_matrix[current_idx]

        for step in range(steps):
            next_idx = np.argmax(current_probs)
            next_behavior = list(BehaviorType)[next_idx]
            confidence = current_probs[next_idx]
            predictions.append((next_behavior, float(confidence)))
            current_probs = transition_matrix[next_idx]

        return predictions

    def predict_location(
        self,
        entity_id: str,
        current_time: float = None
    ) -> Optional[np.ndarray]:
        profile = self.tracker.get_profile(entity_id)
        if not profile or not profile.preferred_locations:
            return None

        current_time = current_time or time.time()
        hour = int((current_time % 86400) / 3600)

        if hour in profile.activity_patterns:
            activity_level = profile.activity_patterns[hour]
            if activity_level > 0:
                idx = int(activity_level * len(profile.preferred_locations)) % len(profile.preferred_locations)
                return profile.preferred_locations[idx]

        return profile.preferred_locations[0] if profile.preferred_locations else None


class BehaviorVisualizer:
    @staticmethod
    def draw_behavior_timeline(
        events: List[BehaviorEvent],
        image_size: Tuple[int, int] = (1200, 400)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if not events:
            return vis

        min_time = min(e.timestamp for e in events)
        max_time = max(e.timestamp for e in events)
        time_range = max_time - min_time if max_time > min_time else 1

        colors = {
            BehaviorType.MOVEMENT: (0, 255, 0),
            BehaviorType.INTERACTION: (255, 0, 0),
            BehaviorType.DWELL: (0, 0, 255),
            BehaviorType.TRANSITION: (255, 255, 0),
            BehaviorType.GESTURE: (255, 0, 255),
            BehaviorType.GAZE: (0, 255, 255),
            BehaviorType.SOCIAL: (128, 0, 128),
            BehaviorType.ANOMALOUS: (255, 255, 255)
        }

        for event in events:
            x = int((event.timestamp - min_time) / (max_time - min_time + 1e-6) * (image_size[0] - 100)) + 50
            y = image_size[1] // 2

            color = colors.get(event.behavior_type, (255, 255, 255))
            width = max(10, int(event.duration * 10))

            cv2.rectangle(vis, (x, y - 20), (x + width, y + 20), color, -1)
            cv2.putText(vis, event.behavior_type.value[:3], (x + 2, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

        return vis

    @staticmethod
    def draw_transition_matrix(
        matrix: np.ndarray,
        labels: List[str],
        image_size: Tuple[int, int] = (600, 600)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        n = len(labels)
        if n == 0 or matrix.shape != (n, n):
            return vis

        cell_size = min(image_size[0], image_size[1]) // n

        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                color_intensity = int(val * 255)
                x = j * cell_size
                y = i * cell_size
                cv2.rectangle(vis, (x, y), (x + cell_size, y + cell_size),
                             (color_intensity, color_intensity, 255 - color_intensity), -1)

                if i == j:
                    cv2.rectangle(vis, (x, y), (x + cell_size, y + cell_size), (0, 255, 0), 2)

        for i, label in enumerate(labels):
            y = i * cell_size + cell_size // 2
            cv2.putText(vis, label, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            x = i * cell_size + cell_size // 2
            cv2.putText(vis, label, (x, image_size[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        return vis


class BehaviorMetrics:
    @staticmethod
    def calculate_entropy(behavior_counts: Dict[BehaviorType, int]) -> float:
        total = sum(behavior_counts.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in behavior_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def calculate_regularity(profile: BehaviorProfile) -> float:
        if not profile.behavior_counts:
            return 0.0

        total = sum(profile.behavior_counts.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in profile.behavior_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        max_entropy = math.log2(len(BehaviorType))
        return 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0

    @staticmethod
    def calculate_stability(profile: BehaviorProfile, previous_profile: BehaviorProfile) -> float:
        if not profile.behavior_counts or not previous_profile.behavior_counts:
            return 0.0

        all_behaviors = set(profile.behavior_counts.keys()) | set(previous_profile.behavior_counts.keys())

        vec1 = np.array([profile.behavior_counts.get(bt, 0) for bt in all_behaviors])
        vec2 = np.array([previous_profile.behavior_counts.get(bt, 0) for bt in all_behaviors])

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))


class BehaviorSerializer:
    @staticmethod
    def to_json(events: List[BehaviorEvent]) -> str:
        import json
        data = []
        for e in events:
            data.append({
                'event_id': e.event_id,
                'entity_id': e.entity_id,
                'behavior_type': e.behavior_type.value,
                'timestamp': e.timestamp,
                'position': e.position.tolist() if e.position is not None else None,
                'features': e.features,
                'duration': e.duration,
                'confidence': e.confidence
            })
        return json.dumps(data, indent=2)

    @staticmethod
    def from_json(json_str: str) -> List[BehaviorEvent]:
        import json
        data = json.loads(json_str)
        events = []
        for d in data:
            events.append(BehaviorEvent(
                event_id=d['event_id'],
                entity_id=d['entity_id'],
                behavior_type=BehaviorType(d['behavior_type']),
                timestamp=d['timestamp'],
                position=np.array(d['position']) if d['position'] else None,
                features=d['features'],
                duration=d['duration'],
                confidence=d['confidence']
            ))
        return events


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet