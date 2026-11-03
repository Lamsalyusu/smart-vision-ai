from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Dict, Any, List
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math
import time
import uuid


class AnalyticsEventType(Enum):
    OBJECT_DETECTED = "object_detected"
    OBJECT_TRACKED = "object_tracked"
    ZONE_ENTER = "zone_enter"
    ZONE_EXIT = "zone_exit"
    ZONE_DWELL = "zone_dwell"
    LINE_CROSS = "line_cross"
    LOITERING = "loitering"
    CROWD_FORMATION = "crowd_formation"
    WRONG_WAY = "wrong_way"
    SPEEDING = "speeding"
    STOPPED_VEHICLE = "stopped_vehicle"
    ABANDONED_OBJECT = "abandoned_object"
    REMOVED_OBJECT = "removed_object"
    TAILGATING = "tailgating"
    FACE_DETECTED = "face_detected"
    PERSON_COUNT = "person_count"
    VEHICLE_COUNT = "vehicle_count"


@dataclass(frozen=True, slots=True)
class AnalyticsEvent:
    event_type: AnalyticsEventType
    timestamp: float
    track_id: Optional[str] = None
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    position: Optional[np.ndarray] = None
    zone_id: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})


@dataclass(frozen=True, slots=True)
class Zone:
    zone_id: str
    name: str
    polygon: np.ndarray
    zone_type: str = "general"
    entry_direction: Optional[np.ndarray] = None
    exit_direction: Optional[np.ndarray] = None
    max_capacity: Optional[int] = None
    dwell_threshold: float = 30.0
    alert_on_entry: bool = False
    alert_on_exit: bool = False
    alert_on_dwell: bool = False


@dataclass(frozen=True, slots=True)
class Tripwire:
    tripwire_id: str
    name: str
    start_point: np.ndarray
    end_point: np.ndarray
    direction: str = "both"
    alert_on_cross: bool = True


class ZoneAnalyzer:
    def __init__(self, zones: List[Zone]):
        self.zones = {z.zone_id: z for z in zones}
        self.track_states: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'inside_zones': set(),
            'entry_times': {},
            'last_position': None,
        })

    def update(self, tracks: List[Dict[str, Any]]) -> List[AnalyticsEvent]:
        events = []
        current_inside = defaultdict(set)

        for track in tracks:
            track_id = track.get('track_id')
            position = track.get('position')
            if position is None:
                continue

            state = self.track_states[track_id]
            state['last_position'] = position

            point = position[:2] if len(position) >= 2 else position

            for zone_id, zone in self.zones.items():
                inside = cv2.pointPolygonTest(zone.polygon, (float(point[0]), float(point[1])), False) >= 0

                if inside:
                    current_inside[track_id].add(zone_id)
                    if zone_id not in state['inside_zones']:
                        state['inside_zones'].add(zone_id)
                        state['entry_times'][zone_id] = track.get('timestamp', time.time())
                        if zone.alert_on_entry:
                            events.append(AnalyticsEvent(
                                event_type=AnalyticsEventType.ZONE_ENTER,
                                timestamp=track.get('timestamp', time.time()),
                                track_id=track_id,
                                class_id=track.get('class_id'),
                                class_name=track.get('class_name'),
                                position=position,
                                zone_id=zone_id,
                            ))

                    dwell_time = track.get('timestamp', time.time()) - state['entry_times'].get(zone_id, 0)
                    if dwell_time > zone.dwell_threshold and zone.alert_on_dwell:
                        events.append(AnalyticsEvent(
                            event_type=AnalyticsEventType.ZONE_DWELL,
                            timestamp=track.get('timestamp', time.time()),
                            track_id=track_id,
                            class_id=track.get('class_id'),
                            class_name=track.get('class_name'),
                            position=position,
                            zone_id=zone_id,
                            metadata={'dwell_time': dwell_time},
                        ))

        for track_id, state in self.track_states.items():
            for zone_id in state['inside_zones'] - current_inside.get(track_id, set()):
                zone = self.zones[zone_id]
                if zone.alert_on_exit:
                    events.append(AnalyticsEvent(
                        event_type=AnalyticsEventType.ZONE_EXIT,
                        timestamp=time.time(),
                        track_id=track_id,
                        zone_id=zone_id,
                    ))
                state['inside_zones'].remove(zone_id)
                if zone_id in state['entry_times']:
                    del state['entry_times'][zone_id]

        return events


class TripwireAnalyzer:
    def __init__(self, tripwires: List[Tripwire]):
        self.tripwires = {t.tripwire_id: t for t in tripwires}
        self.track_positions: Dict[str, np.ndarray] = {}

    def update(self, tracks: List[Dict[str, Any]]) -> List[AnalyticsEvent]:
        events = []

        for track in tracks:
            track_id = track.get('track_id')
            position = track.get('position')
            if position is None:
                continue

            point = position[:2] if len(position) >= 2 else position

            if track_id in self.track_positions:
                prev_point = self.track_positions[track_id]

                for tw_id, tripwire in self.tripwires.items():
                    if not tripwire.alert_on_cross:
                        continue

                    crossed, direction = self._check_crossing(prev_point, point, tripwire)
                    if crossed:
                        if tripwire.direction == "both" or tripwire.direction == direction:
                            events.append(AnalyticsEvent(
                                event_type=AnalyticsEventType.LINE_CROSS,
                                timestamp=track.get('timestamp', time.time()),
                                track_id=track_id,
                                class_id=track.get('class_id'),
                                class_name=track.get('class_name'),
                                position=position,
                                metadata={
                                    'tripwire_id': tw_id,
                                    'direction': direction,
                                },
                            ))

            self.track_positions[track_id] = point

        return events

    def _check_crossing(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        tripwire: Tripwire,
    ) -> tuple[bool, str]:
        x1, y1 = tripwire.start_point
        x2, y2 = tripwire.end_point
        x3, y3 = p1
        x4, y4 = p2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return False, ""

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            dx = x2 - x1
            dy = y2 - y1
            cross = dx * (y4 - y1) - dy * (x4 - x1)
            direction = "A_to_B" if cross > 0 else "B_to_A"
            return True, direction

        return False, ""


class LoiteringDetector:
    def __init__(
        self,
        min_dwell_time: float = 30.0,
        max_movement_radius: float = 2.0,
        min_detections: int = 10,
    ):
        self.min_dwell_time = min_dwell_time
        self.max_movement_radius = max_movement_radius
        self.min_detections = min_detections
        self.track_history: Dict[str, List[Dict]] = defaultdict(list)

    def update(self, tracks: List[Dict[str, Any]]) -> List[AnalyticsEvent]:
        events = []
        current_time = time.time()

        for track in tracks:
            track_id = track.get('track_id')
            position = track.get('position')
            if position is None:
                continue

            self.track_history[track_id].append({
                'position': position,
                'timestamp': track.get('timestamp', current_time),
            })

            history = self.track_history[track_id]
            if len(history) < self.min_detections:
                continue

            recent = [h for h in history if current_time - h['timestamp'] < self.min_dwell_time]
            if len(recent) < self.min_detections:
                continue

            positions = np.array([h['position'][:2] for h in recent])
            centroid = np.mean(positions, axis=0)
            distances = np.linalg.norm(positions - centroid, axis=1)
            max_dist = np.max(distances)

            if max_dist < self.max_movement_radius:
                events.append(AnalyticsEvent(
                    event_type=AnalyticsEventType.LOITERING,
                    timestamp=current_time,
                    track_id=track_id,
                    class_id=track.get('class_id'),
                    class_name=track.get('class_name'),
                    position=position,
                    metadata={
                        'dwell_time': current_time - recent[0]['timestamp'],
                        'max_movement': float(max_dist),
                        'centroid': centroid.tolist(),
                    },
                ))

        return events


class CrowdDetector:
    def __init__(
        self,
        min_group_size: int = 5,
        max_distance: float = 3.0,
        min_duration: float = 10.0,
    ):
        self.min_group_size = min_group_size
        self.max_distance = max_distance
        self.min_duration = min_duration
        self.groups: Dict[str, Dict] = {}
        self.next_group_id = 0

    def update(self, tracks: List[Dict[str, Any]]) -> List[AnalyticsEvent]:
        events = []
        current_time = time.time()

        person_tracks = [t for t in tracks if t.get('class_name') == 'person']

        if len(person_tracks) < self.min_group_size:
            self.groups.clear()
            return events

        positions = np.array([t['position'][:2] for t in person_tracks])
        track_ids = [t['track_id'] for t in person_tracks]

        clusters = self._cluster_positions(positions, track_ids)

        for cluster in clusters:
            if len(cluster) >= self.min_group_size:
                group_id = self._get_or_create_group(cluster, current_time)
                group = self.groups[group_id]
                group['last_seen'] = current_time
                group['members'] = cluster

                if current_time - group['first_seen'] > self.min_duration:
                    events.append(AnalyticsEvent(
                        event_type=AnalyticsEventType.CROWD_FORMATION,
                        timestamp=current_time,
                        metadata={
                            'group_id': group_id,
                            'size': len(cluster),
                            'centroid': group['centroid'].tolist(),
                            'duration': current_time - group['first_seen'],
                        },
                    ))

        to_remove = []
        for gid, group in self.groups.items():
            if current_time - group['last_seen'] > self.min_duration * 2:
                to_remove.append(gid)
        for gid in to_remove:
            del self.groups[gid]

        return events

    def _cluster_positions(self, positions: np.ndarray, track_ids: List[str]) -> List[List[str]]:
        if len(positions) == 0:
            return []

        clusters = []
        visited = set()

        for i, tid in enumerate(track_ids):
            if tid in visited:
                continue

            cluster = [tid]
            visited.add(tid)

            for j, other_tid in enumerate(track_ids):
                if other_tid in visited:
                    continue
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < self.max_distance:
                    cluster.append(other_tid)
                    visited.add(other_tid)

            clusters.append(cluster)

        return clusters

    def _get_or_create_group(self, cluster: List[str], timestamp: float) -> str:
        for gid, group in self.groups.items():
            overlap = set(cluster) & set(group['members'])
            if len(overlap) >= len(cluster) * 0.5:
                return gid

        gid = f"group_{self.next_group_id}"
        self.next_group_id += 1
        positions = [t['position'][:2] for t in cluster if 'position' in t]
        centroid = np.mean(positions, axis=0) if positions else np.zeros(2)
        self.groups[gid] = {
            'members': cluster,
            'centroid': centroid,
            'first_seen': timestamp,
            'last_seen': timestamp,
        }
        return gid


class SpeedEstimator:
    def __init__(
        self,
        pixel_to_meter: float = 0.05,
        smoothing_window: int = 5,
    ):
        self.pixel_to_meter = pixel_to_meter
        self.smoothing_window = smoothing_window
        self.track_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=smoothing_window))

    def update(self, tracks: List[Dict[str, Any]]) -> List[AnalyticsEvent]:
        events = []

        for track in tracks:
            track_id = track.get('track_id')
            position = track.get('position')
            timestamp = track.get('timestamp', time.time())
            class_name = track.get('class_name')

            if position is None or class_name not in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                continue

            self.track_history[track_id].append({
                'position': position[:2],
                'timestamp': timestamp,
            })

            history = self.track_history[track_id]
            if len(history) < 2:
                continue

            pos_now = history[-1]['position']
            time_now = history[-1]['timestamp']
            pos_prev = history[0]['position']
            time_prev = history[0]['timestamp']

            dt = time_now - time_prev
            if dt < 0.1:
                continue

            distance_px = np.linalg.norm(pos_now - pos_prev)
            distance_m = distance_px * self.pixel_to_meter
            speed_mps = distance_m / dt
            speed_kmh = speed_mps * 3.6

            if speed_kmh > 50:
                events.append(AnalyticsEvent(
                    event_type=AnalyticsEventType.SPEEDING,
                    timestamp=timestamp,
                    track_id=track_id,
                    class_id=track.get('class_id'),
                    class_name=class_name,
                    position=position,
                    metadata={
                        'speed_kmh': speed_kmh,
                        'speed_mps': speed_mps,
                    },
                ))

        return events


class DirectionAnalyzer:
    def __init__(
        self,
        allowed_directions: Dict[str, np.ndarray],
        angle_threshold: float = 45.0,
    ):
        self.allowed_directions = allowed_directions
        self.angle_threshold = angle_threshold
        self.track_velocities: Dict[str, np.ndarray] = {}

    def update(self, tracks: List[Dict[str, Any]]) -> List[AnalyticsEvent]:
        events = []
        current_time = time.time()

        for track in tracks:
            track_id = track.get('track_id')
            position = track.get('position')
            class_name = track.get('class_name')
            zone_id = track.get('zone_id')

            if position is None or zone_id not in self.allowed_directions:
                continue

            if track_id in self.track_velocities:
                prev_pos = self.track_velocities[track_id]
                velocity = position[:2] - prev_pos
                speed = np.linalg.norm(velocity)

                if speed > 0.5:
                    direction = velocity / speed
                    allowed = self.allowed_directions[zone_id]
                    angle = math.degrees(math.acos(np.clip(np.dot(direction, allowed), -1, 1)))

                    if angle > self.angle_threshold:
                        events.append(AnalyticsEvent(
                            event_type=AnalyticsEventType.WRONG_WAY,
                            timestamp=current_time,
                            track_id=track_id,
                            class_id=track.get('class_id'),
                            class_name=class_name,
                            position=position,
                            metadata={
                                'zone_id': zone_id,
                                'angle': angle,
                                'speed': speed,
                            },
                        ))

            self.track_velocities[track_id] = position[:2]

        return events


class AbandonedObjectDetector:
    def __init__(
        self,
        min_stationary_time: float = 60.0,
        max_movement: float = 1.0,
        min_size: float = 0.01,
        max_size: float = 1.0,
    ):
        self.min_stationary_time = min_stationary_time
        self.max_movement = max_movement
        self.min_size = min_size
        self.max_size = max_size
        self.object_history: Dict[str, Dict] = {}

    def update(self, tracks: List[Dict[str, Any]]) -> List[AnalyticsEvent]:
        events = []
        current_time = time.time()

        for track in tracks:
            track_id = track.get('track_id')
            position = track.get('position')
            class_name = track.get('class_name')
            bbox = track.get('bbox')

            if class_name not in ['suitcase', 'backpack', 'handbag', 'box', 'package']:
                continue

            if bbox is not None:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                area = w * h
                if area < self.min_size or area > self.max_size:
                    continue

            if track_id not in self.object_history:
                self.object_history[track_id] = {
                    'position': position[:2] if position is not None else None,
                    'first_seen': current_time,
                    'last_moved': current_time,
                    'class_name': class_name,
                }
                continue

            history = self.object_history[track_id]
            if position is not None:
                moved = np.linalg.norm(position[:2] - history['position'])
                if moved > self.max_movement:
                    history['position'] = position[:2]
                    history['last_moved'] = current_time
                    continue

            stationary_time = current_time - history['last_moved']
            if stationary_time > self.min_stationary_time:
                events.append(AnalyticsEvent(
                    event_type=AnalyticsEventType.ABANDONED_OBJECT,
                    timestamp=current_time,
                    track_id=track_id,
                    class_id=track.get('class_id'),
                    class_name=class_name,
                    position=position,
                    metadata={
                        'stationary_time': stationary_time,
                        'object_class': class_name,
                    },
                ))

        return events


class AnalyticsEngine:
    def __init__(self):
        self.analyzers: List[Any] = []
        self.event_handlers: Dict[AnalyticsEventType, List[Callable]] = defaultdict(list)

    def add_analyzer(self, analyzer: Any):
        self.analyzers.append(analyzer)

    def register_handler(self, event_type: AnalyticsEventType, handler: Callable):
        self.event_handlers[event_type].append(handler)

    def process(self, tracks: List[Dict[str, Any]]) -> List[AnalyticsEvent]:
        all_events = []

        for analyzer in self.analyzers:
            events = analyzer.update(tracks)
            all_events.extend(events)

        for event in all_events:
            for handler in self.event_handlers.get(event.event_type, []):
                try:
                    handler(event)
                except Exception:
                    pass

        return all_events

    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            'total_analyzers': len(self.analyzers),
            'analyzer_types': [type(a).__name__ for a in self.analyzers],
        }
        return stats


class BehaviorAnalyzer:
    def __init__(
        self,
        behavior_patterns: Dict[str, List[AnalyticsEventType]],
        sequence_window: float = 10.0,
    ):
        self.behavior_patterns = behavior_patterns
        self.sequence_window = sequence_window
        self.track_sequences: Dict[str, List[AnalyticsEvent]] = defaultdict(list)

    def process_event(self, event: AnalyticsEvent) -> List[str]:
        if event.track_id is None:
            return []

        self.track_sequences[event.track_id].append(event)
        current_time = event.timestamp

        self.track_sequences[event.track_id] = [
            e for e in self.track_sequences[event.track_id]
            if current_time - e.timestamp < self.sequence_window
        ]

        detected_behaviors = []
        for behavior_name, pattern in self.behavior_patterns.items():
            if self._match_pattern(self.track_sequences[event.track_id], pattern):
                detected_behaviors.append(behavior_name)

        return detected_behaviors

    def _match_pattern(self, sequence: List[AnalyticsEvent], pattern: List[AnalyticsEventType]) -> bool:
        if len(sequence) < len(pattern):
            return False

        seq_types = [e.event_type for e in sequence]
        return self._contains_subsequence(seq_types, pattern)

    def _contains_subsequence(self, seq: List, pattern: List) -> bool:
        if not pattern:
            return True
        if not seq:
            return False

        i = 0
        for item in seq:
            if item == pattern[i]:
                i += 1
                if i == len(pattern):
                    return True
        return False
