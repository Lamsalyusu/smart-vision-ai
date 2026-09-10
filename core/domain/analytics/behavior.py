from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple


class AnomalyType(str, Enum):
    RAPID_ACCELERATION = "rapid_acceleration"
    ERRATIC_MOTION = "erratic_motion"
    LOITERING = "loitering"
    SUDDEN_STOP = "sudden_stop"


@dataclass(frozen=True, slots=True)
class BehavioralAnomalyEvent:
    track_id: int
    anomaly_type: AnomalyType
    confidence: float
    description: str
    timestamp: float


class TrajectoryBehaviorAnalyzer:
    """Detects kinematic anomalies, loitering, and erratic trajectory curvature."""

    def __init__(
        self,
        history_window: int = 15,
        max_speed_threshold: float = 8.0,       # m/s or pixel/frame
        max_accel_threshold: float = 6.0,       # m/s^2
        loitering_time_seconds: float = 5.0,
        loitering_radius: float = 1.5,
    ) -> None:
        self.window = history_window
        self.max_speed = max_speed_threshold
        self.max_accel = max_accel_threshold
        self.loiter_time = loitering_time_seconds
        self.loiter_radius = loitering_radius
        self._tracks_history: Dict[int, Deque[Tuple[float, float, float]]] = {}  # track_id -> deque of (x, y, ts)

    def update_position(self, track_id: int, pos: Tuple[float, float], timestamp: float) -> List[BehavioralAnomalyEvent]:
        anomalies: List[BehavioralAnomalyEvent] = []
        if track_id not in self._tracks_history:
            self._tracks_history[track_id] = deque(maxlen=self.window)

        history = self._tracks_history[track_id]
        history.append((pos[0], pos[1], timestamp))

        if len(history) < 3:
            return anomalies

        # 1. Compute velocity & acceleration
        p1, p2, p3 = history[-3], history[-2], history[-1]
        dt1 = max(1e-3, p2[2] - p1[2])
        dt2 = max(1e-3, p3[2] - p2[2])

        v1 = math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / dt1
        v2 = math.hypot(p3[0] - p2[0], p3[1] - p2[1]) / dt2

        accel = abs(v2 - v1) / dt2
        if accel > self.max_accel:
            anomalies.append(
                BehavioralAnomalyEvent(
                    track_id=track_id,
                    anomaly_type=AnomalyType.RAPID_ACCELERATION,
                    confidence=min(1.0, accel / (self.max_accel * 1.5)),
                    description=f"Track #{track_id} acceleration spike {accel:.1f} exceeds threshold",
                    timestamp=timestamp,
                )
            )

        # 2. Curvature / Erratic motion
        if len(history) >= 4:
            v_vec1 = (p2[0] - p1[0], p2[1] - p1[1])
            v_vec2 = (p3[0] - p2[0], p3[1] - p2[1])
            dot = v_vec1[0] * v_vec2[0] + v_vec1[1] * v_vec2[1]
            n1 = math.hypot(v_vec1[0], v_vec1[1])
            n2 = math.hypot(v_vec2[0], v_vec2[1])
            if n1 > 0.1 and n2 > 0.1:
                cos_angle = max(-1.0, min(1.0, dot / (n1 * n2)))
                heading_change = math.acos(cos_angle)
                if heading_change > (math.pi * 0.75):  # sharp turn > 135 degrees
                    anomalies.append(
                        BehavioralAnomalyEvent(
                            track_id=track_id,
                            anomaly_type=AnomalyType.ERRATIC_MOTION,
                            confidence=0.85,
                            description=f"Track #{track_id} sharp erratic trajectory reversal",
                            timestamp=timestamp,
                        )
                    )

        # 3. Loitering Detection
        first_p = history[0]
        time_span = history[-1][2] - first_p[2]
        if time_span >= self.loiter_time:
            # Check maximum displacement from origin
            max_dist = max(math.hypot(p[0] - first_p[0], p[1] - first_p[1]) for p in history)
            if max_dist <= self.loiter_radius:
                anomalies.append(
                    BehavioralAnomalyEvent(
                        track_id=track_id,
                        anomaly_type=AnomalyType.LOITERING,
                        confidence=0.90,
                        description=f"Track #{track_id} stationary/loitering for {time_span:.1f}s within {max_dist:.1f}m",
                        timestamp=timestamp,
                    )
                )

        return anomalies

    def purge_track(self, track_id: int) -> None:
        self._tracks_history.pop(track_id, None)
