from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class ZoneType(str, Enum):
    INCLUSION = "inclusion"
    RESTRICTED = "restricted"
    EXCLUSION = "exclusion"
    CRITICAL = "critical"


class BreachSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class GeofenceBreachEvent:
    zone_id: str
    track_id: int
    severity: BreachSeverity
    timestamp: float
    point_2d: tuple[float, float]
    dwell_duration_seconds: float
    message: str


class GeofenceZone:
    """Arbitrary 2D/3D Polygon Geofence with ray-casting containment and dwell-time tracking."""

    def __init__(
        self,
        zone_id: str,
        name: str,
        polygon_vertices: list[tuple[float, float]],
        zone_type: ZoneType = ZoneType.RESTRICTED,
        dwell_time_threshold_seconds: float = 3.0,
        min_altitude: float | None = None,
        max_altitude: float | None = None,
    ) -> None:
        if len(polygon_vertices) < 3:
            raise ValueError("Polygon geofence must have at least 3 vertices")
        self.zone_id = zone_id
        self.name = name
        self.vertices = polygon_vertices
        self.zone_type = zone_type
        self.dwell_threshold = dwell_time_threshold_seconds
        self.min_alt = min_altitude
        self.max_alt = max_altitude
        self._dwell_tracker: dict[int, float] = {}  # track_id -> first_seen_time

    def contains_point(self, point: tuple[float, float], altitude: float | None = None) -> bool:
        """Ray-casting algorithm for point-in-polygon containment."""
        if altitude is not None and (
            (self.min_alt is not None and altitude < self.min_alt)
            or (self.max_alt is not None and altitude > self.max_alt)
        ):
            return False

        px, py = point
        n = len(self.vertices)
        inside = False

        p1x, p1y = self.vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = self.vertices[i % n]
            if (
                py > min(p1y, p2y)
                and py <= max(p1y, p2y)
                and px <= max(p1x, p2x)
                and p1y != p2y
            ):
                x_inters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or px <= x_inters:
                    inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def evaluate_track(
        self,
        track_id: int,
        point: tuple[float, float],
        altitude: float | None = None,
        now: float | None = None,
    ) -> GeofenceBreachEvent | None:
        now_ts = now if now is not None else time.monotonic()
        is_inside = self.contains_point(point, altitude)

        if not is_inside:
            self._dwell_tracker.pop(track_id, None)
            return None

        first_seen = self._dwell_tracker.get(track_id)
        if first_seen is None:
            self._dwell_tracker[track_id] = now_ts
            dwell = 0.0
        else:
            dwell = now_ts - first_seen

        # Immediate alert for EXCLUSION or CRITICAL zones, or when dwell threshold is breached
        if self.zone_type in (ZoneType.EXCLUSION, ZoneType.CRITICAL) or dwell >= self.dwell_threshold:
            severity = (
                BreachSeverity.CRITICAL if self.zone_type == ZoneType.CRITICAL
                else BreachSeverity.HIGH if self.zone_type == ZoneType.EXCLUSION
                else BreachSeverity.MEDIUM if dwell >= self.dwell_threshold * 2
                else BreachSeverity.LOW
            )
            return GeofenceBreachEvent(
                zone_id=self.zone_id,
                track_id=track_id,
                severity=severity,
                timestamp=now_ts,
                point_2d=point,
                dwell_duration_seconds=round(dwell, 2),
                message=f"Track #{track_id} breached {self.zone_type.value} zone '{self.name}' (dwell: {dwell:.1f}s)",
            )

        return None


class TripwireDetector:
    """Directional 2D Vector Tripwire for boundary crossing detection."""

    def __init__(self, wire_id: str, point_a: tuple[float, float], point_b: tuple[float, float]) -> None:
        self.wire_id = wire_id
        self.pt_a = point_a
        self.pt_b = point_b

    @staticmethod
    def _ccw(p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]) -> bool:
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

    def has_crossed(self, prev_pos: tuple[float, float], curr_pos: tuple[float, float]) -> bool:
        """Check whether trajectory segment prev->curr intersects the tripwire segment A->B."""
        return (
            self._ccw(self.pt_a, prev_pos, curr_pos) != self._ccw(self.pt_b, prev_pos, curr_pos)
            and self._ccw(self.pt_a, self.pt_b, prev_pos) != self._ccw(self.pt_a, self.pt_b, curr_pos)
        )
