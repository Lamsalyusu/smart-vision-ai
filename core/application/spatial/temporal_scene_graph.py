from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True, slots=True)
class SpatialEntity:
    track_id: int
    class_name: str
    position_xyz_m: tuple[float, float, float]
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float = 1.0

    def __post_init__(self) -> None:
        values = (*self.position_xyz_m, *self.bbox_xyxy, self.confidence)
        if self.track_id < 0 or not self.class_name:
            raise ValueError("track id and class name are required")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("spatial entity values must be finite")
        if self.bbox_xyxy[0] >= self.bbox_xyxy[2] or self.bbox_xyxy[1] >= self.bbox_xyxy[3]:
            raise ValueError("entity bounding box must have positive dimensions")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between zero and one")


@dataclass(frozen=True, slots=True)
class HazardZone:
    zone_id: str
    polygon_xz_m: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not self.zone_id or len(self.polygon_xz_m) < 3:
            raise ValueError("hazard zone needs an id and at least three polygon vertices")
        if not all(math.isfinite(value) for point in self.polygon_xz_m for value in point):
            raise ValueError("hazard polygon coordinates must be finite")


@dataclass(frozen=True, slots=True)
class TemporalSpatialEdge:
    source_id: str
    target_id: str
    relation: str
    distance_m: float | None = None


@dataclass(frozen=True, slots=True)
class TemporalSceneSnapshot:
    timestamp_seconds: float
    nodes: tuple[SpatialEntity, ...]
    edges: tuple[TemporalSpatialEdge, ...]


class TemporalSceneGraph:
    """Build frame-level spatial relations and retain bounded temporal history."""

    def __init__(
        self,
        *,
        history_capacity: int = 120,
        approaching_epsilon_m: float = 0.02,
        proximity_threshold_m: float = 2.0,
        occlusion_overlap_threshold: float = 0.5,
        minimum_occlusion_depth_delta_m: float = 0.1,
    ) -> None:
        if history_capacity < 2:
            raise ValueError("history capacity must be at least two snapshots")
        thresholds = (
            approaching_epsilon_m,
            proximity_threshold_m,
            occlusion_overlap_threshold,
            minimum_occlusion_depth_delta_m,
        )
        if not all(math.isfinite(value) and value >= 0 for value in thresholds):
            raise ValueError("graph thresholds must be finite and non-negative")
        if occlusion_overlap_threshold > 1:
            raise ValueError("occlusion overlap threshold must not exceed one")
        self._history: deque[TemporalSceneSnapshot] = deque(maxlen=history_capacity)
        self._approaching_epsilon = approaching_epsilon_m
        self._proximity_threshold = proximity_threshold_m
        self._occlusion_overlap_threshold = occlusion_overlap_threshold
        self._minimum_depth_delta = minimum_occlusion_depth_delta_m

    @property
    def history(self) -> tuple[TemporalSceneSnapshot, ...]:
        return tuple(self._history)

    def update(
        self,
        timestamp_seconds: float,
        entities: Sequence[SpatialEntity],
        hazard_zones: Sequence[HazardZone] = (),
    ) -> TemporalSceneSnapshot:
        if not math.isfinite(timestamp_seconds) or timestamp_seconds < 0:
            raise ValueError("timestamp must be finite and non-negative")
        if self._history and timestamp_seconds <= self._history[-1].timestamp_seconds:
            raise ValueError("snapshot timestamps must increase monotonically")
        entity_ids = [entity.track_id for entity in entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("track ids must be unique within a snapshot")

        edges: list[TemporalSpatialEdge] = []
        previous_entities = (
            {entity.track_id: entity for entity in self._history[-1].nodes}
            if self._history
            else {}
        )
        for index, first in enumerate(entities):
            for second in entities[index + 1 :]:
                distance = self._distance(first, second)
                if distance <= self._proximity_threshold:
                    edges.append(
                        TemporalSpatialEdge(
                            str(first.track_id), str(second.track_id), "near", distance
                        )
                    )
                previous_first = previous_entities.get(first.track_id)
                previous_second = previous_entities.get(second.track_id)
                if previous_first is not None and previous_second is not None:
                    previous_distance = self._distance(previous_first, previous_second)
                    if previous_distance - distance > self._approaching_epsilon:
                        edges.append(
                            TemporalSpatialEdge(
                                str(first.track_id), str(second.track_id), "approaching", distance
                            )
                        )

                overlap = self._overlap_ratio(first.bbox_xyxy, second.bbox_xyxy)
                depth_delta = first.position_xyz_m[2] - second.position_xyz_m[2]
                if (
                    overlap >= self._occlusion_overlap_threshold
                    and abs(depth_delta) >= self._minimum_depth_delta
                ):
                    nearer, farther = (first, second) if depth_delta < 0 else (second, first)
                    edges.append(
                        TemporalSpatialEdge(
                            str(nearer.track_id), str(farther.track_id), "occluding", distance
                        )
                    )

        for entity in entities:
            for zone in hazard_zones:
                if self._contains(zone.polygon_xz_m, entity.position_xyz_m[0], entity.position_xyz_m[2]):
                    edges.append(
                        TemporalSpatialEdge(
                            str(entity.track_id), zone.zone_id, "inside_hazard_zone"
                        )
                    )

        snapshot = TemporalSceneSnapshot(timestamp_seconds, tuple(entities), tuple(edges))
        self._history.append(snapshot)
        return snapshot

    def query(
        self,
        relation: str,
        *,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
    ) -> tuple[tuple[float, TemporalSpatialEdge], ...]:
        if start_seconds is not None and not math.isfinite(start_seconds):
            raise ValueError("query start must be finite")
        if end_seconds is not None and not math.isfinite(end_seconds):
            raise ValueError("query end must be finite")
        if start_seconds is not None and end_seconds is not None and start_seconds > end_seconds:
            raise ValueError("query start must not be after query end")
        return tuple(
            (snapshot.timestamp_seconds, edge)
            for snapshot in self._history
            if (start_seconds is None or snapshot.timestamp_seconds >= start_seconds)
            and (end_seconds is None or snapshot.timestamp_seconds <= end_seconds)
            for edge in snapshot.edges
            if edge.relation == relation
        )

    @staticmethod
    def _distance(first: SpatialEntity, second: SpatialEntity) -> float:
        return math.sqrt(
            sum(
                (left - right) ** 2
                for left, right in zip(first.position_xyz_m, second.position_xyz_m)
            )
        )

    @staticmethod
    def _overlap_ratio(first: tuple[float, ...], second: tuple[float, ...]) -> float:
        intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
        intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
        intersection = intersection_width * intersection_height
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        denominator = min(first_area, second_area)
        return intersection / denominator if denominator > 0 else 0.0

    @staticmethod
    def _contains(polygon: Sequence[tuple[float, float]], x: float, z: float) -> bool:
        inside = False
        previous_x, previous_z = polygon[-1]
        for current_x, current_z in polygon:
            crosses_height = (current_z > z) != (previous_z > z)
            if crosses_height:
                intersection_x = (previous_x - current_x) * (z - current_z) / (
                    previous_z - current_z
                ) + current_x
                if x < intersection_x:
                    inside = not inside
            previous_x, previous_z = current_x, current_z
        return inside