from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
import random
from collections import defaultdict


class SemanticClass(Enum):
    UNKNOWN = "unknown"
    WALL = "wall"
    FLOOR = "floor"
    CEILING = "ceiling"
    DOOR = "door"
    WINDOW = "window"
    TABLE = "table"
    CHAIR = "chair"
    PERSON = "person"
    VEHICLE = "vehicle"
    ROAD = "road"
    SIDEWALK = "sidewalk"
    BUILDING = "building"
    TREE = "tree"
    SKY = "sky"
    GROUND = "ground"


class MapType(Enum):
    SEMANTIC = "semantic"
    TOPOLOGICAL = "topological"
    METRIC = "metric"
    HYBRID = "hybrid"
    LAYERED = "layered"


@dataclass(frozen=True, slots=True)
class SemanticEntity:
    entity_id: int
    class_id: SemanticClass
    position: np.ndarray
    orientation: np.ndarray
    dimensions: np.ndarray
    confidence: float
    timestamp: float
    attributes: Dict[str, Any] = field(default_factory=dict)
    observations: int = 1


@dataclass(frozen=True, slots=True)
class SemanticMap:
    entities: Dict[int, SemanticEntity]
    grid_resolution: float
    origin: np.ndarray
    bounds: Tuple[np.ndarray, np.ndarray]
    entity_counter: int = 0


@dataclass(frozen=True, slots=True)
class Place:
    place_id: int
    name: str
    center: np.ndarray
    extent: np.ndarray
    entities: List[int]
    connections: List[int]
    semantic_label: Optional[str] = None


@dataclass(frozen=True, slots=True)
class TopologicalMap:
    places: Dict[int, Place]
    edges: Dict[Tuple[int, int], Dict[str, Any]]
    place_counter: int = 0


class SemanticMapper:
    def __init__(
        self,
        grid_resolution: float = 0.1,
        min_confidence: float = 0.5,
        max_entities: int = 10000
    ):
        self.grid_resolution = grid_resolution
        self.min_confidence = min_confidence
        self.max_entities = max_entities

        self.semantic_map = SemanticMap(
            entities={},
            grid_resolution=grid_resolution,
            origin=np.zeros(3),
            bounds=(np.zeros(3), np.zeros(3))
        )
        self.entity_counter = 0

    def add_observation(
        self,
        detections: List[Dict[str, Any]],
        camera_pose: np.ndarray,
        camera_intrinsics: np.ndarray,
        depth_image: Optional[np.ndarray] = None
    ) -> List[int]:
        entity_ids = []

        for det in detections:
            if det.get('confidence', 0) < self.min_confidence:
                continue

            class_id = SemanticClass(det.get('class', 'unknown'))
            bbox = det.get('bbox', [0, 0, 0, 0])

            if depth_image is not None:
                depth = self._get_depth_from_bbox(depth_image, bbox)
            else:
                depth = 5.0

            position = self._bbox_to_position(bbox, depth, camera_intrinsics, camera_pose)

            entity_id = self._find_or_create_entity(class_id, position, det.get('confidence', 0.5))
            entity_ids.append(entity_id)

        return entity_ids

    def _get_depth_from_bbox(self, depth_image: np.ndarray, bbox: List[float]) -> float:
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, depth_image.shape[1] - 1))
        x2 = max(0, min(x2, depth_image.shape[1] - 1))
        y1 = max(0, min(y1, depth_image.shape[0] - 1))
        y2 = max(0, min(y2, depth_image.shape[0] - 1))

        roi = depth_image[y1:y2, x1:x2]
        valid_depths = roi[roi > 0]
        if len(valid_depths) > 0:
            return float(np.median(valid_depths))
        return 5.0

    def _bbox_to_position(
        self,
        bbox: List[float],
        depth: float,
        intrinsics: np.ndarray,
        camera_pose: np.ndarray
    ) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        u = (x1 + x2) / 2
        v = (y1 + y2) / 2

        x = (u - intrinsics[0, 2]) * depth / intrinsics[0, 0]
        y = (v - intrinsics[1, 2]) * depth / intrinsics[1, 1]
        z = depth

        point_cam = np.array([x, y, z, 1])
        point_world = camera_pose @ point_cam

        return point_world[:3]

    def _find_or_create_entity(
        self,
        class_id: SemanticClass,
        position: np.ndarray,
        confidence: float
    ) -> int:
        for entity_id, entity in self.semantic_map.entities.items():
            if entity.class_id == class_id:
                dist = np.linalg.norm(entity.position - position)
                if dist < 1.0:
                    entity.observations += 1
                    entity.confidence = max(entity.confidence, confidence)
                    return entity_id

        if len(self.semantic_map.entities) >= self.max_entities:
            self._remove_oldest_entity()

        entity_id = self.entity_counter
        self.entity_counter += 1

        entity = SemanticEntity(
            entity_id=entity_id,
            class_id=class_id,
            position=position,
            orientation=np.eye(3),
            dimensions=np.ones(3),
            confidence=confidence,
            timestamp=time.time()
        )

        self.semantic_map.entities[entity_id] = entity
        return entity_id

    def _remove_oldest_entity(self):
        if not self.semantic_map.entities:
            return
        oldest = min(self.semantic_map.entities.items(), key=lambda x: x[1].timestamp)
        del self.semantic_map.entities[oldest[0]]

    def get_entities_by_class(self, class_id: SemanticClass) -> List[SemanticEntity]:
        return [e for e in self.semantic_map.entities.values() if e.class_id == class_id]

    def get_entities_in_region(
        self,
        center: np.ndarray,
        radius: float
    ) -> List[SemanticEntity]:
        entities = []
        for entity in self.semantic_map.entities.values():
            if np.linalg.norm(entity.position - center) <= radius:
                entities.append(entity)
        return entities

    def export_grid(self) -> np.ndarray:
        if not self.semantic_map.entities:
            return np.zeros((100, 100, 100), dtype=np.int32)

        positions = np.array([e.position for e in self.semantic_map.entities.values()])
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)

        grid_shape = tuple(np.ceil((max_coords - min_coords) / self.grid_resolution).astype(int))
        grid = np.zeros(grid_shape, dtype=np.int32)

        for entity in self.semantic_map.entities.values():
            idx = ((entity.position - min_coords) / self.grid_resolution).astype(int)
            idx = np.clip(idx, 0, np.array(grid_shape) - 1)
            grid[tuple(idx)] = entity.class_id.value if hasattr(entity.class_id, 'value') else hash(entity.class_id) % 100

        return grid


class TopologicalMapper:
    def __init__(
        self,
        semantic_mapper: SemanticMapper,
        connection_distance: float = 5.0
    ):
        self.semantic_mapper = semantic_mapper
        self.connection_distance = connection_distance
        self.topological_map = TopologicalMap(places={}, edges={})
        self.place_counter = 0

    def build_from_semantic(self):
        entities = list(self.semantic_mapper.semantic_map.entities.values())
        self._cluster_into_places(entities)
        self._connect_places()

    def _cluster_into_places(self, entities: List[SemanticEntity]):
        visited = set()

        for entity in entities:
            if entity.entity_id in visited:
                continue

            cluster = [entity]
            queue = [entity]
            visited.add(entity.entity_id)

            while queue:
                current = queue.pop(0)
                neighbors = self._find_neighbors(current, entities)

                for neighbor in neighbors:
                    if neighbor.entity_id not in visited:
                        visited.add(neighbor.entity_id)
                        queue.append(neighbor)
                        cluster.append(neighbor)

            if len(cluster) >= 3:
                place_id = self.place_counter
                self.place_counter += 1

                center = np.mean([e.position for e in cluster], axis=0)
                extent = np.max([e.position for e in cluster], axis=0) - np.min([e.position for e in cluster], axis=0)

                place = Place(
                    place_id=place_id,
                    name=f"place_{place_id}",
                    center=center,
                    extent=extent,
                    entity_ids=[e.entity_id for e in cluster],
                    connections=[]
                )

                self.topological_map.places[place_id] = place

    def _find_neighbors(self, entity: SemanticEntity, entities: List[SemanticEntity]) -> List[SemanticEntity]:
        neighbors = []
        for other in entities:
            if other.entity_id == entity.entity_id:
                continue
            dist = np.linalg.norm(entity.position - other.position)
            if dist < self.connection_distance:
                neighbors.append(other)
        return neighbors

    def _connect_places(self):
        place_ids = list(self.topological_map.places.keys())

        for i in range(len(place_ids)):
            for j in range(i + 1, len(place_ids)):
                id1, id2 = place_ids[i], place_ids[j]
                place1 = self.topological_map.places[id1]
                place2 = self.topological_map.places[id2]

                dist = np.linalg.norm(place1.center - place2.center)
                if dist < self.connection_distance * 2:
                    edge = {
                        'distance': dist,
                        'type': 'navigable'
                    }
                    self.topological_map.edges[(id1, id2)] = edge
                    self.topological_map.edges[(id2, id1)] = edge

                    place1.connections.append(id2)
                    place2.connections.append(id1)

    def find_path(self, start_id: int, goal_id: int) -> List[int]:
        if start_id not in self.topological_map.places or goal_id not in self.topological_map.places:
            return []

        from heapq import heappush, heappop

        open_set = [(0, start_id)]
        came_from = {}
        g_score = {start_id: 0}
        f_score = {start_id: self._heuristic(start_id, goal_id)}

        while open_set:
            _, current = heappop(open_set)

            if current == goal_id:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_id)
                return path[::-1]

            for neighbor in self.topological_map.places[current].connections:
                if neighbor not in self.topological_map.places:
                    continue

                edge = self.topological_map.edges.get((current, neighbor), {})
                tentative_g = g_score[current] + edge.get('distance', 1.0)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_id)
                    heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def _heuristic(self, a: int, b: int) -> float:
        if a not in self.topological_map.places or b not in self.topological_map.places:
            return 0.0
        return np.linalg.norm(
            self.topological_map.places[a].center - self.topological_map.places[b].center
        )


class SemanticNavigation:
    def __init__(
        self,
        topological_mapper: TopologicalMapper,
        planner_type: str = "astar"
    ):
        self.topological_mapper = topological_mapper
        self.planner_type = planner_type

    def plan_route(
        self,
        start_position: np.ndarray,
        goal_position: np.ndarray
    ) -> Optional[List[np.ndarray]]:
        start_place = self._find_nearest_place(start_position)
        goal_place = self._find_nearest_place(goal_position)

        if start_place is None or goal_place is None:
            return None

        path = self.topological_mapper.find_path(start_place, goal_place)

        if not path:
            return None

        waypoints = []
        for place_id in path:
            waypoints.append(self.topological_mapper.topological_map.places[place_id].center)

        return waypoints

    def _find_nearest_place(self, position: np.ndarray) -> Optional[int]:
        min_dist = float('inf')
        nearest = None

        for place_id, place in self.topological_mapper.topological_map.places.items():
            dist = np.linalg.norm(place.center - position)
            if dist < min_dist:
                min_dist = dist
                nearest = place_id

        return nearest


class SemanticMapUpdater:
    def __init__(
        self,
        semantic_mapper: SemanticMapper,
        decay_rate: float = 0.01,
        merge_threshold: float = 0.5
    ):
        self.semantic_mapper = semantic_mapper
        self.decay_rate = decay_rate
        self.merge_threshold = merge_threshold

    def update(self, dt: float):
        current_time = time.time()
        to_remove = []

        for entity_id, entity in self.semantic_mapper.semantic_map.entities.items():
            age = current_time - entity.timestamp
            entity.confidence *= math.exp(-self.decay_rate * age)

            if entity.confidence < 0.1:
                to_remove.append(entity_id)

        for entity_id in to_remove:
            del self.semantic_mapper.semantic_map.entities[entity_id]

        self._merge_nearby_entities()

    def _merge_nearby_entities(self):
        entities = list(self.semantic_mapper.semantic_map.entities.values())

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                e1, e2 = entities[i], entities[j]
                if e1.class_id != e2.class_id:
                    continue

                dist = np.linalg.norm(e1.position - e2.position)
                if dist < self.merge_threshold:
                    new_position = (e1.position * e1.observations + e2.position * e2.observations) / (e1.observations + e2.observations)
                    new_confidence = max(e1.confidence, e2.confidence)
                    new_observations = e1.observations + e2.observations

                    e1.position = new_position
                    e1.confidence = new_confidence
                    e1.observations = new_observations

                    del self.semantic_mapper.semantic_map.entities[e2.entity_id]


class SemanticMapVisualizer:
    @staticmethod
    def draw_semantic_map(
        semantic_map: SemanticMap,
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if not semantic_map.entities:
            return vis

        positions = np.array([e.position[:2] for e in semantic_map.entities.values()])
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1

        scale = min(image_size[0], image_size[1]) * 0.8 / max(range_coords)

        colors = {
            SemanticClass.WALL: (0, 0, 255),
            SemanticClass.FLOOR: (0, 255, 0),
            SemanticClass.CEILING: (255, 0, 0),
            SemanticClass.DOOR: (255, 255, 0),
            SemanticClass.WINDOW: (0, 255, 255),
            SemanticClass.TABLE: (128, 0, 128),
            SemanticClass.CHAIR: (255, 128, 0),
            SemanticClass.PERSON: (0, 128, 255),
            SemanticClass.VEHICLE: (128, 128, 128),
        }

        for entity in semantic_map.entities.values():
            x = int((entity.position[0] - min_coords[0]) * 100 + image_size[0] * 0.1)
            y = int(image_size[1] - (entity.position[1] - min_coords[1]) * 100 - image_size[1] * 0.1)

            color = colors.get(entity.class_id, (255, 255, 255))
            size = max(3, int(entity.confidence * 10))
            cv2.circle(vis, (x, y), size, color, -1)

        return vis

    @staticmethod
    def draw_topological_map(
        topological_map: TopologicalMap,
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if not topological_map.places:
            return vis

        positions = np.array([p.center[:2] for p in topological_map.places.values()])
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1

        scale = min(image_size[0], image_size[1]) * 0.8 / max(range_coords)

        for (a, b), edge in topological_map.edges.items():
            if a in topological_map.places and b in topological_map.places:
                pos1 = topological_map.places[a].center
                pos2 = topological_map.places[b].center

                x1 = int((pos1[0] - min(positions[:, 0])) * 100 + image_size[0] * 0.1)
                y1 = int(image_size[1] - (pos1[1] - min(positions[:, 1])) * 100 - image_size[1] * 0.1)
                x2 = int((pos2[0] - min(positions[:, 0])) * 100 + image_size[0] * 0.1)
                y2 = int(image_size[1] - (pos2[1] - min(positions[:, 1])) * 100 - image_size[1] * 0.1)

                cv2.line(vis, (x1, y1), (x2, y2), (100, 100, 100), 2)

        for place_id, place in topological_map.places.items():
            x = int((place.center[0] - min(positions[:, 0])) * 100 + image_size[0] * 0.1)
            y = int(image_size[1] - (place.center[1] - min(positions[:, 1])) * 100 - image_size[1] * 0.1)

            cv2.circle(vis, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(vis, str(place_id), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return vis


class SemanticMapMetrics:
    @staticmethod
    def compute_entity_density(semantic_map: SemanticMap, volume: float) -> float:
        if volume <= 0:
            return 0.0
        return len(semantic_map.entities) / volume

    @staticmethod
    def compute_class_distribution(semantic_map: SemanticMap) -> Dict[SemanticClass, int]:
        dist = defaultdict(int)
        for entity in semantic_map.entities.values():
            dist[entity.class_id] += 1
        return dict(dist)

    @staticmethod
    def compute_coverage(semantic_map: SemanticMap, grid_resolution: float) -> float:
        if not semantic_map.entities:
            return 0.0

        occupied_cells = set()
        for entity in semantic_map.entities.values():
            cell = tuple((entity.position / grid_resolution).astype(int))
            occupied_cells.add(cell)

        return len(occupied_cells)

    @staticmethod
    def compute_topological_connectivity(topological_map: TopologicalMap) -> float:
        if not topological_map.places:
            return 0.0

        n = len(topological_map.places)
        if n < 2:
            return 1.0

        max_edges = n * (n - 1) / 2
        actual_edges = len(topological_map.edges) / 2
        return actual_edges / max_edges


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet