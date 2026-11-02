from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict
import heapq


class PlaceType(Enum):
    ROOM = "room"
    CORRIDOR = "corridor"
    JUNCTION = "junction"
    DOORWAY = "doorway"
    LANDMARK = "landmark"
    OUTDOOR = "outdoor"
    UNKNOWN = "unknown"


class EdgeType(Enum):
    TOPOLOGICAL = "topological"
    METRIC = "metric"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"


@dataclass(frozen=True, slots=True)
class Place:
    id: int
    place_type: PlaceType
    position: np.ndarray
    descriptor: np.ndarray
    timestamp: float
    features: Dict[str, Any] = field(default_factory=dict)
    observations: List[int] = field(default_factory=list)
    confidence: float = 1.0
    semantic_label: Optional[str] = None


@dataclass(frozen=True, slots=True)
class TopologicalEdge:
    from_id: int
    to_id: int
    edge_type: EdgeType
    weight: float
    distance: float
    transform: Optional[np.ndarray] = None
    observations: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class TopologicalMap:
    places: Dict[int, Place]
    edges: Dict[Tuple[int, int], TopologicalEdge]
    place_id_counter: int = 0
    edge_id_counter: int = 0


class PlaceRecognizer:
    def __init__(
        self,
        descriptor_dim: int = 256,
        similarity_threshold: float = 0.8,
        max_places: int = 10000
    ):
        self.descriptor_dim = descriptor_dim
        self.similarity_threshold = similarity_threshold
        self.max_places = max_places
        self.places: Dict[int, Place] = {}
        self.place_id_counter = 0
        self.kdtree = None

    def add_place(self, position: np.ndarray, descriptor: np.ndarray, place_type: PlaceType = PlaceType.UNKNOWN, **kwargs) -> int:
        if len(self.places) >= self.max_places:
            self._remove_oldest_place()

        place_id = self.place_id_counter
        self.place_id_counter += 1

        place = Place(
            id=place_id,
            place_type=place_type,
            position=position,
            descriptor=descriptor,
            timestamp=time.time(),
            **kwargs
        )

        self.places[place_id] = place
        self._rebuild_kdtree()

        return place_id

    def recognize(self, descriptor: np.ndarray, position: np.ndarray) -> Optional[int]:
        if not self.places:
            return None

        best_match = None
        best_score = -1.0

        for place_id, place in self.places.items():
            score = self._compute_similarity(descriptor, place.descriptor)
            if score > self.similarity_threshold and score > best_score:
                best_score = score
                best_match = place_id

        return best_match

    def recognize_knn(self, descriptor: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        if not self.places:
            return []

        scores = []
        for place_id, place in self.places.items():
            score = self._compute_similarity(descriptor, place.descriptor)
            scores.append((place_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def _compute_similarity(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        norm1 = np.linalg.norm(desc1)
        norm2 = np.linalg.norm(desc2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(desc1, desc2) / (norm1 * norm2))

    def _rebuild_kdtree(self):
        try:
            from scipy.spatial import KDTree
        except ImportError:
            self.kdtree = None
            return

        if self.places:
            descriptors = np.array([p.descriptor for p in self.places.values()])
            self.kdtree = KDTree(descriptors)

    def _remove_oldest_place(self):
        if not self.places:
            return
        oldest = min(self.places.items(), key=lambda x: x[1].timestamp)
        del self.places[oldest[0]]


class TopologicalMapper:
    def __init__(
        self,
        place_recognizer: PlaceRecognizer,
        max_edge_distance: float = 10.0,
        min_observations: int = 3
    ):
        self.place_recognizer = place_recognizer
        self.max_edge_distance = max_edge_distance
        self.min_observations = min_observations
        self.map = TopologicalMap(places={}, edges={})
        self.current_place: Optional[int] = None
        self.trajectory: List[int] = []

    def add_observation(
        self,
        descriptor: np.ndarray,
        position: np.ndarray,
        place_type: PlaceType = PlaceType.UNKNOWN,
        transform: Optional[np.ndarray] = None,
        **kwargs
    ) -> int:
        place_id = self.place_recognizer.recognize(descriptor, position)

        if place_id is None:
            place_id = self.place_recognizer.add_place(position, descriptor, place_type, **kwargs)
            self.map.places[place_id] = self.place_recognizer.places[place_id]
        else:
            self._update_place(place_id, position, descriptor, transform, **kwargs)

        if self.current_place is not None and self.current_place != place_id:
            self._add_edge(self.current_place, place_id, transform)

        self.current_place = place_id
        self.trajectory.append(place_id)

        return place_id

    def _update_place(
        self,
        place_id: int,
        position: np.ndarray,
        descriptor: np.ndarray,
        transform: Optional[np.ndarray] = None,
        **kwargs
    ):
        place = self.map.places[place_id]
        place.observations.append(len(self.trajectory))

        alpha = 0.1
        place.position = (1 - alpha) * place.position + alpha * position
        place.descriptor = (1 - alpha) * place.descriptor + alpha * descriptor
        place.descriptor /= (np.linalg.norm(place.descriptor) + 1e-6)

        for k, v in kwargs.items():
            place.features[k] = v

    def _add_edge(
        self,
        from_id: int,
        to_id: int,
        transform: Optional[np.ndarray] = None
    ):
        edge_key = tuple(sorted((from_id, to_id)))

        if edge_key in self.map.edges:
            edge = self.map.edges[edge_key]
            edge.observations += 1
            edge.weight = min(edge.weight + 0.1, 1.0)
        else:
            from_pos = self.map.places[from_id].position
            to_pos = self.map.places[to_id].position
            distance = np.linalg.norm(to_pos - from_pos)

            if distance > self.max_edge_distance:
                return

            edge = TopologicalEdge(
                from_id=from_id,
                to_id=to_id,
                edge_type=EdgeType.TOPOLOGICAL,
                weight=0.5,
                distance=distance,
                transform=transform,
                observations=1
            )
            self.map.edges[edge_key] = edge

    def get_neighbors(self, place_id: int) -> List[int]:
        neighbors = []
        for (a, b), edge in self.map.edges.items():
            if a == place_id:
                neighbors.append(b)
            elif b == place_id:
                neighbors.append(a)
        return neighbors

    def find_path(self, start_id: int, goal_id: int) -> List[int]:
        if start_id not in self.map.places or goal_id not in self.map.places:
            return []

        open_set = [(0, start_id)]
        came_from = {}
        g_score = {start_id: 0}
        f_score = {start_id: self._heuristic(start_id, goal_id)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_id:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_id)
                return path[::-1]

            for neighbor in self.get_neighbors(current):
                edge_key = tuple(sorted((current, neighbor)))
                edge = self.map.edges.get(edge_key)
                if edge is None:
                    continue

                tentative_g = g_score[current] + edge.distance / max(edge.weight, 0.1)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_id)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def _heuristic(self, a: int, b: int) -> float:
        pos_a = self.map.places[a].position
        pos_b = self.map.places[b].position
        return np.linalg.norm(pos_a - pos_b)

    def get_loop_candidates(self, current_id: int, min_distance: float = 5.0, max_distance: float = 20.0) -> List[int]:
        candidates = []
        current_pos = self.map.places[current_id].position

        for place_id, place in self.map.places.items():
            if place_id == current_id:
                continue
            dist = np.linalg.norm(place.position - current_pos)
            if min_distance <= dist <= max_distance:
                candidates.append(place_id)

        return candidates


class SemanticMapper:
    def __init__(
        self,
        place_recognizer: PlaceRecognizer,
        object_detector: Optional[Callable] = None
    ):
        self.place_recognizer = place_recognizer
        self.object_detector = object_detector
        self.semantic_map: Dict[str, List[int]] = defaultdict(list)

    def add_semantic_observation(
        self,
        place_id: int,
        objects: List[Dict[str, Any]]
    ):
        for obj in objects:
            label = obj.get('label', 'unknown')
            self.semantic_map[label].append(place_id)

            place = self.place_recognizer.places.get(place_id)
            if place:
                place.semantic_label = label

    def query_places_by_label(self, label: str) -> List[int]:
        return self.semantic_map.get(label, [])

    def get_place_semantics(self, place_id: int) -> Dict[str, float]:
        place = self.place_recognizer.places.get(place_id)
        if not place:
            return {}

        semantics = {}
        for label, places in self.semantic_map.items():
            if place_id in places:
                semantics[label] = places.count(place_id) / len(places)

        return semantics


class MetricTopologicalFusion:
    def __init__(
        self,
        topological_mapper: TopologicalMapper,
        metric_map: Optional[Any] = None
    ):
        self.topological_mapper = topological_mapper
        self.metric_map = metric_map
        self.alignment_transforms: Dict[int, np.ndarray] = {}

    def add_metric_constraint(
        self,
        place_id: int,
        metric_pose: np.ndarray
    ):
        self.alignment_transforms[place_id] = metric_pose

    def optimize_alignment(self) -> Dict[int, np.ndarray]:
        if len(self.alignment_transforms) < 2:
            return self.alignment_transforms

        place_ids = list(self.alignment_transforms.keys())
        poses = [self.alignment_transforms[pid] for pid in place_ids]

        try:
            import g2o
            return self._optimize_g2o(place_ids, poses)
        except ImportError:
            return self._optimize_scipy(place_ids, poses)

    def _optimize_g2o(self, place_ids: List[int], poses: List[np.ndarray]) -> Dict[int, np.ndarray]:
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(solver)

        vertex_map = {}
        for i, place_id in enumerate(place_ids):
            v = g2o.VertexSE3()
            v.set_id(place_id)
            v.set_estimate(g2o.Isometry3d(poses[i]))
            if i == 0:
                v.set_fixed(True)
            optimizer.add_vertex(v)
            vertex_map[place_id] = v

        for (a, b), edge in self.topological_mapper.map.edges.items():
            if a in vertex_map and b in vertex_map and edge.transform is not None:
                e = g2o.EdgeSE3()
                e.set_vertex(0, vertex_map[a])
                e.set_vertex(1, vertex_map[b])
                e.set_measurement(g2o.Isometry3d(edge.transform))
                e.set_information(np.eye(6) * edge.weight)
                optimizer.add_edge(e)

        for place_id in place_ids:
            if place_id in vertex_map:
                v = vertex_map[place_id]
                v.set_fixed(True)
                break

        optimizer.initialize_optimization()
        optimizer.optimize(20)

        result = {}
        for place_id, v in vertex_map.items():
            result[place_id] = np.array(v.estimate().matrix())

        return result

    def _optimize_scipy(self, place_ids: List[int], poses: List[np.ndarray]) -> Dict[int, np.ndarray]:
        from scipy.optimize import least_squares

        n = len(place_ids)
        x0 = []
        for pose in poses:
            rvec = cv2.Rodrigues(pose[:3, :3])[0].flatten()
            tvec = pose[:3, 3]
            x0.extend(rvec)
            x0.extend(tvec)

        x0 = np.array(x0)

        def residuals(x):
            opt_poses = {}
            idx = 0
            for place_id in place_ids:
                rvec = x[idx:idx+3]
                tvec = x[idx+3:idx+6]
                idx += 6
                R = cv2.Rodrigues(rvec)[0]
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = tvec
                opt_poses[place_id] = pose

            res = []
            for (a, b), edge in self.topological_mapper.map.edges.items():
                if a in opt_poses and b in opt_poses and edge.transform is not None:
                    pred = np.linalg.inv(opt_poses[a]) @ opt_poses[b]
                    error = self._se3_log(pred @ np.linalg.inv(edge.transform))
                    res.extend(error * edge.weight)

            return np.array(res)

        result = least_squares(residuals, x0, max_nfev=100)
        x_opt = result.x

        opt_poses = {}
        idx = 0
        for place_id in place_ids:
            rvec = x_opt[idx:idx+3]
            tvec = x_opt[idx+3:idx+6]
            idx += 6
            R = cv2.Rodrigues(rvec)[0]
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = tvec
            opt_poses[place_id] = pose

        return opt_poses

    def _se3_log(self, T: np.ndarray) -> np.ndarray:
        R = T[:3, :3]
        t = T[:3, 3]

        theta = math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        if theta < 1e-6:
            omega = np.zeros(3)
        else:
            omega = theta / (2 * math.sin(theta)) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

        V = np.eye(3) - (1 - math.cos(theta)) / (theta**2) * cv2.Rodrigues(omega)[0] + \
            (theta - math.sin(theta)) / (theta**3) * cv2.Rodrigues(omega)[0] @ cv2.Rodrigues(omega)[0]
        v = V @ t

        return np.concatenate([v, omega])


class TopologicalVisualizer:
    @staticmethod
    def draw_topological_map(
        topological_map: TopologicalMap,
        image_size: Tuple[int, int] = (800, 600),
        current_place: Optional[int] = None
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if not topological_map.places:
            return vis

        positions = np.array([p.position for p in topological_map.places.values()])
        min_coords = np.min(positions[:, :2], axis=0)
        max_coords = np.max(positions[:, :2], axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1

        scale = min(image_size[0], image_size[1]) * 0.8 / max(range_coords)

        place_positions = {}
        for place_id, place in topological_map.places.items():
            x = int((place.position[0] - min_coords[0]) * scale + image_size[0] * 0.1)
            y = int(image_size[1] - (place.position[1] - min_coords[1]) * scale - image_size[1] * 0.1)
            place_positions[place_id] = (x, y)

        for (a, b), edge in topological_map.edges.items():
            if a in place_positions and b in place_positions:
                color_intensity = int(edge.weight * 255)
                cv2.line(vis, place_positions[a], place_positions[b], (color_intensity, color_intensity, 255), 2)

        for place_id, place in topological_map.places.items():
            if place_id in place_positions:
                x, y = place_positions[place_id]
                color = {
                    PlaceType.ROOM: (0, 255, 0),
                    PlaceType.CORRIDOR: (255, 255, 0),
                    PlaceType.JUNCTION: (255, 0, 255),
                    PlaceType.DOORWAY: (0, 255, 255),
                    PlaceType.LANDMARK: (255, 0, 0),
                    PlaceType.OUTDOOR: (0, 128, 255),
                    PlaceType.UNKNOWN: (128, 128, 128)
                }.get(place.place_type, (255, 255, 255))

                if place_id == current_place:
                    cv2.circle(vis, (x, y), 10, (0, 0, 255), 3)
                else:
                    cv2.circle(vis, (x, y), 6, color, -1)

                cv2.putText(vis, str(place_id), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return vis

    @staticmethod
    def draw_path(
        image: np.ndarray,
        topological_map: TopologicalMap,
        path: List[int],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 3
    ) -> np.ndarray:
        vis = image.copy()

        if not topological_map.places:
            return vis

        positions = np.array([p.position for p in topological_map.places.values()])
        min_coords = np.min(positions[:, :2], axis=0)
        max_coords = np.max(positions[:, :2], axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1

        scale = min(image.shape[1], image.shape[0]) * 0.8 / max(range_coords)

        place_positions = {}
        for place_id, place in topological_map.places.items():
            x = int((place.position[0] - min_coords[0]) * scale + image.shape[1] * 0.1)
            y = int(image.shape[0] - (place.position[1] - min_coords[1]) * scale - image.shape[0] * 0.1)
            place_positions[place_id] = (x, y)

        for i in range(len(path) - 1):
            if path[i] in place_positions and path[i+1] in place_positions:
                cv2.line(vis, place_positions[path[i]], place_positions[path[i+1]], color, thickness)

        return vis


class TopologicalMetrics:
    @staticmethod
    def compute_map_coverage(topological_map: TopologicalMap) -> float:
        if len(topological_map.places) < 2:
            return 0.0

        positions = np.array([p.position for p in topological_map.places.values()])
        area = np.max(positions[:, 0]) - np.min(positions[:, 0])
        area *= np.max(positions[:, 1]) - np.min(positions[:, 1])
        return float(area)

    @staticmethod
    def compute_connectivity(topological_map: TopologicalMap) -> float:
        if not topological_map.places:
            return 0.0

        n = len(topological_map.places)
        max_edges = n * (n - 1) / 2
        actual_edges = len(topological_map.edges)
        return actual_edges / max_edges if max_edges > 0 else 0.0

    @staticmethod
    def compute_average_edge_weight(topological_map: TopologicalMap) -> float:
        if not topological_map.edges:
            return 0.0
        return float(np.mean([e.weight for e in topological_map.edges.values()]))

    @staticmethod
    def compute_place_type_distribution(topological_map: TopologicalMap) -> Dict[PlaceType, int]:
        dist = defaultdict(int)
        for place in topological_map.places.values():
            dist[place.place_type] += 1
        return dict(dist)


class TopologicalMapperPipeline:
    def __init__(self):
        self.recognizer = PlaceRecognizer()
        self.mapper = TopologicalMapper(self.recognizer)
        self.semantic_mapper = SemanticMapper(self.recognizer)

    def process_observation(
        self,
        descriptor: np.ndarray,
        position: np.ndarray,
        place_type: PlaceType = PlaceType.UNKNOWN,
        transform: Optional[np.ndarray] = None,
        objects: List[Dict] = None
    ) -> int:
        place_id = self.mapper.add_observation(descriptor, position, place_type, transform)

        if objects:
            self.semantic_mapper.add_semantic_observation(place_id, objects)

        return place_id

    def get_path(self, start_id: int, goal_id: int) -> List[int]:
        return self.mapper.find_path(start_id, goal_id)

    def get_loop_candidates(self, current_id: int) -> List[int]:
        return self.mapper.get_loop_candidates(current_id)


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet