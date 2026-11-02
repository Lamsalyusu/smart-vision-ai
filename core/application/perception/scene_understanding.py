from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math


class SceneType(Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    URBAN = "urban"
    NATURAL = "natural"
    INDUSTRIAL = "industrial"
    MEDICAL = "medical"
    SPACE = "space"
    UNDERWATER = "underwater"


@dataclass(frozen=True, slots=True)
class SceneElement:
    element_id: str
    class_id: int
    class_name: str
    position: np.ndarray
    orientation: np.ndarray
    dimensions: np.ndarray
    confidence: float
    attributes: Dict[str, Any] = field(default_factory=dict)
    mesh: Optional[np.ndarray] = None
    texture: Optional[np.ndarray] = None


@dataclass(frozen=True, slots=True)
class SceneGraph:
    elements: List[SceneElement]
    relationships: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SceneUnderstanding:
    scene_type: SceneType
    scene_graph: SceneGraph
    global_features: np.ndarray
    layout: Dict[str, Any]
    affordances: List[str]
    navigation_graph: Dict[str, Any]


class SceneElementDetector:
    def __init__(
        self,
        detector_type: str = "yolo",
        confidence_threshold: float = 0.5,
    ):
        self.detector_type = detector_type
        self.confidence_threshold = confidence_threshold

    def detect(self, image: np.ndarray, depth: np.ndarray = None) -> List[SceneElement]:
        detections = self._detect_objects(image)
        elements = []

        for i, det in enumerate(detections):
            if det['confidence'] < self.confidence_threshold:
                continue

            position = self._estimate_position(det, depth)
            orientation = self._estimate_orientation(det, depth)
            dimensions = self._estimate_dimensions(det, depth)

            element = SceneElement(
                element_id=f"elem_{i}",
                class_id=det['class_id'],
                class_name=det['class_name'],
                position=position,
                orientation=orientation,
                dimensions=dimensions,
                confidence=det['confidence'],
            )
            elements.append(element)

        return elements

    def _detect_objects(self, image: np.ndarray) -> List[Dict]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detections.append({
                'bbox': [x, y, x + w, y + h],
                'confidence': 0.5,
                'class_id': 0,
                'class_name': 'object',
            })

        return detections

    def _estimate_position(self, det: Dict, depth: np.ndarray = None) -> np.ndarray:
        x1, y1, x2, y2 = det['bbox']
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        if depth is not None and 0 <= int(cy) < depth.shape[0] and 0 <= int(cx) < depth.shape[1]:
            z = depth[int(cy), int(cx)]
            if z > 0:
                fx, fy = 800.0, 800.0
                cx_cam, cy_cam = 320, 240
                x = (cx - cx_cam) * z / fx
                y = (cy - cy_cam) * z / fy
                return np.array([x, y, z])

        return np.array([cx, cy, 1.0])

    def _estimate_orientation(self, det: Dict, depth: np.ndarray = None) -> np.ndarray:
        return np.eye(3)

    def _estimate_dimensions(self, det: Dict, depth: np.ndarray = None) -> np.ndarray:
        x1, y1, x2, y2 = det['bbox']
        w, h = x2 - x1, y2 - y1

        if depth is not None:
            x1, y1, x2, y2 = map(int, det['bbox'])
            roi_depth = depth[y1:y2, x1:x2]
            valid = roi_depth[roi_depth > 0]
            if len(valid) > 0:
                depth_est = np.median(valid)
                return np.array([w * depth_est / 800, h * depth_est / 800, depth_est * 0.1])

        return np.array([w, h, 1.0])


class SceneRelationshipAnalyzer:
    def __init__(
        self,
        spatial_threshold: float = 1.0,
        support_threshold: float = 0.1,
    ):
        self.spatial_threshold = spatial_threshold
        self.support_threshold = support_threshold

    def analyze(self, elements: List[SceneElement]) -> List[Dict[str, Any]]:
        relationships = []

        for i, elem_a in enumerate(elements):
            for j, elem_b in enumerate(elements):
                if i >= j:
                    continue

                rel = self._analyze_pair(elem_a, elem_b)
                if rel:
                    relationships.append(rel)

        return relationships

    def _analyze_pair(self, a: SceneElement, b: SceneElement) -> Optional[Dict]:
        rels = []

        dist = np.linalg.norm(a.position - b.position)
        if dist < self.spatial_threshold:
            rels.append({
                'type': 'near',
                'elements': (a.element_id, b.element_id),
                'distance': float(dist),
                'confidence': max(0, 1 - dist / self.spatial_threshold),
            })

        dx = b.position[0] - a.position[0]
        dy = b.position[1] - a.position[1]
        dz = b.position[2] - a.position[2]

        if abs(dx) > abs(dy) and abs(dx) > abs(dz):
            rel_type = 'right_of' if dx > 0 else 'left_of'
        elif abs(dy) > abs(dx) and abs(dy) > abs(dz):
            rel_type = 'above' if dy > 0 else 'below'
        elif abs(dz) > abs(dx) and abs(dz) > abs(dy):
            rel_type = 'in_front_of' if dz > 0 else 'behind'
        else:
            rel_type = 'near'

        rels.append({
            'type': rel_type,
            'elements': (a.element_id, b.element_id),
            'confidence': 0.7,
        })

        if self._is_supporting(a, b):
            rels.append({
                'type': 'supporting',
                'elements': (a.element_id, b.element_id),
                'confidence': 0.6,
            })

        if self._is_inside(a, b):
            rels.append({
                'type': 'inside',
                'elements': (a.element_id, b.element_id),
                'confidence': 0.8,
            })
        elif self._is_inside(b, a):
            rels.append({
                'type': 'contains',
                'elements': (a.element_id, b.element_id),
                'confidence': 0.8,
            })

        return {'relationships': rels}

    def _is_supporting(self, a: SceneElement, b: SceneElement) -> bool:
        if a.dimensions is None or b.dimensions is None:
            return False

        a_bottom = a.position[1] - a.dimensions[1] / 2
        b_bottom = b.position[1] - b.dimensions[1] / 2

        if abs(a_bottom - b_bottom) > 0.1:
            return False

        dx = abs(a.position[0] - b.position[0])
        dz = abs(a.position[2] - b.position[2])

        if dx < a.dimensions[0] / 2 + b.dimensions[0] / 2:
            if dz < a.dimensions[2] / 2 + b.dimensions[2] / 2:
                return True
        return False

    def _is_inside(self, a: SceneElement, b: SceneElement) -> bool:
        if b.dimensions is None:
            return False

        half_dims = b.dimensions / 2
        rel_pos = a.position - b.position

        return (abs(rel_pos[0]) < half_dims[0] and
                abs(rel_pos[1]) < half_dims[1] and
                abs(rel_pos[2]) < half_dims[2])


class SceneLayoutAnalyzer:
    def __init__(
        self,
        floor_threshold: float = 0.1,
        wall_threshold: float = 0.8,
    ):
        self.floor_threshold = floor_threshold
        self.wall_threshold = wall_threshold

    def analyze(self, elements: List[SceneElement], depth: np.ndarray = None) -> Dict[str, Any]:
        layout = {
            'floor_plane': None,
            'walls': [],
            'ceiling': None,
            'objects': [],
            'rooms': [],
        }

        floor_candidates = [e for e in elements if e.class_name in ['floor', 'ground']]
        if floor_candidates:
            layout['floor_plane'] = self._fit_plane(floor_candidates[0])

        wall_candidates = [e for e in elements if e.class_name in ['wall']]
        for wall in wall_candidates:
            layout['walls'].append(self._fit_plane(wall))

        object_candidates = [e for e in elements if e.class_name not in ['floor', 'wall', 'ceiling']]
        layout['objects'] = [self._object_to_dict(obj) for obj in object_candidates]

        if depth is not None:
            layout['depth_map'] = depth.tolist()

        return layout

    def _fit_plane(self, element: SceneElement) -> Dict:
        return {
            'normal': element.orientation[:, 2] if element.orientation is not None else [0, 1, 0],
            'point': element.position,
            'area': np.prod(element.dimensions[:2]) if element.dimensions is not None else 1.0,
        }

    def _object_to_dict(self, obj: SceneElement) -> Dict:
        return {
            'id': obj.element_id,
            'class': obj.class_name,
            'position': obj.position.tolist(),
            'dimensions': obj.dimensions.tolist() if obj.dimensions is not None else [1, 1, 1],
        }


class SceneGraphBuilder:
    def __init__(
        self,
        detector: SceneElementDetector,
        relationship_analyzer: SceneRelationshipAnalyzer,
        layout_analyzer: SceneLayoutAnalyzer,
    ):
        self.detector = detector
        self.relationship_analyzer = relationship_analyzer
        self.layout_analyzer = layout_analyzer

    def build(self, image: np.ndarray, depth: np.ndarray = None) -> SceneUnderstanding:
        elements = self.detector.detect(image, depth)
        relationships = self.relationship_analyzer.analyze(elements)
        layout = self.layout_analyzer.analyze(elements, depth)

        scene_graph = SceneGraph(
            elements=elements,
            relationships=relationships,
        )

        scene_type = self._classify_scene(elements)
        global_features = self._extract_global_features(image)
        affordances = self._extract_affordances(elements)
        navigation_graph = self._build_navigation_graph(elements)

        return SceneUnderstanding(
            scene_type=scene_type,
            scene_graph=scene_graph,
            global_features=global_features,
            layout=layout,
            affordances=affordances,
            navigation_graph=navigation_graph,
        )

    def _classify_scene(self, elements: List[SceneElement]) -> SceneType:
        classes = [e.class_name for e in elements]

        if any(c in classes for c in ['bed', 'sofa', 'table', 'chair']):
            return SceneType.INDOOR
        elif any(c in classes for c in ['car', 'road', 'building', 'tree']):
            return SceneType.OUTDOOR
        elif any(c in classes for c in ['desk', 'monitor', 'keyboard', 'chair']):
            return SceneType.INDOOR
        elif any(c in classes for c in ['machine', 'conveyor', 'robot']):
            return SceneType.INDUSTRIAL

        return SceneType.INDOOR

    def _extract_global_features(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        gray = cv2.resize(gray, (64, 64))
        return gray.flatten()

    def _extract_affordances(self, elements: List[SceneElement]) -> List[str]:
        affordances = []
        for elem in elements:
            if elem.class_name in ['chair', 'sofa', 'bench']:
                affordances.append('sittable')
            elif elem.class_name in ['table', 'desk', 'counter']:
                affordances.append('placeable')
            elif elem.class_name in ['door', 'window']:
                affordances.append('passable')
            elif elem.class_name in ['floor', 'ground']:
                affordances.append('walkable')
        return list(set(affordances))

    def _build_navigation_graph(self, elements: List[SceneElement]) -> Dict[str, Any]:
        walkable = [e for e in elements if e.class_name in ['floor', 'ground']]

        graph = {
            'nodes': [],
            'edges': [],
        }

        for i, e in enumerate(walkable):
            graph['nodes'].append({
                'id': e.element_id,
                'position': e.position.tolist(),
                'area': np.prod(e.dimensions[:2]) if e.dimensions is not None else 1.0,
            })

        for i, a in enumerate(walkable):
            for b in walkable[i+1:]:
                dist = np.linalg.norm(a.position - b.position)
                if dist < 5.0:
                    graph['edges'].append({
                        'from': a.element_id,
                        'to': b.element_id,
                        'distance': float(dist),
                    })

        return graph


class SceneClassifier:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            self._initialized = True
        except ImportError:
            self._initialized = False

    def classify(self, image: np.ndarray) -> Tuple[SceneType, float]:
        if not self._initialized:
            self._initialize()

        if self._initialized:
            return self._classify_deep(image)
        else:
            return self._classify_classical(image)

    def _classify_deep(self, image: np.ndarray) -> Tuple[SceneType, float]:
        return SceneType.INDOOR, 0.9

    def _classify_classical(self, image: np.ndarray) -> Tuple[SceneType, float]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)

        horizontal = vertical = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(math.atan2(y2 - y1, x2 - x1))
                if angle < 0.2 or angle > 3.0:
                    horizontal += 1
                elif abs(angle - 1.57) < 0.2:
                    vertical += 1

        if horizontal > vertical * 2:
            return SceneType.INDOOR, 0.7
        elif vertical > horizontal * 2:
            return SceneType.URBAN, 0.7
        return SceneType.NATURAL, 0.6


class SemanticMapper:
    def __init__(
        self,
        resolution: float = 0.1,
        map_size: int = 1000,
    ):
        self.resolution = resolution
        self.map_size = map_size
        self.semantic_map = np.zeros((map_size, map_size, 10), dtype=np.float32)
        self.class_names = ['floor', 'wall', 'ceiling', 'door', 'window', 'chair', 'table', 'sofa', 'bed', 'other']

    def update(self, elements: List[SceneElement], robot_pose: np.ndarray):
        for elem in elements:
            local_pos = elem.position - robot_pose[:3, 3]
            grid_x = int(local_pos[0] / self.resolution + self.map_size / 2)
            grid_y = int(local_pos[2] / self.resolution + self.map_size / 2)

            if 0 <= grid_x < self.map_size and 0 <= grid_y < self.map_size:
                class_idx = min(elem.class_id, 9)
                self.semantic_map[grid_y, grid_x, class_idx] += elem.confidence

    def get_semantic_map(self) -> np.ndarray:
        return self.semantic_map

    def get_object_map(self, class_id: int) -> np.ndarray:
        if class_id >= self.semantic_map.shape[2]:
            return np.zeros((self.map_size, self.map_size))
        return self.semantic_map[:, :, class_id]


class SceneUnderstandingEngine:
    def __init__(
        self,
        detector: SceneElementDetector,
        relationship_analyzer: SceneRelationshipAnalyzer,
        layout_analyzer: SceneLayoutAnalyzer,
        classifier: SceneClassifier,
        semantic_mapper: SemanticMapper,
    ):
        self.detector = detector
        self.relationship_analyzer = relationship_analyzer
        self.layout_analyzer = layout_analyzer
        self.classifier = classifier
        self.semantic_mapper = semantic_mapper
        self.graph_builder = SceneGraphBuilder(detector, relationship_analyzer, layout_analyzer)

    def understand(self, image: np.ndarray, depth: np.ndarray = None, robot_pose: np.ndarray = None) -> SceneUnderstanding:
        scene_type, _ = self.classifier.classify(image)
        understanding = self.graph_builder.build(image, depth)

        if robot_pose is not None:
            self.semantic_mapper.update(understanding.scene_graph.elements, robot_pose)

        return understanding


class SceneQueryEngine:
    def __init__(self, understanding: SceneUnderstanding):
        self.understanding = understanding

    def find_objects_by_class(self, class_name: str) -> List[SceneElement]:
        return [e for e in self.understanding.scene_graph.elements if e.class_name == class_name]

    def find_objects_near(self, position: np.ndarray, radius: float) -> List[SceneElement]:
        results = []
        for elem in self.understanding.scene_graph.elements:
            dist = np.linalg.norm(elem.position - position)
            if dist <= radius:
                results.append(elem)
        return results

    def find_path(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        graph = self.understanding.navigation_graph
        return [start, goal]

    def get_affordances_at(self, position: np.ndarray) -> List[str]:
        for elem in self.understanding.scene_graph.elements:
            dist = np.linalg.norm(elem.position - position)
            if dist < 1.0:
                return self.understanding.affordances
        return []


class SceneMemory:
    def __init__(self, max_scenes: int = 100):
        self.max_scenes = max_scenes
        self.scenes: List[SceneUnderstanding] = []
        self.embeddings: List[np.ndarray] = []

    def add_scene(self, understanding: SceneUnderstanding):
        if len(self.scenes) >= self.max_scenes:
            self.scenes.pop(0)
            self.embeddings.pop(0)

        self.scenes.append(understanding)
        self.embeddings.append(understanding.global_features)

    def find_similar(self, query_features: np.ndarray, top_k: int = 5) -> List[SceneUnderstanding]:
        if not self.embeddings:
            return []

        similarities = [np.dot(query_features, emb) / (np.linalg.norm(query_features) * np.linalg.norm(emb) + 1e-8)
                       for emb in self.embeddings]

        top_indices = np.argsort(similarities)[-min(top_k, len(similarities)):]
        return [self.scenes[i] for i in reversed(top_indices)]

    def get_scene_at_time(self, timestamp: float) -> Optional[SceneUnderstanding]:
        for scene in reversed(self.scenes):
            if abs(scene.scene_graph.metadata.get('timestamp', 0) - timestamp) < 1.0:
                return scene
        return None