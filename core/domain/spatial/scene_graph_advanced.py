from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Any
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict
import math
import heapq


class SpatialRelationType(Enum):
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    NEAR = "near"
    FAR = "far"
    INSIDE = "inside"
    CONTAINS = "contains"
    TOUCHING = "touching"
    OVERLAPPING = "overlapping"
    SUPPORTING = "supporting"
    SUPPORTED_BY = "supported_by"
    ATTACHED_TO = "attached_to"
    BETWEEN = "between"


@dataclass(frozen=True, slots=True)
class SpatialEntity:
    entity_id: str
    bbox: np.ndarray
    class_id: int
    class_name: str
    confidence: float
    center_3d: Optional[np.ndarray] = None
    dimensions_3d: Optional[np.ndarray] = None
    orientation: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None
    attributes: dict = None

    def __post_init__(self):
        if self.attributes is None:
            object.__setattr__(self, 'attributes', {})

    @property
    def center_2d(self) -> np.ndarray:
        return np.array([(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2])

    @property
    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


@dataclass(frozen=True, slots=True)
class SpatialRelation:
    subject_id: str
    object_id: str
    relation: SpatialRelationType
    confidence: float
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})


class SceneGraph:
    def __init__(self):
        self.entities: dict[str, SpatialEntity] = {}
        self.relations: list[SpatialRelation] = []
        self._adjacency: dict[str, dict[str, list[SpatialRelation]]] = defaultdict(lambda: defaultdict(list))

    def add_entity(self, entity: SpatialEntity):
        self.entities[entity.entity_id] = entity

    def add_relation(self, relation: SpatialRelation):
        self.relations.append(relation)
        self._adjacency[relation.subject_id][relation.object_id].append(relation)

    def get_relations(self, subject_id: str, object_id: str = None) -> list[SpatialRelation]:
        if object_id is None:
            relations = []
            for obj_rels in self._adjacency[subject_id].values():
                relations.extend(obj_rels)
            return relations
        return self._adjacency[subject_id][object_id]

    def get_entities_by_class(self, class_id: int) -> list[SpatialEntity]:
        return [e for e in self.entities.values() if e.class_id == class_id]

    def get_neighbors(self, entity_id: str, relation_type: SpatialRelationType = None) -> list[SpatialEntity]:
        neighbors = []
        for obj_id, rels in self._adjacency[entity_id].items():
            for rel in rels:
                if relation_type is None or rel.relation == relation_type:
                    if obj_id in self.entities:
                        neighbors.append(self.entities[obj_id])
        return neighbors

    def query(self, subject_class: int = None, relation: SpatialRelationType = None, object_class: int = None) -> list[tuple[SpatialEntity, SpatialRelation, SpatialEntity]]:
        results = []
        for rel in self.relations:
            subj = self.entities.get(rel.subject_id)
            obj = self.entities.get(rel.object_id)
            if subj is None or obj is None:
                continue
            if subject_class is not None and subj.class_id != subject_class:
                continue
            if relation is not None and rel.relation != relation:
                continue
            if object_class is not None and obj.class_id != object_class:
                continue
            results.append((subj, rel, obj))
        return results

    def to_networkx(self):
        try:
            import networkx as nx
            G = nx.DiGraph()
            for eid, entity in self.entities.items():
                G.add_node(eid, **entity.attributes, class_id=entity.class_id, class_name=entity.class_name)
            for rel in self.relations:
                G.add_edge(rel.subject_id, rel.object_id, relation=rel.relation.value, confidence=rel.confidence)
            return G
        except ImportError:
            return None


class SpatialReasoner:
    def __init__(
        self,
        intrinsics: Optional[np.ndarray] = None,
        depth_threshold: float = 0.5,
        overlap_threshold: float = 0.3,
        support_angle_threshold: float = 30.0,
    ):
        self.intrinsics = intrinsics
        self.depth_threshold = depth_threshold
        self.overlap_threshold = overlap_threshold
        self.support_angle_threshold = support_angle_threshold

    def compute_relations(self, entities: list[SpatialEntity]) -> list[SpatialRelation]:
        relations = []
        n = len(entities)

        for i in range(n):
            for j in range(i + 1, n):
                subj, obj = entities[i], entities[j]
                rels = self._compute_pairwise_relations(subj, obj)
                relations.extend(rels)
                rels_rev = self._compute_pairwise_relations(obj, subj)
                relations.extend(rels_rev)

        return relations

    def _compute_pairwise_relations(self, subj: SpatialEntity, obj: SpatialEntity) -> list[SpatialRelation]:
        relations = []

        iou = self._compute_iou(subj.bbox, obj.bbox)
        if iou > self.overlap_threshold:
            relations.append(SpatialRelation(
                subject_id=subj.entity_id,
                object_id=obj.entity_id,
                relation=SpatialRelationType.OVERLAPPING,
                confidence=min(iou, 1.0),
                metadata={'iou': float(iou)},
            ))

        center_subj = subj.center_2d
        center_obj = obj.center_2d
        dx = center_obj[0] - center_subj[0]
        dy = center_obj[1] - center_subj[1]
        dist = np.sqrt(dx**2 + dy**2)

        if dist < 50:
            relations.append(SpatialRelation(
                subject_id=subj.entity_id,
                object_id=obj.entity_id,
                relation=SpatialRelationType.NEAR,
                confidence=max(0, 1 - dist / 100),
                metadata={'distance': float(dist)},
            ))
        else:
            relations.append(SpatialRelation(
                subject_id=subj.entity_id,
                object_id=obj.entity_id,
                relation=SpatialRelationType.FAR,
                confidence=min(1.0, dist / 500),
                metadata={'distance': float(dist)},
            ))

        if dx > 0 and abs(dx) > abs(dy):
            relations.append(SpatialRelation(
                subject_id=subj.entity_id,
                object_id=obj.entity_id,
                relation=SpatialRelationType.RIGHT_OF,
                confidence=min(1.0, abs(dx) / (abs(dx) + abs(dy) + 1)),
                metadata={'dx': float(dx), 'dy': float(dy)},
            ))
        elif dx < 0 and abs(dx) > abs(dy):
            relations.append(SpatialRelation(
                subject_id=subj.entity_id,
                object_id=obj.entity_id,
                relation=SpatialRelationType.LEFT_OF,
                confidence=min(1.0, abs(dx) / (abs(dx) + abs(dy) + 1)),
                metadata={'dx': float(dx), 'dy': float(dy)},
            ))

        if dy > 0 and abs(dy) > abs(dx):
            relations.append(SpatialRelation(
                subject_id=subj.entity_id,
                object_id=obj.entity_id,
                relation=SpatialRelationType.BELOW,
                confidence=min(1.0, abs(dy) / (abs(dx) + abs(dy) + 1)),
                metadata={'dx': float(dx), 'dy': float(dy)},
            ))
        elif dy < 0 and abs(dy) > abs(dx):
            relations.append(SpatialRelation(
                subject_id=subj.entity_id,
                object_id=obj.entity_id,
                relation=SpatialRelationType.ABOVE,
                confidence=min(1.0, abs(dy) / (abs(dx) + abs(dy) + 1)),
                metadata={'dx': float(dx), 'dy': float(dy)},
            ))

        if subj.center_3d is not None and obj.center_3d is not None:
            dz = obj.center_3d[2] - subj.center_3d[2]
            if abs(dz) > self.depth_threshold:
                if dz > 0:
                    relations.append(SpatialRelation(
                        subject_id=subj.entity_id,
                        object_id=obj.entity_id,
                        relation=SpatialRelationType.BEHIND,
                        confidence=min(1.0, abs(dz) / 5.0),
                        metadata={'dz': float(dz)},
                    ))
                else:
                    relations.append(SpatialRelation(
                        subject_id=subj.entity_id,
                        object_id=obj.entity_id,
                        relation=SpatialRelationType.IN_FRONT_OF,
                        confidence=min(1.0, abs(dz) / 5.0),
                        metadata={'dz': float(dz)},
                    ))

        if self._is_inside(subj.bbox, obj.bbox):
            relations.append(SpatialRelation(
                subject_id=subj.entity_id,
                object_id=obj.entity_id,
                relation=SpatialRelationType.INSIDE,
                confidence=0.9,
                metadata={},
            ))
        elif self._is_inside(obj.bbox, subj.bbox):
            relations.append(SpatialRelation(
                subject_id=subj.entity_id,
                object_id=obj.entity_id,
                relation=SpatialRelationType.CONTAINS,
                confidence=0.9,
                metadata={},
            ))

        if self._is_touching(subj.bbox, obj.bbox):
            relations.append(SpatialRelation(
                subject_id=subj.entity_id,
                object_id=obj.entity_id,
                relation=SpatialRelationType.TOUCHING,
                confidence=0.8,
                metadata={},
            ))

        if (subj.dimensions_3d is not None and obj.dimensions_3d is not None and
            subj.center_3d is not None and obj.center_3d is not None):
            if self._is_supporting(subj, obj):
                relations.append(SpatialRelation(
                    subject_id=subj.entity_id,
                    object_id=obj.entity_id,
                    relation=SpatialRelationType.SUPPORTING,
                    confidence=0.7,
                    metadata={},
                ))
            elif self._is_supported_by(subj, obj):
                relations.append(SpatialRelation(
                    subject_id=subj.entity_id,
                    object_id=obj.entity_id,
                    relation=SpatialRelationType.SUPPORTED_BY,
                    confidence=0.7,
                    metadata={},
                ))

        return relations

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-6)

    def _is_inside(self, inner: np.ndarray, outer: np.ndarray) -> bool:
        return (inner[0] >= outer[0] and inner[1] >= outer[1] and
                inner[2] <= outer[2] and inner[3] <= outer[3])

    def _is_touching(self, box1: np.ndarray, box2: np.ndarray, threshold: float = 5.0) -> bool:
        dx = max(box1[0], box2[0]) - min(box1[2], box2[2])
        dy = max(box1[1], box2[1]) - min(box1[3], box2[3])
        return dx <= threshold and dy <= threshold

    def _is_supporting(self, subj: SpatialEntity, obj: SpatialEntity) -> bool:
        if subj.center_3d is None or obj.center_3d is None:
            return False
        if subj.dimensions_3d is None or obj.dimensions_3d is None:
            return False

        subj_bottom = subj.center_3d[1] - subj.dimensions_3d[1] / 2
        obj_bottom = obj.center_3d[1] - obj.dimensions_3d[1] / 2

        if abs(subj_bottom - obj_bottom) > 0.1:
            return False

        dx = abs(subj.center_3d[0] - obj.center_3d[0])
        dz = abs(subj.center_3d[2] - obj.center_3d[2])
        if dx < subj.dimensions_3d[0] / 2 + obj.dimensions_3d[0] / 2:
            if dz < subj.dimensions_3d[2] / 2 + obj.dimensions_3d[2] / 2:
                return True
        return False

    def _is_supported_by(self, subj: SpatialEntity, obj: SpatialEntity) -> bool:
        return self._is_supporting(obj, subj)


class SceneGraphBuilder:
    def __init__(
        self,
        reasoner: Optional[SpatialReasoner] = None,
        max_entities: int = 100,
        relation_confidence_threshold: float = 0.3,
    ):
        self.reasoner = reasoner or SpatialReasoner()
        self.max_entities = max_entities
        self.relation_confidence_threshold = relation_confidence_threshold
        self.scene_graph = SceneGraph()

    def build(self, detections: list[Detection], depth_frame: Optional[DepthFrame] = None) -> SceneGraph:
        self.scene_graph = SceneGraph()

        entities = self._create_entities(detections, depth_frame)
        entities = entities[:self.max_entities]

        for entity in entities:
            self.scene_graph.add_entity(entity)

        relations = self.reasoner.compute_relations(entities)
        for rel in relations:
            if rel.confidence >= self.relation_confidence_threshold:
                self.scene_graph.add_relation(rel)

        return self.scene_graph

    def _create_entities(self, detections: list[Detection], depth_frame: Optional[DepthFrame]) -> list[SpatialEntity]:
        entities = []

        for i, det in enumerate(detections):
            entity_id = f"{det.class_name}_{i}"

            center_3d = None
            dimensions_3d = None

            if depth_frame is not None and depth_frame.intrinsics is not None:
                cx, cy = det.center_2d
                if 0 <= cx < depth_frame.depth.shape[1] and 0 <= cy < depth_frame.depth.shape[0]:
                    depth_val = depth_frame.depth[int(cy), int(cx)]
                    if depth_val > 0:
                        point_2d = np.array([[cx, cy]])
                        depth_arr = np.array([depth_val])
                        center_3d = depth_frame.intrinsics.unproject(point_2d, depth_arr)[0]

                        w, h = det.bbox[2] - det.bbox[0], det.bbox[3] - det.bbox[1]
                        corners_2d = np.array([
                            [det.bbox[0], det.bbox[1]],
                            [det.bbox[2], det.bbox[1]],
                            [det.bbox[0], det.bbox[3]],
                            [det.bbox[2], det.bbox[3]],
                        ])
                        corner_depths = []
                        for corner in corners_2d:
                            x, y = int(corner[0]), int(corner[1])
                            if 0 <= x < depth_frame.depth.shape[1] and 0 <= y < depth_frame.depth.shape[0]:
                                d = depth_frame.depth[y, x]
                                if d > 0:
                                    corner_depths.append(d)
                        if len(corner_depths) >= 3:
                            corner_points = depth_frame.intrinsics.unproject(corners_2d[:len(corner_depths)], np.array(corner_depths))
                            dimensions_3d = np.array([
                                np.max(corner_points[:, 0]) - np.min(corner_points[:, 0]),
                                np.max(corner_points[:, 1]) - np.min(corner_points[:, 1]),
                                np.max(corner_points[:, 2]) - np.min(corner_points[:, 2]),
                            ])

            entity = SpatialEntity(
                entity_id=entity_id,
                bbox=det.bbox,
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                center_3d=center_3d,
                dimensions_3d=dimensions_3d,
                mask=det.mask,
                keypoints=det.keypoints,
                attributes=det.attributes,
            )
            entities.append(entity)

        return entities


class TemporalSceneGraph:
    def __init__(
        self,
        max_history: int = 100,
        entity_ttl: float = 5.0,
    ):
        self.max_history = max_history
        self.entity_ttl = entity_ttl
        self.history: list[SceneGraph] = []
        self.entity_timestamps: dict[str, float] = {}

    def add_frame(self, scene_graph: SceneGraph, timestamp: float):
        self.history.append(scene_graph)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        for eid in scene_graph.entities:
            self.entity_timestamps[eid] = timestamp

        self._cleanup_old_entities(timestamp)

    def _cleanup_old_entities(self, current_time: float):
        to_remove = []
        for eid, ts in self.entity_timestamps.items():
            if current_time - ts > self.entity_ttl:
                to_remove.append(eid)
        for eid in to_remove:
            del self.entity_timestamps[eid]

    def get_entity_trajectory(self, entity_id: str) -> list[tuple[float, SpatialEntity]]:
        trajectory = []
        for sg in self.history:
            if entity_id in sg.entities:
                trajectory.append((sg.entities[entity_id].attributes.get('timestamp', 0), sg.entities[entity_id]))
        return trajectory

    def get_relation_history(self, subject_id: str, object_id: str, relation_type: SpatialRelationType) -> list[tuple[float, SpatialRelation]]:
        history = []
        for sg in self.history:
            for rel in sg.get_relations(subject_id, object_id):
                if rel.relation == relation_type:
                    history.append((sg.entities[subject_id].attributes.get('timestamp', 0), rel))
        return history

    def predict_relations(self, entity_id: str, future_time: float) -> list[SpatialRelation]:
        if len(self.history) < 2:
            return []

        current_sg = self.history[-1]
        if entity_id not in current_sg.entities:
            return []

        predictions = []
        for rel in current_sg.get_relations(entity_id):
            history = self.get_relation_history(entity_id, rel.object_id, rel.relation)
            if len(history) >= 2:
                confidences = [h[1].confidence for h in history]
                trend = np.polyfit(range(len(confidences)), confidences, 1)[0]
                predicted_conf = confidences[-1] + trend
                if predicted_conf > 0.3:
                    predictions.append(SpatialRelation(
                        subject_id=rel.subject_id,
                        object_id=rel.object_id,
                        relation=rel.relation,
                        confidence=max(0, min(1, predicted_conf)),
                        metadata={'predicted': True, 'trend': float(trend)},
                    ))
        return predictions


class SceneGraphQueryEngine:
    def __init__(self, scene_graph: SceneGraph):
        self.scene_graph = scene_graph

    def find_objects_near(self, entity_id: str, max_distance: float = 100) -> list[SpatialEntity]:
        relations = self.scene_graph.get_relations(entity_id)
        near = []
        for rel in relations:
            if rel.relation == SpatialRelationType.NEAR and rel.confidence > 0.5:
                obj = self.scene_graph.entities.get(rel.object_id)
                if obj:
                    near.append(obj)
        return near

    def find_supporting(self, entity_id: str) -> list[SpatialEntity]:
        relations = self.scene_graph.get_relations(entity_id)
        supporting = []
        for rel in relations:
            if rel.relation == SpatialRelationType.SUPPORTED_BY:
                obj = self.scene_graph.entities.get(rel.object_id)
                if obj:
                    supporting.append(obj)
        return supporting

    def find_contained_in(self, entity_id: str) -> list[SpatialEntity]:
        relations = self.scene_graph.get_relations(entity_id)
        containers = []
        for rel in relations:
            if rel.relation == SpatialRelationType.INSIDE:
                obj = self.scene_graph.entities.get(rel.object_id)
                if obj:
                    containers.append(obj)
        return containers

    def find_path(self, start_id: str, end_id: str, relation_types: list[SpatialRelationType] = None) -> list[SpatialEntity]:
        try:
            import networkx as nx
            G = self.scene_graph.to_networkx()
            if G is None or not G.has_node(start_id) or not G.has_node(end_id):
                return []

            if relation_types:
                edges = [(u, v) for u, v, d in G.edges(data=True)
                         if d['relation'] in [rt.value for rt in relation_types]]
                G = G.edge_subgraph(edges)

            path = nx.shortest_path(G, start_id, end_id)
            return [self.scene_graph.entities[pid] for pid in path if pid in self.scene_graph.entities]
        except ImportError:
            return []

    def count_by_class(self, class_id: int) -> int:
        return len(self.scene_graph.get_entities_by_class(class_id))

    def get_spatial_summary(self) -> dict:
        summary = {
            'num_entities': len(self.scene_graph.entities),
            'num_relations': len(self.scene_graph.relations),
            'classes': {},
            'relation_types': {},
        }

        for entity in self.scene_graph.entities.values():
            class_name = entity.class_name
            if class_name not in summary['classes']:
                summary['classes'][class_name] = 0
            summary['classes'][class_name] += 1

        for rel in self.scene_graph.relations:
            rel_type = rel.relation.value
            if rel_type not in summary['relation_types']:
                summary['relation_types'][rel_type] = 0
            summary['relation_types'][rel_type] += 1

        return summary


from core.domain.vision.detection_heads import Detection
from core.domain.vision.depth_estimation import DepthFrame
