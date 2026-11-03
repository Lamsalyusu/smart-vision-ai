from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
import random
from collections import defaultdict, Counter


class SceneGraphNodeType(Enum):
    ENTITY = "entity"
    OBJECT = "object"
    PERSON = "person"
    VEHICLE = "vehicle"
    ROOM = "room"
    REGION = "region"
    SCENE = "scene"


class RelationType(Enum):
    SPATIAL = "spatial"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    FUNCTIONAL = "functional"
    SOCIAL = "social"
    PART_OF = "part_of"
    CONTAINS = "contains"
    SUPPORTS = "supports"
    ATTACHED_TO = "attached_to"
    NEAR = "near"
    INSIDE = "inside"
    ON = "on"
    UNDER = "under"
    ABOVE = "above"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    FACING = "facing"
    LOOKING_AT = "looking_at"
    HOLDING = "holding"
    CARRYING = "carrying"
    USING = "using"
    INTERACTING_WITH = "interacting_with"
    TALKING_TO = "talking_to"
    FOLLOWING = "following"
    LEADING = "leading"


@dataclass(frozen=True, slots=True)
class SceneGraphNode:
    node_id: int
    node_type: SceneGraphNodeType
    label: str
    position: np.ndarray
    dimensions: np.ndarray
    orientation: np.ndarray
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    embedding: Optional[np.ndarray] = None


@dataclass(frozen=True, slots=True)
class SceneGraphEdge:
    edge_id: int
    source_id: int
    target_id: int
    relation_type: RelationType
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class SceneGraph:
    nodes: Dict[int, SceneGraphNode]
    edges: Dict[Tuple[int, int], SceneGraphEdge]
    node_counter: int = 0
    edge_counter: int = 0
    timestamp: float = field(default_factory=time.time)


class SceneGraphBuilder:
    def __init__(self):
        self.graph = SceneGraph(nodes={}, edges={})
        self.node_counter = 0
        self.edge_counter = 0

    def add_node(
        self,
        node_type: SceneGraphNodeType,
        label: str,
        position: np.ndarray,
        dimensions: np.ndarray = None,
        orientation: np.ndarray = None,
        attributes: Dict[str, Any] = None,
        confidence: float = 1.0,
        embedding: Optional[np.ndarray] = None
    ) -> int:
        node_id = self.node_counter
        self.node_counter += 1

        node = SceneGraphNode(
            node_id=node_id,
            node_type=node_type,
            label=label,
            position=position,
            dimensions=dimensions if dimensions is not None else np.ones(3),
            orientation=orientation if orientation is not None else np.eye(3),
            attributes=attributes or {},
            confidence=confidence,
            embedding=embedding
        )

        self.graph.nodes[node_id] = node
        return node_id

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        relation_type: RelationType,
        confidence: float = 1.0,
        attributes: Dict[str, Any] = None
    ) -> int:
        edge_id = self.edge_counter
        self.edge_counter += 1

        edge = SceneGraphEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            confidence=confidence,
            attributes=attributes or {}
        )

        self.graph.edges[(source_id, target_id)] = edge
        return edge_id

    def get_node(self, node_id: int) -> Optional[SceneGraphNode]:
        return self.graph.nodes.get(node_id)

    def get_edges(self, node_id: int) -> List[SceneGraphEdge]:
        edges = []
        for (src, tgt), edge in self.graph.edges.items():
            if src == node_id or tgt == node_id:
                edges.append(edge)
        return edges

    def get_neighbors(self, node_id: int) -> List[int]:
        neighbors = []
        for (src, tgt) in self.graph.edges.keys():
            if src == node_id:
                neighbors.append(tgt)
            elif tgt == node_id:
                neighbors.append(src)
        return neighbors


class SceneGraphReasoner:
    def __init__(self, graph: SceneGraph):
        self.graph = graph

    def query_related(
        self,
        node_id: int,
        relation_types: Optional[List[RelationType]] = None,
        max_hops: int = 2
    ) -> List[SceneGraphNode]:
        visited = set()
        results = []

        def dfs(current_id: int, depth: int):
            if depth > max_hops or current_id in visited:
                return
            visited.add(current_id)

            for (src, tgt), edge in self.graph.edges.items():
                if src == node_id or tgt == node_id:
                    if relation_types is None or edge.relation_type in relation_types:
                        other_id = tgt if src == node_id else src
                        if other_id not in visited:
                            other_node = self.graph.nodes.get(other_id)
                            if other_node:
                                results.append(other_node)
                                dfs(other_id, depth + 1)

        dfs(node_id, 0)
        return results

    def find_path(
        self,
        start_id: int,
        goal_id: int,
        relation_types: Optional[List[RelationType]] = None
    ) -> List[int]:
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

            for (src, tgt), edge in self.graph.edges.items():
                if src == current:
                    neighbor = tgt
                elif tgt == current:
                    neighbor = src
                else:
                    continue

                if relation_types and edge.relation_type not in relation_types:
                    continue

                tentative_g = g_score[current] + 1.0 / max(edge.confidence, 0.1)
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_id)
                    heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def _heuristic(self, a: int, b: int) -> float:
        if a not in self.graph.nodes or b not in self.graph.nodes:
            return 0.0
        return np.linalg.norm(self.graph.nodes[a].position - self.graph.nodes[b].position)

    def infer_relations(self, node_id: int) -> List[Tuple[int, RelationType, float]]:
        inferred = []

        for (src, tgt), edge in self.graph.edges.items():
            if src == node_id:
                inferred.append((tgt, edge.relation_type, edge.confidence))
            elif tgt == node_id:
                inferred.append((src, self._inverse_relation(edge.relation_type), edge.confidence))

        return inferred

    def _inverse_relation(self, relation: RelationType) -> RelationType:
        inverses = {
            RelationType.PART_OF: RelationType.CONTAINS,
            RelationType.CONTAINS: RelationType.PART_OF,
            RelationType.SUPPORTS: RelationType.SUPPORTS,
            RelationType.ATTACHED_TO: RelationType.ATTACHED_TO,
            RelationType.NEAR: RelationType.NEAR,
            RelationType.INSIDE: RelationType.CONTAINS,
            RelationType.ON: RelationType.SUPPORTS,
            RelationType.UNDER: RelationType.ABOVE,
            RelationType.ABOVE: RelationType.UNDER,
            RelationType.LEFT_OF: RelationType.RIGHT_OF,
            RelationType.RIGHT_OF: RelationType.LEFT_OF,
            RelationType.IN_FRONT_OF: RelationType.BEHIND,
            RelationType.BEHIND: RelationType.IN_FRONT_OF,
            RelationType.FACING: RelationType.FACING,
            RelationType.LOOKING_AT: RelationType.LOOKING_AT,
            RelationType.HOLDING: RelationType.HELD_BY,
            RelationType.CARRYING: RelationType.CARRIED_BY,
            RelationType.USING: RelationType.USED_BY,
            RelationType.INTERACTING_WITH: RelationType.INTERACTING_WITH,
            RelationType.TALKING_TO: RelationType.TALKING_TO,
            RelationType.FOLLOWING: RelationType.LEADING,
            RelationType.LEADING: RelationType.FOLLOWING,
        }
        return inverses.get(relation, relation)

    def find_subgraph(self, node_ids: List[int]) -> SceneGraph:
        subgraph_nodes = {nid: self.graph.nodes[nid] for nid in node_ids if nid in self.graph.nodes}
        subgraph_edges = {}

        for (src, tgt), edge in self.graph.edges.items():
            if src in node_ids and tgt in node_ids:
                subgraph_edges[(src, tgt)] = edge

        return SceneGraph(
            nodes=subgraph_nodes,
            edges=subgraph_edges
        )


class SceneGraphEmbedder:
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.node_embeddings: Dict[int, np.ndarray] = {}
        self.relation_embeddings: Dict[RelationType, np.ndarray] = {}

    def compute_embeddings(self, graph: SceneGraph):
        self.node_embeddings.clear()

        for node_id, node in graph.nodes.items():
            if node.embedding is not None:
                self.node_embeddings[node_id] = node.embedding
            else:
                self.node_embeddings[node_id] = self._generate_node_embedding(node)

        for rel_type in RelationType:
            self.relation_embeddings[rel_type] = np.random.randn(self.embedding_dim)

    def _generate_node_embedding(self, node: SceneGraphNode) -> np.ndarray:
        features = []

        class_onehot = np.zeros(len(SceneGraphNodeType))
        class_onehot[list(SceneGraphNodeType).index(node.node_type)] = 1
        features.append(class_onehot)

        features.append(node.position / 100.0)
        features.append(node.dimensions / 10.0)

        if node.embedding is not None:
            features.append(node.embedding)

        embedding = np.concatenate(features) if features else np.zeros(self.embedding_dim)

        if len(embedding) > self.embedding_dim:
            embedding = embedding[:self.embedding_dim]
        elif len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))

        return embedding / (np.linalg.norm(embedding) + 1e-6)

    def query_similar_nodes(
        self,
        graph: SceneGraph,
        query_node_id: int,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        if query_node_id not in self.node_embeddings:
            return []

        query_emb = self.node_embeddings[query_node_id]
        similarities = []

        for node_id, emb in self.node_embeddings.items():
            if node_id == query_node_id:
                continue
            sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8)
            similarities.append((node_id, float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def predict_relation(
        self,
        graph: SceneGraph,
        source_id: int,
        target_id: int
    ) -> List[Tuple[RelationType, float]]:
        if source_id not in self.node_embeddings or target_id not in self.node_embeddings:
            return []

        source_emb = self.node_embeddings[source_id]
        target_emb = self.node_embeddings[target_id]

        combined = np.concatenate([source_emb, target_emb])

        predictions = []
        for rel_type, rel_emb in self.relation_embeddings.items():
            score = np.dot(combined, rel_emb) / (np.linalg.norm(combined) * np.linalg.norm(rel_emb) + 1e-8)
            predictions.append((rel_type, float(score)))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:5]


class SceneGraphUpdater:
    def __init__(
        self,
        graph: SceneGraph,
        max_age: float = 3600.0,
        confidence_decay: float = 0.001
    ):
        self.graph = graph
        self.max_age = max_age
        self.confidence_decay = confidence_decay

    def update(self, dt: float):
        current_time = time.time()
        to_remove = []

        for node_id, node in self.graph.nodes.items():
            age = current_time - node.timestamp
            if age > self.max_age:
                to_remove.append(node_id)
            else:
                decayed_confidence = node.confidence * math.exp(-self.confidence_decay * age)
                new_node = SceneGraphNode(
                    node_id=node.node_id,
                    node_type=node.node_type,
                    label=node.label,
                    position=node.position,
                    dimensions=node.dimensions,
                    orientation=node.orientation,
                    attributes=node.attributes,
                    confidence=decayed_confidence,
                    timestamp=node.timestamp,
                    embedding=node.embedding
                )
                self.graph.nodes[node_id] = new_node

        for node_id in to_remove:
            del self.graph.nodes[node_id]
            edges_to_remove = [
                (src, tgt) for (src, tgt) in self.graph.edges.keys()
                if src == node_id or tgt == node_id
            ]
            for edge_key in edges_to_remove:
                del self.graph.edges[edge_key]

        edges_to_remove = []
        for (src, tgt), edge in self.graph.edges.items():
            age = current_time - edge.timestamp
            if age > self.max_age:
                edges_to_remove.append((src, tgt))

        for edge_key in edges_to_remove:
            del self.graph.edges[edge_key]


class SceneGraphVisualizer:
    @staticmethod
    def draw_graph(
        graph: SceneGraph,
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if not graph.nodes:
            return vis

        positions = np.array([node.position[:2] for node in graph.nodes.values()])
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1

        scale = min(image_size[0], image_size[1]) * 0.8 / max(range_coords)

        node_positions = {}
        for node_id, node in graph.nodes.items():
            x = int((node.position[0] - min_coords[0]) * scale + image_size[0] * 0.1)
            y = int(image_size[1] - (node.position[1] - min_coords[1]) * scale - image_size[1] * 0.1)
            node_positions[node_id] = (x, y)

        for (src, tgt), edge in graph.edges.items():
            if src in node_positions and tgt in node_positions:
                color = {
                    RelationType.SPATIAL: (100, 100, 100),
                    RelationType.SEMANTIC: (0, 255, 0),
                    RelationType.TEMPORAL: (0, 0, 255),
                    RelationType.FUNCTIONAL: (255, 128, 0),
                    RelationType.SOCIAL: (255, 0, 255),
                    RelationType.PART_OF: (0, 128, 128),
                    RelationType.CONTAINS: (128, 0, 128),
                }.get(edge.relation_type, (255, 255, 255))

                cv2.line(vis, node_positions[src], node_positions[tgt], color, 2)

        for node_id, node in graph.nodes.items():
            if node_id in node_positions:
                x, y = node_positions[node_id]
                color = {
                    SceneGraphNodeType.ENTITY: (255, 255, 255),
                    SceneGraphNodeType.OBJECT: (0, 255, 255),
                    SceneGraphNodeType.PERSON: (255, 0, 255),
                    SceneGraphNodeType.VEHICLE: (255, 255, 0),
                    SceneGraphNodeType.ROOM: (0, 255, 0),
                    SceneGraphNodeType.REGION: (128, 128, 0),
                    SceneGraphNodeType.SCENE: (0, 0, 255),
                }.get(node.node_type, (255, 255, 255))

                cv2.circle(vis, (x, y), 8, color, -1)
                cv2.putText(vis, node.label[:10], (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return vis

    @staticmethod
    def draw_subgraph(
        graph: SceneGraph,
        node_ids: List[int],
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        subgraph = SceneGraph(
            nodes={nid: graph.nodes[nid] for nid in node_ids if nid in graph.nodes},
            edges={
                (src, tgt): edge for (src, tgt), edge in graph.edges.items()
                if src in node_ids and tgt in node_ids
            }
        )
        return SceneGraphVisualizer.draw_graph(subgraph, image_size)


class SceneGraphSerializer:
    @staticmethod
    def to_json(graph: SceneGraph) -> str:
        import json

        data = {
            'nodes': [],
            'edges': [],
            'timestamp': graph.timestamp
        }

        for node_id, node in graph.nodes.items():
            data['nodes'].append({
                'node_id': node.node_id,
                'node_type': node.node_type.value,
                'label': node.label,
                'position': node.position.tolist(),
                'dimensions': node.dimensions.tolist(),
                'orientation': node.orientation.tolist(),
                'attributes': node.attributes,
                'confidence': node.confidence,
                'timestamp': node.timestamp,
                'embedding': node.embedding.tolist() if node.embedding is not None else None
            })

        for (src, tgt), edge in graph.edges.items():
            data['edges'].append({
                'edge_id': edge.edge_id,
                'source_id': edge.source_id,
                'target_id': edge.target_id,
                'relation_type': edge.relation_type.value,
                'confidence': edge.confidence,
                'attributes': edge.attributes,
                'timestamp': edge.timestamp
            })

        return json.dumps(data, indent=2)

    @staticmethod
    def from_json(json_str: str) -> SceneGraph:
        import json
        data = json.loads(json_str)

        nodes = {}
        for node_data in data['nodes']:
            node = SceneGraphNode(
                node_id=node_data['node_id'],
                node_type=SceneGraphNodeType(node_data['node_type']),
                label=node_data['label'],
                position=np.array(node_data['position']),
                dimensions=np.array(node_data['dimensions']),
                orientation=np.array(node_data['orientation']),
                attributes=node_data['attributes'],
                confidence=node_data['confidence'],
                timestamp=node_data['timestamp'],
                embedding=np.array(node_data['embedding']) if node_data.get('embedding') else None
            )
            nodes[node_data['node_id']] = node

        edges = {}
        for edge_data in data['edges']:
            edge = SceneGraphEdge(
                edge_id=edge_data['edge_id'],
                source_id=edge_data['source_id'],
                target_id=edge_data['target_id'],
                relation_type=RelationType(edge_data['relation_type']),
                confidence=edge_data['confidence'],
                attributes=edge_data['attributes'],
                timestamp=edge_data['timestamp']
            )
            edges[(edge_data['source_id'], edge_data['target_id'])] = edge

        return SceneGraph(
            nodes=nodes,
            edges=edges,
            timestamp=data.get('timestamp', time.time())
        )


class SceneGraphMetrics:
    @staticmethod
    def node_count(graph: SceneGraph) -> int:
        return len(graph.nodes)

    @staticmethod
    def edge_count(graph: SceneGraph) -> int:
        return len(graph.edges)

    @staticmethod
    def average_degree(graph: SceneGraph) -> float:
        if not graph.nodes:
            return 0.0
        total_degree = 0
        for node_id in graph.nodes:
            degree = sum(1 for (src, tgt) in graph.edges.keys() if src == node_id or tgt == node_id)
            total_degree += degree
        return total_degree / len(graph.nodes)

    @staticmethod
    def clustering_coefficient(graph: SceneGraph) -> float:
        if len(graph.nodes) < 3:
            return 0.0

        total_clustering = 0.0
        for node_id in graph.nodes:
            neighbors = set()
            for (src, tgt) in graph.edges.keys():
                if src == node_id:
                    neighbors.add(tgt)
                elif tgt == node_id:
                    neighbors.add(src)

            if len(neighbors) < 2:
                continue

            edges_between_neighbors = 0
            neighbor_list = list(neighbors)
            for i in range(len(neighbor_list)):
                for j in range(i + 1, len(neighbor_list)):
                    if (neighbor_list[i], neighbor_list[j]) in graph.edges or \
                       (neighbor_list[j], neighbor_list[i]) in graph.edges:
                        edges_between_neighbors += 1

            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
            if possible_edges > 0:
                total_clustering += edges_between_neighbors / possible_edges

        return total_clustering / len(graph.nodes) if graph.nodes else 0.0

    @staticmethod
    def type_distribution(graph: SceneGraph) -> Dict[SceneGraphNodeType, int]:
        dist = defaultdict(int)
        for node in graph.nodes.values():
            dist[node.node_type] += 1
        return dict(dist)

    @staticmethod
    def relation_distribution(graph: SceneGraph) -> Dict[RelationType, int]:
        dist = defaultdict(int)
        for edge in graph.edges.values():
            dist[edge.relation_type] += 1
        return dict(dist)


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
