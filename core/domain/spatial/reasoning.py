from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Union, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict, deque
from scipy.spatial import KDTree, cKDTree, ConvexHull, Delaunay, Voronoi
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.optimize import linear_sum_assignment
from scipy.linalg import svd, eig, pinv
import warnings

class SpatialRelationType(Enum):
    CONTAINS = "contains"
    WITHIN = "within"
    OVERLAPS = "overlaps"
    DISJOINT = "disjoint"
    TOUCHES = "touches"
    INTERSECTS = "intersects"
    COVERS = "covers"
    COVERED_BY = "covered_by"
    EQUALS = "equals"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    FRONT_OF = "front_of"
    BEHIND = "behind"
    NEAR = "near"
    FAR = "far"

class GeometryType(Enum):
    POINT = "point"
    LINE = "line"
    POLYGON = "polygon"
    POLYLINE = "polyline"
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    BOX3D = "box3d"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    CONE = "cone"
    MESH = "mesh"
    POINT_CLOUD = "point_cloud"

class CoordinateSystem(Enum):
    CARTESIAN_2D = "cartesian_2d"
    CARTESIAN_3D = "cartesian_3d"
    POLAR = "polar"
    SPHERICAL = "spherical"
    CYLINDRICAL = "cylindrical"
    GEOGRAPHIC = "geographic"
    PROJECTED = "projected"

@dataclass(frozen=True, slots=True)
class Point2D:
    x: float
    y: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def distance_to(self, other: 'Point2D') -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def __add__(self, other: 'Point2D') -> 'Point2D':
        return Point2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point2D') -> 'Point2D':
        return Point2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Point2D':
        return Point2D(self.x * scalar, self.y * scalar)

@dataclass(frozen=True, slots=True)
class Point3D:
    x: float
    y: float
    z: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def distance_to(self, other: 'Point3D') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

    def __add__(self, other: 'Point3D') -> 'Point3D':
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Point3D') -> 'Point3D':
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Point3D':
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)

@dataclass(frozen=True, slots=True)
class BoundingBox2D:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def center(self) -> Point2D:
        return Point2D((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains(self, point: Point2D) -> bool:
        return (self.min_x <= point.x <= self.max_x and
                self.min_y <= point.y <= self.max_y)

    def intersects(self, other: 'BoundingBox2D') -> bool:
        return not (self.max_x < other.min_x or self.min_x > other.max_x or
                    self.max_y < other.min_y or self.min_y > other.max_y)

    def union(self, other: 'BoundingBox2D') -> 'BoundingBox2D':
        return BoundingBox2D(
            min(self.min_x, other.min_x),
            min(self.min_y, other.min_y),
            max(self.max_x, other.max_x),
            max(self.max_y, other.max_y)
        )

    def intersection(self, other: 'BoundingBox2D') -> Optional['BoundingBox2D']:
        if not self.intersects(other):
            return None
        return BoundingBox2D(
            max(self.min_x, other.min_x),
            max(self.min_y, other.min_y),
            min(self.max_x, other.max_x),
            min(self.max_y, other.max_y)
        )

    def iou(self, other: 'BoundingBox2D') -> float:
        inter = self.intersection(other)
        if inter is None:
            return 0.0
        inter_area = inter.area
        union_area = self.area + other.area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

@dataclass(frozen=True, slots=True)
class BoundingBox3D:
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    @property
    def center(self) -> Point3D:
        return Point3D(
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2,
            (self.min_z + self.max_z) / 2
        )

    @property
    def volume(self) -> float:
        return (self.max_x - self.min_x) * (self.max_y - self.min_y) * (self.max_z - self.min_z)

    def contains(self, point: Point3D) -> bool:
        return (self.min_x <= point.x <= self.max_x and
                self.min_y <= point.y <= self.max_y and
                self.min_z <= point.z <= self.max_z)

    def intersects(self, other: 'BoundingBox3D') -> bool:
        return not (self.max_x < other.min_x or self.min_x > other.max_x or
                    self.max_y < other.min_y or self.min_y > other.max_y or
                    self.max_z < other.min_z or self.min_z > other.max_z)

@dataclass(frozen=True, slots=True)
class Polygon2D:
    vertices: List[Point2D]
    holes: List[List[Point2D]] = field(default_factory=list)

    @property
    def area(self) -> float:
        if len(self.vertices) < 3:
            return 0.0
        area = 0.0
        for i in range(len(self.vertices)):
            j = (i + 1) % len(self.vertices)
            area += self.vertices[i].x * self.vertices[j].y
            area -= self.vertices[j].x * self.vertices[i].y
        area = abs(area) / 2.0

        for hole in self.holes:
            if len(hole) >= 3:
                hole_area = 0.0
                for i in range(len(hole)):
                    j = (i + 1) % len(hole)
                    hole_area += hole[i].x * hole[j].y
                    hole_area -= hole[j].x * hole[i].y
                area -= abs(hole_area) / 2.0

        return area

    @property
    def centroid(self) -> Point2D:
        if len(self.vertices) < 3:
            return self.vertices[0] if self.vertices else Point2D(0, 0)

        cx = cy = 0.0
        area = 0.0
        for i in range(len(self.vertices)):
            j = (i + 1) % len(self.vertices)
            cross = self.vertices[i].x * self.vertices[j].y - self.vertices[j].x * self.vertices[i].y
            area += cross
            cx += (self.vertices[i].x + self.vertices[j].x) * cross
            cy += (self.vertices[i].y + self.vertices[j].y) * cross

        area *= 0.5
        if abs(area) < 1e-10:
            return self.vertices[0]

        cx /= (6 * area)
        cy /= (6 * area)
        return Point2D(cx, cy)

    def bounding_box(self) -> BoundingBox2D:
        if not self.vertices:
            return BoundingBox2D(0, 0, 0, 0)
        xs = [v.x for v in self.vertices]
        ys = [v.y for v in self.vertices]
        return BoundingBox2D(min(xs), min(ys), max(xs), max(ys))

    def contains_point(self, point: Point2D) -> bool:
        if len(self.vertices) < 3:
            return False

        inside = False
        for i in range(len(self.vertices)):
            j = (i + 1) % len(self.vertices)
            vi, vj = self.vertices[i], self.vertices[j]
            if ((vi.y > point.y) != (vj.y > point.y)) and \
               (point.x < (vj.x - vi.x) * (point.y - vi.y) / (vj.y - vi.y) + vi.x):
                inside = not inside

        for hole in self.holes:
            if len(hole) >= 3:
                hole_inside = False
                for i in range(len(hole)):
                    j = (i + 1) % len(hole)
                    hi, hj = hole[i], hole[j]
                    if ((hi.y > point.y) != (hj.y > point.y)) and \
                       (point.x < (hj.x - hi.x) * (point.y - hi.y) / (hj.y - hi.y) + hi.x):
                        hole_inside = not hole_inside
                if hole_inside:
                    return False

        return inside

@dataclass(frozen=True, slots=True)
class Circle2D:
    center: Point2D
    radius: float

    @property
    def area(self) -> float:
        return math.pi * self.radius ** 2

    @property
    def circumference(self) -> float:
        return 2 * math.pi * self.radius

    def contains(self, point: Point2D) -> bool:
        return self.center.distance_to(point) <= self.radius

    def intersects(self, other: 'Circle2D') -> bool:
        return self.center.distance_to(other.center) <= (self.radius + other.radius)

@dataclass(frozen=True, slots=True)
class Sphere3D:
    center: Point3D
    radius: float

    @property
    def volume(self) -> float:
        return 4/3 * math.pi * self.radius ** 3

    @property
    def surface_area(self) -> float:
        return 4 * math.pi * self.radius ** 2

    def contains(self, point: Point3D) -> bool:
        return self.center.distance_to(point) <= self.radius

@dataclass(frozen=True, slots=True)
class SpatialRelation:
    subject_id: str
    object_id: str
    relation: SpatialRelationType
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class SpatialIndex:
    def __init__(self, dimension: int = 2):
        self.dimension = dimension
        self.points: List[np.ndarray] = []
        self.ids: List[str] = []
        self.tree: Optional[cKDTree] = None
        self.dirty = False

    def insert(self, id: str, point: np.ndarray):
        self.points.append(point)
        self.ids.append(id)
        self.dirty = True

    def remove(self, id: str) -> bool:
        if id in self.ids:
            idx = self.ids.index(id)
            self.points.pop(idx)
            self.ids.pop(idx)
            self.dirty = True
            return True
        return False

    def _build_tree(self):
        if self.dirty and self.points:
            self.tree = cKDTree(np.array(self.points))
            self.dirty = False

    def query_radius(self, point: np.ndarray, radius: float) -> List[Tuple[str, float]]:
        self._build_tree()
        if self.tree is None:
            return []
        indices = self.tree.query_ball_point(point, radius)
        return [(self.ids[i], np.linalg.norm(self.points[i] - point)) for i in indices]

    def query_knn(self, point: np.ndarray, k: int) -> List[Tuple[str, float]]:
        self._build_tree()
        if self.tree is None or k <= 0:
            return []
        k = min(k, len(self.points))
        distances, indices = self.tree.query(point, k=k)
        if k == 1:
            return [(self.ids[indices], distances)]
        return [(self.ids[i], d) for i, d in zip(indices, distances)]

    def query_bbox(self, min_bounds: np.ndarray, max_bounds: np.ndarray) -> List[str]:
        self._build_tree()
        if self.tree is None:
            return []
        indices = self.tree.query_ball_point((min_bounds + max_bounds) / 2,
                                              np.max(max_bounds - min_bounds) / 2)
        result = []
        for i in indices:
            p = self.points[i]
            if np.all(p >= min_bounds) and np.all(p <= max_bounds):
                result.append(self.ids[i])
        return result

class SpatialReasoner:
    def __init__(self):
        self.objects: Dict[str, Any] = {}
        self.relations: List[SpatialRelation] = []
        self.index_2d = SpatialIndex(2)
        self.index_3d = SpatialIndex(3)

    def add_object(self, obj_id: str, geometry: Any):
        self.objects[obj_id] = geometry

        if isinstance(geometry, (Point2D, Point3D)):
            if isinstance(geometry, Point2D):
                self.index_2d.insert(obj_id, geometry.to_array())
            else:
                self.index_3d.insert(obj_id, geometry.to_array())
        elif isinstance(geometry, (BoundingBox2D, BoundingBox3D)):
            center = geometry.center
            if isinstance(center, Point2D):
                self.index_2d.insert(obj_id, center.to_array())
            else:
                self.index_3d.insert(obj_id, center.to_array())
        elif isinstance(geometry, (Polygon2D, Circle2D, Sphere3D)):
            center = geometry.center if hasattr(geometry, 'center') else geometry.center
            if isinstance(center, Point2D):
                self.index_2d.insert(obj_id, center.to_array())
            else:
                self.index_3d.insert(obj_id, center.to_array())

    def remove_object(self, obj_id: str):
        if obj_id in self.objects:
            del self.objects[obj_id]
        self.index_2d.remove(obj_id)
        self.index_3d.remove(obj_id)

    def compute_relations(self, obj_id: str, radius: float = float('inf')) -> List[SpatialRelation]:
        if obj_id not in self.objects:
            return []

        obj = self.objects[obj_id]
        relations = []

        if isinstance(obj, Point2D):
            neighbors = self.index_2d.query_radius(obj.to_array(), radius)
        elif isinstance(obj, Point3D):
            neighbors = self.index_3d.query_radius(obj.to_array(), radius)
        else:
            center = obj.center if hasattr(obj, 'center') else None
            if center is None:
                return []
            if isinstance(center, Point2D):
                neighbors = self.index_2d.query_radius(center.to_array(), radius)
            else:
                neighbors = self.index_3d.query_radius(center.to_array(), radius)

        for other_id, dist in neighbors:
            if other_id == obj_id:
                continue
            other = self.objects.get(other_id)
            if other is None:
                continue

            relation = self._compute_relation(obj, other, obj_id, other_id)
            if relation:
                relations.append(relation)

        return relations

    def _compute_relation(self, obj1: Any, obj2: Any, id1: str, id2: str) -> Optional[SpatialRelation]:
        if isinstance(obj1, Point2D) and isinstance(obj2, Point2D):
            return self._point_point_relation(obj1, obj2, id1, id2)
        elif isinstance(obj1, BoundingBox2D) and isinstance(obj2, BoundingBox2D):
            return self._bbox_bbox_relation(obj1, obj2, id1, id2)
        elif isinstance(obj1, Polygon2D) and isinstance(obj2, Polygon2D):
            return self._polygon_polygon_relation(obj1, obj2, id1, id2)
        elif isinstance(obj1, Circle2D) and isinstance(obj2, Circle2D):
            return self._circle_circle_relation(obj1, obj2, id1, id2)
        elif isinstance(obj1, Point2D) and isinstance(obj2, (BoundingBox2D, Polygon2D, Circle2D)):
            return self._point_geometry_relation(obj1, obj2, id1, id2)
        elif isinstance(obj1, (BoundingBox2D, Polygon2D, Circle2D)) and isinstance(obj2, Point2D):
            rel = self._point_geometry_relation(obj2, obj1, id2, id1)
            if rel:
                return SpatialRelation(id1, id2, self._inverse_relation(rel.relation), rel.confidence)
        return None

    def _point_point_relation(self, p1: Point2D, p2: Point2D, id1: str, id2: str) -> SpatialRelation:
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dist = math.hypot(dx, dy)

        if dist < 1e-6:
            return SpatialRelation(id1, id2, SpatialRelationType.EQUALS, 1.0)

        angle = math.atan2(dy, dx)

        if abs(angle) < math.pi / 4:
            rel = SpatialRelationType.RIGHT_OF
        elif abs(angle) > 3 * math.pi / 4:
            rel = SpatialRelationType.LEFT_OF
        elif angle > 0:
            rel = SpatialRelationType.ABOVE
        else:
            rel = SpatialRelationType.BELOW

        return SpatialRelation(id1, id2, rel, 1.0, {"distance": dist, "angle": angle})

    def _bbox_bbox_relation(self, b1: BoundingBox2D, b2: BoundingBox2D, id1: str, id2: str) -> SpatialRelation:
        if not b1.intersects(b2):
            return SpatialRelation(id1, id2, SpatialRelationType.DISJOINT, 1.0)

        inter = b1.intersection(b2)
        if inter is None:
            return SpatialRelation(id1, id2, SpatialRelationType.DISJOINT, 1.0)

        if inter.area == b1.area and inter.area == b2.area:
            return SpatialRelation(id1, id2, SpatialRelationType.EQUALS, 1.0)

        if inter.area == b1.area:
            return SpatialRelation(id1, id2, SpatialRelationType.WITHIN, 1.0)
        if inter.area == b2.area:
            return SpatialRelation(id1, id2, SpatialRelationType.CONTAINS, 1.0)

        iou = b1.iou(b2)
        if iou > 0.5:
            return SpatialRelation(id1, id2, SpatialRelationType.OVERLAPS, iou)
        else:
            return SpatialRelation(id1, id2, SpatialRelationType.TOUCHES, iou)

    def _polygon_polygon_relation(self, p1: Polygon2D, p2: Polygon2D, id1: str, id2: str) -> SpatialRelation:
        bbox1 = p1.bounding_box()
        bbox2 = p2.bounding_box()

        if not bbox1.intersects(bbox2):
            return SpatialRelation(id1, id2, SpatialRelationType.DISJOINT, 1.0)

        center1_in_2 = p2.contains_point(p1.centroid)
        center2_in_1 = p1.contains_point(p2.centroid)

        if center1_in_2 and center2_in_1:
            return SpatialRelation(id1, id2, SpatialRelationType.OVERLAPS, 0.8)
        elif center1_in_2:
            return SpatialRelation(id1, id2, SpatialRelationType.WITHIN, 0.9)
        elif center2_in_1:
            return SpatialRelation(id1, id2, SpatialRelationType.CONTAINS, 0.9)
        else:
            return SpatialRelation(id1, id2, SpatialRelationType.OVERLAPS, 0.5)

    def _circle_circle_relation(self, c1: Circle2D, c2: Circle2D, id1: str, id2: str) -> SpatialRelation:
        dist = c1.center.distance_to(c2.center)
        r1, r2 = c1.radius, c2.radius

        if dist >= r1 + r2:
            return SpatialRelation(id1, id2, SpatialRelationType.DISJOINT if dist > r1 + r2 else SpatialRelationType.TOUCHES, 1.0)

        if dist <= abs(r1 - r2):
            if r1 > r2:
                return SpatialRelation(id1, id2, SpatialRelationType.CONTAINS, 1.0)
            else:
                return SpatialRelation(id1, id2, SpatialRelationType.WITHIN, 1.0)

        overlap_area = self._circle_overlap_area(r1, r2, dist)
        iou = overlap_area / (math.pi * r1**2 + math.pi * r2**2 - overlap_area)
        return SpatialRelation(id1, id2, SpatialRelationType.OVERLAPS, iou)

    def _circle_overlap_area(self, r1: float, r2: float, d: float) -> float:
        if d >= r1 + r2:
            return 0.0
        if d <= abs(r1 - r2):
            return math.pi * min(r1, r2)**2

        r1_sq = r1**2
        r2_sq = r2**2
        d_sq = d**2

        alpha = math.acos((d_sq + r1_sq - r2_sq) / (2 * d * r1))
        beta = math.acos((d_sq + r2_sq - r1_sq) / (2 * d * r2))

        return r1_sq * alpha + r2_sq * beta - 0.5 * math.sqrt(
            (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
        )

    def _point_geometry_relation(self, point: Point2D, geometry: Any, id1: str, id2: str) -> SpatialRelation:
        if isinstance(geometry, BoundingBox2D):
            if geometry.contains(point):
                return SpatialRelation(id1, id2, SpatialRelationType.WITHIN, 1.0)
            else:
                return SpatialRelation(id1, id2, SpatialRelationType.DISJOINT, 1.0)
        elif isinstance(geometry, Polygon2D):
            if geometry.contains_point(point):
                return SpatialRelation(id1, id2, SpatialRelationType.WITHIN, 1.0)
            else:
                return SpatialRelation(id1, id2, SpatialRelationType.DISJOINT, 1.0)
        elif isinstance(geometry, Circle2D):
            if geometry.contains(point):
                return SpatialRelation(id1, id2, SpatialRelationType.WITHIN, 1.0)
            else:
                return SpatialRelation(id1, id2, SpatialRelationType.DISJOINT, 1.0)
        return SpatialRelation(id1, id2, SpatialRelationType.DISJOINT, 0.0)

    def _inverse_relation(self, rel: SpatialRelationType) -> SpatialRelationType:
        inverses = {
            SpatialRelationType.CONTAINS: SpatialRelationType.WITHIN,
            SpatialRelationType.WITHIN: SpatialRelationType.CONTAINS,
            SpatialRelationType.LEFT_OF: SpatialRelationType.RIGHT_OF,
            SpatialRelationType.RIGHT_OF: SpatialRelationType.LEFT_OF,
            SpatialRelationType.ABOVE: SpatialRelationType.BELOW,
            SpatialRelationType.BELOW: SpatialRelationType.ABOVE,
            SpatialRelationType.FRONT_OF: SpatialRelationType.BEHIND,
            SpatialRelationType.BEHIND: SpatialRelationType.FRONT_OF,
            SpatialRelationType.COVERS: SpatialRelationType.COVERED_BY,
            SpatialRelationType.COVERED_BY: SpatialRelationType.COVERS,
        }
        return inverses.get(rel, rel)

    def query_spatial(self, query: Dict[str, Any]) -> List[str]:
        relation_type = query.get("relation")
        target_id = query.get("target_id")
        radius = query.get("radius", float('inf'))

        if target_id not in self.objects:
            return []

        target = self.objects[target_id]
        results = []

        for obj_id, obj in self.objects.items():
            if obj_id == target_id:
                continue

            rel = self._compute_relation(obj, target, obj_id, target_id)
            if rel and rel.relation == relation_type:
                results.append(obj_id)

        return results

class TopologicalReasoner:
    def __init__(self):
        self.regions: Dict[str, Polygon2D] = {}
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        self.connectivity: Dict[str, Dict[str, float]] = defaultdict(dict)

    def add_region(self, region_id: str, polygon: Polygon2D):
        self.regions[region_id] = polygon
        self._update_adjacency(region_id)

    def _update_adjacency(self, region_id: str):
        polygon = self.regions[region_id]
        for other_id, other_poly in self.regions.items():
            if other_id == region_id:
                continue
            if self._polygons_adjacent(polygon, other_poly):
                self.adjacency[region_id].add(other_id)
                self.adjacency[other_id].add(region_id)

    def _polygons_adjacent(self, p1: Polygon2D, p2: Polygon2D) -> bool:
        bbox1 = p1.bounding_box()
        bbox2 = p2.bounding_box()
        if not bbox1.intersects(bbox2):
            return False

        for v1 in p1.vertices:
            if p2.contains_point(v1):
                return True
        for v2 in p2.vertices:
            if p1.contains_point(v2):
                return True
        return False

    def find_path(self, start_id: str, goal_id: str) -> Optional[List[str]]:
        if start_id not in self.regions or goal_id not in self.regions:
            return None

        queue = deque([(start_id, [start_id])])
        visited = {start_id}

        while queue:
            current, path = queue.popleft()
            if current == goal_id:
                return path

            for neighbor in self.adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def compute_connectivity(self):
        for r1_id, r1 in self.regions.items():
            for r2_id, r2 in self.regions.items():
                if r1_id != r2_id:
                    dist = self._region_distance(r1, r2)
                    self.connectivity[r1_id][r2_id] = dist

    def _region_distance(self, r1: Polygon2D, r2: Polygon2D) -> float:
        min_dist = float('inf')
        for v1 in r1.vertices:
            for v2 in r2.vertices:
                dist = v1.distance_to(v2)
                min_dist = min(min_dist, dist)
        return min_dist

class MetricReasoner:
    def __init__(self):
        self.landmarks: Dict[str, Point2D] = {}
        self.distances: Dict[Tuple[str, str], float] = {}

    def add_landmark(self, landmark_id: str, point: Point2D):
        self.landmarks[landmark_id] = point

    def add_distance(self, id1: str, id2: str, distance: float):
        self.distances[(id1, id2)] = distance
        self.distances[(id2, id1)] = distance

    def estimate_position(self, landmark_distances: Dict[str, float]) -> Optional[Point2D]:
        if len(landmark_distances) < 3:
            return None

        positions = []
        distances = []
        for lid, dist in landmark_distances.items():
            if lid in self.landmarks:
                positions.append(self.landmarks[lid].to_array())
                distances.append(dist)

        if len(positions) < 3:
            return None

        positions = np.array(positions)
        distances = np.array(distances)

        A = 2 * (positions[1:] - positions[0])
        b = distances[0]**2 - distances[1:]**2 + np.sum(positions[1:]**2 - positions[0]**2, axis=1)

        try:
            pos = np.linalg.lstsq(A, b, rcond=None)[0]
            return Point2D(pos[0], pos[1])
        except:
            return None

    def triangulate(self, known_points: Dict[str, Point2D],
                    angle_measurements: Dict[Tuple[str, str], float]) -> Dict[str, Point2D]:
        estimated = {}

        for (p1_id, p2_id), angle in angle_measurements.items():
            if p1_id in known_points and p2_id in known_points:
                p1 = known_points[p1_id]
                p2 = known_points[p2_id]

                mid = Point2D((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
                dx = p2.x - p1.x
                dy = p2.y - p1.y
                dist = math.hypot(dx, dy)

                if dist > 0:
                    perp_x = -dy / dist
                    perp_y = dx / dist
                    offset = dist / 2 * math.tan(math.pi / 2 - angle)
                    estimated_pos = Point2D(mid.x + perp_x * offset, mid.y + perp_y * offset)
                    estimated[f"tri_{p1_id}_{p2_id}"] = estimated_pos

        return estimated

class QualitativeSpatialReasoner:
    def __init__(self):
        self.calculi = {
            "RCC8": self._rcc8_reasoning,
            "CDC": self._cdc_reasoning,
            "OPRA": self._opra_reasoning,
        }

    def reason(self, calculus: str, objects: Dict[str, Any]) -> List[SpatialRelation]:
        if calculus in self.calculi:
            return self.calculi[calculus](objects)
        return []

    def _rcc8_reasoning(self, objects: Dict[str, Any]) -> List[SpatialRelation]:
        relations = []
        obj_ids = list(objects.keys())

        for i, id1 in enumerate(obj_ids):
            for id2 in obj_ids[i+1:]:
                obj1 = objects[id1]
                obj2 = objects[id2]

                if hasattr(obj1, 'centroid') and hasattr(obj2, 'centroid'):
                    c1 = obj1.centroid
                    c2 = obj2.centroid
                    dist = c1.distance_to(c2)

                    area1 = obj1.area if hasattr(obj1, 'area') else 0
                    area2 = obj2.area if hasattr(obj2, 'area') else 0

                    if dist < 1e-6:
                        rel = SpatialRelationType.EQUALS
                    elif hasattr(obj1, 'contains_point') and hasattr(obj2, 'contains_point'):
                        if obj1.contains_point(c2) and obj2.contains_point(c1):
                            rel = SpatialRelationType.OVERLAPS
                        elif obj1.contains_point(c2):
                            rel = SpatialRelationType.CONTAINS
                        elif obj2.contains_point(c1):
                            rel = SpatialRelationType.WITHIN
                        else:
                            rel = SpatialRelationType.DISJOINT
                    else:
                        rel = SpatialRelationType.DISJOINT

                    relations.append(SpatialRelation(id1, id2, rel, 1.0))

        return relations

    def _cdc_reasoning(self, objects: Dict[str, Any]) -> List[SpatialRelation]:
        relations = []
        obj_ids = list(objects.keys())

        for i, id1 in enumerate(obj_ids):
            for id2 in obj_ids[i+1:]:
                obj1 = objects[id1]
                obj2 = objects[id2]

                if hasattr(obj1, 'centroid') and hasattr(obj2, 'centroid'):
                    c1 = obj1.centroid
                    c2 = obj2.centroid

                    dx = c2.x - c1.x
                    dy = c2.y - c1.y

                    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                        rel = SpatialRelationType.EQUALS
                    elif abs(dx) > abs(dy):
                        rel = SpatialRelationType.RIGHT_OF if dx > 0 else SpatialRelationType.LEFT_OF
                    else:
                        rel = SpatialRelationType.ABOVE if dy > 0 else SpatialRelationType.BELOW

                    relations.append(SpatialRelation(id1, id2, rel, 1.0))

        return relations

    def _opra_reasoning(self, objects: Dict[str, Any]) -> List[SpatialRelation]:
        return self._cdc_reasoning(objects)

class SpatialPlanner:
    def __init__(self, space_bounds: BoundingBox2D, resolution: float = 1.0):
        self.space_bounds = space_bounds
        self.resolution = resolution
        self.obstacles: List[Polygon2D] = []
        self.grid: Optional[np.ndarray] = None
        self._build_grid()

    def add_obstacle(self, polygon: Polygon2D):
        self.obstacles.append(polygon)
        self._build_grid()

    def _build_grid(self):
        width = int(self.space_bounds.width / self.resolution)
        height = int(self.space_bounds.height / self.resolution)
        self.grid = np.zeros((height, width), dtype=bool)

        for y in range(height):
            for x in range(width):
                wx = self.space_bounds.min_x + (x + 0.5) * self.resolution
                wy = self.space_bounds.min_y + (y + 0.5) * self.resolution
                point = Point2D(wx, wy)

                for obs in self.obstacles:
                    if obs.contains_point(point):
                        self.grid[y, x] = True
                        break

    def plan_path(self, start: Point2D, goal: Point2D) -> Optional[List[Point2D]]:
        if self.grid is None:
            return [start, goal]

        sx = int((start.x - self.space_bounds.min_x) / self.resolution)
        sy = int((start.y - self.space_bounds.min_y) / self.resolution)
        gx = int((goal.x - self.space_bounds.min_x) / self.resolution)
        gy = int((goal.y - self.space_bounds.min_y) / self.resolution)

        if not (0 <= sx < self.grid.shape[1] and 0 <= sy < self.grid.shape[0]):
            return None
        if not (0 <= gx < self.grid.shape[1] and 0 <= gy < self.grid.shape[0]):
            return None
        if self.grid[sy, sx] or self.grid[gy, gx]:
            return None

        return self._astar((sy, sx), (gy, gx))

    def _astar(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Point2D]]:
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    y, x = current
                    path.append(Point2D(
                        self.space_bounds.min_x + (x + 0.5) * self.resolution,
                        self.space_bounds.min_y + (y + 0.5) * self.resolution
                    ))
                    current = came_from[current]
                y, x = start
                path.append(Point2D(
                    self.space_bounds.min_x + (x + 0.5) * self.resolution,
                    self.space_bounds.min_y + (y + 0.5) * self.resolution
                ))
                return path[::-1]

            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                neighbor = (current[0] + dy, current[1] + dx)
                ny, nx = neighbor

                if not (0 <= nx < self.grid.shape[1] and 0 <= ny < self.grid.shape[0]):
                    continue
                if self.grid[ny, nx]:
                    continue

                cost = math.hypot(dx, dy) * self.resolution
                tentative_g = g_score[current] + cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1]) * self.resolution

class SpatialQueryEngine:
    def __init__(self):
        self.index = SpatialIndex(2)
        self.objects: Dict[str, Any] = {}

    def add_object(self, obj_id: str, geometry: Any):
        self.objects[obj_id] = geometry
        if isinstance(geometry, Point2D):
            self.index.insert(obj_id, geometry.to_array())
        elif hasattr(geometry, 'center'):
            center = geometry.center
            if isinstance(center, Point2D):
                self.index.insert(obj_id, center.to_array())

    def range_query(self, center: Point2D, radius: float) -> List[Tuple[str, float]]:
        return self.index.query_radius(center.to_array(), radius)

    def knn_query(self, center: Point2D, k: int) -> List[Tuple[str, float]]:
        return self.index.query_knn(center.to_array(), k)

    def bbox_query(self, bbox: BoundingBox2D) -> List[str]:
        return self.index.query_bbox(
            np.array([bbox.min_x, bbox.min_y]),
            np.array([bbox.max_x, bbox.max_y])
        )

    def spatial_join(self, other: 'SpatialQueryEngine', relation: SpatialRelationType,
                     max_distance: float = float('inf')) -> List[Tuple[str, str]]:
        results = []
        for id1, obj1 in self.objects.items():
            for id2, obj2 in other.objects.items():
                rel = self._compute_pair_relation(obj1, obj2, id1, id2)
                if rel and rel.relation == relation:
                    results.append((id1, id2))
        return results

    def _compute_pair_relation(self, obj1: Any, obj2: Any, id1: str, id2: str) -> Optional[SpatialRelation]:
        if isinstance(obj1, Point2D) and isinstance(obj2, Point2D):
            dist = obj1.distance_to(obj2)
            if dist <= max_distance:
                return SpatialRelation(id1, id2, SpatialRelationType.NEAR if dist < 10 else SpatialRelationType.FAR)
        return None

def create_spatial_reasoner() -> SpatialReasoner:
    return SpatialReasoner()

def create_topological_reasoner() -> TopologicalReasoner:
    return TopologicalReasoner()

def create_metric_reasoner() -> MetricReasoner:
    return MetricReasoner()

def create_spatial_planner(bounds: BoundingBox2D, resolution: float = 1.0) -> SpatialPlanner:
    return SpatialPlanner(bounds, resolution)