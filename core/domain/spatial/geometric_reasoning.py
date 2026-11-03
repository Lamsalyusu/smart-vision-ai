from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict
import itertools


class GeometricPrimitive(Enum):
    POINT = "point"
    LINE = "line"
    PLANE = "plane"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    CONE = "cone"
    BOX = "box"
    POLYGON = "polygon"
    POLYHEDRON = "polyhedron"
    CONVEX_HULL = "convex_hull"


class SpatialRelation(Enum):
    INTERSECTS = "intersects"
    CONTAINS = "contains"
    CONTAINED_BY = "contained_by"
    DISJOINT = "disjoint"
    TOUCHES = "touches"
    OVERLAPS = "overlaps"
    EQUALS = "equals"
    COVERS = "covers"
    COVERED_BY = "covered_by"
    CROSSES = "crosses"
    WITHIN = "within"


class GeometricRelation(Enum):
    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"
    COLLINEAR = "collinear"
    COPLANAR = "coplanar"
    TANGENT = "tangent"
    SECANT = "secant"
    SKEW = "skew"


@dataclass(frozen=True, slots=True)
class Point3D:
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @staticmethod
    def from_array(arr: np.ndarray) -> 'Point3D':
        return Point3D(arr[0], arr[1], arr[2])

    def distance_to(self, other: 'Point3D') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

    def __add__(self, other: 'Point3D') -> 'Point3D':
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Point3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Point3D':
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)


@dataclass(frozen=True, slots=True)
class Vector3D:
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @staticmethod
    def from_array(arr: np.ndarray) -> 'Vector3D':
        return Vector3D(arr[0], arr[1], arr[2])

    def norm(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> 'Vector3D':
        n = self.norm()
        return Vector3D(self.x/n, self.y/n, self.z/n) if n > 1e-10 else Vector3D()

    def dot(self, other: 'Vector3D') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)


@dataclass(frozen=True, slots=True)
class Line3D:
    point: Point3D
    direction: Vector3D

    def closest_point(self, point: Point3D) -> Point3D:
        v = Vector3D(point.x - self.point.x, point.y - self.point.y, point.z - self.point.z)
        t = v.dot(self.direction) / (self.direction.norm()**2 + 1e-10)
        return Point3D(
            self.point.x + self.direction.x * t,
            self.point.y + self.direction.y * t,
            self.point.z + self.direction.z * t
        )

    def distance_to_point(self, point: Point3D) -> float:
        closest = self.closest_point(point)
        return point.distance_to(closest)

    def intersection_with_plane(self, plane: 'Plane3D') -> Optional[Point3D]:
        denom = plane.normal.dot(self.direction)
        if abs(denom) < 1e-10:
            return None

        t = (plane.distance - plane.normal.dot(Vector3D(self.point.x, self.point.y, self.point.z))) / denom
        return Point3D(
            self.point.x + self.direction.x * t,
            self.point.y + self.direction.y * t,
            self.point.z + self.direction.z * t
        )


@dataclass(frozen=True, slots=True)
class Plane3D:
    normal: Vector3D
    distance: float

    @staticmethod
    def from_points(p1: Point3D, p2: Point3D, p3: Point3D) -> 'Plane3D':
        v1 = Vector3D(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)
        v2 = Vector3D(p3.x - p1.x, p3.y - p1.y, p3.z - p1.z)
        normal = v1.cross(v2).normalized()
        distance = normal.dot(Vector3D(p1.x, p1.y, p1.z))
        return Plane3D(normal, distance)

    def distance_to_point(self, point: Point3D) -> float:
        return abs(self.normal.dot(Vector3D(point.x, point.y, point.z)) - self.distance)

    def intersection_with_line(self, line: Line3D) -> Optional[Point3D]:
        return line.intersection_with_plane(self)

    def intersection_with_plane(self, other: 'Plane3D') -> Optional[Line3D]:
        cross = self.normal.cross(other.normal)
        if cross.norm() < 1e-10:
            return None

        dir_vec = cross.normalized()
        d1 = self.distance
        d2 = other.distance
        n1 = self.normal.to_array()
        n2 = other.normal.to_array()

        n1n2 = np.dot(self.normal.to_array(), other.normal.to_array())
        det = 1 - n1n2**2

        if abs(det) < 1e-10:
            return None

        c1 = (self.distance - other.distance * n1n2) / det
        c2 = (other.distance - self.distance * n1n2) / det

        point_vec = Vector3D(c1 * n1[0] + c2 * n2[0], c1 * n1[1] + c2 * n2[1], c1 * n1[2] + c2 * n2[2])
        return Line3D(Point3D(point_vec.x, point_vec.y, point_vec.z), Vector3D(dir_vec.x, dir_vec.y, dir_vec.z))

    def project_point(self, point: Point3D) -> Point3D:
        t = self.distance - self.normal.dot(Vector3D(point.x, point.y, point.z))
        return Point3D(
            point.x + self.normal.x * t,
            point.y + self.normal.y * t,
            point.z + self.normal.z * t
        )


@dataclass(frozen=True, slots=True)
class Sphere3D:
    center: Point3D
    radius: float

    def contains_point(self, point: Point3D) -> bool:
        return self.center.distance_to(point) <= self.radius + 1e-10

    def distance_to_point(self, point: Point3D) -> float:
        return max(0.0, self.center.distance_to(point) - self.radius)

    def intersects_sphere(self, other: 'Sphere3D') -> bool:
        return self.center.distance_to(other.center) <= self.radius + other.radius + 1e-10

    def intersection_with_line(self, line: Line3D) -> List[Point3D]:
        oc = Vector3D(line.point.x - self.center.x, line.point.y - self.center.y, line.point.z - self.center.z)
        a = line.direction.norm()**2
        b = 2 * line.direction.dot(Vector3D(line.point.x - self.center.x, line.point.y - self.center.y, line.point.z - self.center.z))
        c = (line.point.x - self.center.x)**2 + (line.point.y - self.center.y)**2 + (line.point.z - self.center.z)**2 - self.radius**2

        disc = b**2 - 4*a*c
        if disc < 0:
            return []

        t1 = (-b + math.sqrt(max(0, disc))) / (2*a)
        t2 = (-b - math.sqrt(max(0, disc))) / (2*a)

        points = []
        for t in [t1, t2]:
            points.append(Point3D(
                line.point.x + line.direction.x * t,
                line.point.y + line.direction.y * t,
                line.point.z + line.direction.z * t
            ))
        return points


@dataclass(frozen=True, slots=True)
class Box3D:
    min_corner: Point3D
    max_corner: Point3D

    @property
    def center(self) -> Point3D:
        return Point3D(
            (self.min_corner.x + self.max_corner.x) / 2,
            (self.min_corner.y + self.max_corner.y) / 2,
            (self.min_corner.z + self.max_corner.z) / 2
        )

    @property
    def dimensions(self) -> Vector3D:
        return Vector3D(
            self.max_corner.x - self.min_corner.x,
            self.max_corner.y - self.min_corner.y,
            self.max_corner.z - self.min_corner.z
        )

    def contains_point(self, point: Point3D) -> bool:
        return (self.min_corner.x <= point.x <= self.max_corner.x and
                self.min_corner.y <= point.y <= self.max_corner.y and
                self.min_corner.z <= point.z <= self.max_corner.z)

    def intersects_box(self, other: 'Box3D') -> bool:
        return (self.min_corner.x <= other.max_corner.x and self.max_corner.x >= other.min_corner.x and
                self.min_corner.y <= other.max_corner.y and self.max_corner.y >= other.min_corner.y and
                self.min_corner.z <= other.max_corner.z and self.max_corner.z >= other.min_corner.z)

    def distance_to_point(self, point: Point3D) -> float:
        dx = max(self.min_corner.x - point.x, 0, point.x - self.max_corner.x)
        dy = max(self.min_corner.y - point.y, 0, point.y - self.max_corner.y)
        dz = max(self.min_corner.z - point.z, 0, point.z - self.max_corner.z)
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def intersection_with_box(self, other: 'Box3D') -> Optional['Box3D']:
        min_c = Point3D(
            max(self.min_corner.x, other.min_corner.x),
            max(self.min_corner.y, other.min_corner.y),
            max(self.min_corner.z, other.min_corner.z)
        )
        max_c = Point3D(
            min(self.max_corner.x, other.max_corner.x),
            min(self.max_corner.y, other.max_corner.y),
            min(self.max_corner.z, other.max_corner.z)
        )

        if min_c.x <= max_c.x and min_c.y <= max_c.y and min_c.z <= max_c.z:
            return Box3D(min_c, max_c)
        return None


class GeometricReasoner:
    def __init__(self):
        pass

    def check_relation(self, obj1: Any, obj2: Any, relation: SpatialRelation) -> bool:
        if isinstance(obj1, Sphere3D) and isinstance(obj2, Sphere3D):
            return self._sphere_sphere_relation(obj1, obj2, relation)
        elif isinstance(obj1, Box3D) and isinstance(obj2, Box3D):
            return self._box_box_relation(obj1, obj2, relation)
        elif isinstance(obj1, Sphere3D) and isinstance(obj2, Box3D):
            return self._sphere_box_relation(obj1, obj2, relation)
        elif isinstance(obj1, Box3D) and isinstance(obj2, Sphere3D):
            return self._sphere_box_relation(obj2, obj1, relation)
        elif isinstance(obj1, Plane3D) and isinstance(obj2, Point3D):
            return self._plane_point_relation(obj1, obj2, relation)
        elif isinstance(obj1, Point3D) and isinstance(obj2, Plane3D):
            return self._plane_point_relation(obj2, obj1, relation)
        return False

    def _sphere_sphere_relation(self, s1: Sphere3D, s2: Sphere3D, relation: SpatialRelation) -> bool:
        dist = s1.center.distance_to(s2.center)
        sum_r = s1.radius + s2.radius
        diff_r = abs(s1.radius - s2.radius)

        if relation == SpatialRelation.INTERSECTS:
            return s1.intersects_sphere(s2)
        elif relation == SpatialRelation.DISJOINT:
            return dist > s1.radius + s2.radius + 1e-10
        elif relation == SpatialRelation.CONTAINS:
            return s1.radius >= s2.radius and s1.center.distance_to(s2.center) + s2.radius <= s1.radius + 1e-10
        elif relation == SpatialRelation.CONTAINED_BY:
            return s2.radius >= s1.radius and s2.center.distance_to(s1.center) + s1.radius <= s2.radius + 1e-10
        elif relation == SpatialRelation.TOUCHES:
            return abs(s1.center.distance_to(s2.center) - (s1.radius + s2.radius)) < 1e-6
        return False

    def _box_box_relation(self, b1: Box3D, b2: Box3D, relation: SpatialRelation) -> bool:
        if relation == SpatialRelation.INTERSECTS:
            return b1.intersects_box(b2)
        elif relation == SpatialRelation.DISJOINT:
            return not b1.intersects_box(b2)
        elif relation == SpatialRelation.CONTAINS:
            return (b1.min_corner.x <= b2.min_corner.x and b1.max_corner.x >= b2.max_corner.x and
                    b1.min_corner.y <= b2.min_corner.y and b1.max_corner.y >= b2.max_corner.y and
                    b1.min_corner.z <= b2.min_corner.z and b1.max_corner.z >= b2.max_corner.z)
        elif relation == SpatialRelation.CONTAINED_BY:
            return (b2.min_corner.x <= b1.min_corner.x and b2.max_corner.x >= b1.max_corner.x and
                    b2.min_corner.y <= b1.min_corner.y and b2.max_corner.y >= b1.max_corner.y and
                    b2.min_corner.z <= b1.min_corner.z and b2.max_corner.z >= b1.max_corner.z)
        elif relation == SpatialRelation.TOUCHES:
            return b1.intersects_box(b2) and not (b1.min_corner.x < b2.max_corner.x and b1.max_corner.x > b2.min_corner.x and
                                                   b1.min_corner.y < b2.max_corner.y and b1.max_corner.y > b2.min_corner.y and
                                                   b1.min_corner.z < b2.max_corner.z and b1.max_corner.z > b2.min_corner.z)
        return False

    def _sphere_box_relation(self, sphere: Sphere3D, box: Box3D, relation: SpatialRelation) -> bool:
        closest = Point3D(
            max(box.min_corner.x, min(sphere.center.x, box.max_corner.x)),
            max(box.min_corner.y, min(sphere.center.y, box.max_corner.y)),
            max(box.min_corner.z, min(sphere.center.z, box.max_corner.z))
        )
        dist = sphere.center.distance_to(closest)

        if relation == SpatialRelation.INTERSECTS:
            return dist <= sphere.radius + 1e-10
        elif relation == SpatialRelation.DISJOINT:
            return dist > sphere.radius + 1e-10
        elif relation == SpatialRelation.CONTAINS:
            return False
        elif relation == SpatialRelation.CONTAINED_BY:
            return (sphere.center.x - sphere.radius >= box.min_corner.x and
                    sphere.center.x + sphere.radius <= box.max_corner.x and
                    sphere.center.y - sphere.radius >= box.min_corner.y and
                    sphere.center.y + sphere.radius <= box.max_corner.y and
                    sphere.center.z - sphere.radius >= box.min_corner.z and
                    sphere.center.z + sphere.radius <= box.max_corner.z)
        elif relation == SpatialRelation.TOUCHES:
            return abs(dist - sphere.radius) < 1e-6
        return False

    def _plane_point_relation(self, plane: Plane3D, point: Point3D, relation: SpatialRelation) -> bool:
        dist = plane.distance_to_point(point)

        if relation == SpatialRelation.INTERSECTS:
            return dist < 1e-10
        elif relation == SpatialRelation.DISJOINT:
            return dist > 1e-10
        elif relation == SpatialRelation.CONTAINS:
            return dist < 1e-10
        elif relation == SpatialRelation.CONTAINED_BY:
            return dist < 1e-10
        elif relation == SpatialRelation.TOUCHES:
            return dist < 1e-10
        return False

    def compute_geometric_relation(self, obj1: Any, obj2: Any) -> List[GeometricRelation]:
        relations = []

        if isinstance(obj1, Line3D) and isinstance(obj2, Line3D):
            if self._lines_parallel(obj1, obj2):
                relations.append(GeometricRelation.PARALLEL)
            elif self._lines_perpendicular(obj1, obj2):
                relations.append(GeometricRelation.PERPENDICULAR)
            elif self._lines_intersect(obj1, obj2):
                relations.append(GeometricRelation.SECANT)
            else:
                relations.append(GeometricRelation.SKEW)

        elif isinstance(obj1, Line3D) and isinstance(obj2, Plane3D):
            if self._line_parallel_plane(obj1, obj2):
                relations.append(GeometricRelation.PARALLEL)
            elif self._line_perpendicular_plane(obj1, obj2):
                relations.append(GeometricRelation.PERPENDICULAR)

        elif isinstance(obj1, Plane3D) and isinstance(obj2, Plane3D):
            if self._planes_parallel(obj1, obj2):
                relations.append(GeometricRelation.PARALLEL)
            elif self._planes_perpendicular(obj1, obj2):
                relations.append(GeometricRelation.PERPENDICULAR)

        return relations

    def _lines_parallel(self, l1: Line3D, l2: Line3D) -> bool:
        cross = l1.direction.cross(l2.direction)
        return cross.norm() < 1e-10

    def _lines_perpendicular(self, l1: Line3D, l2: Line3D) -> bool:
        return abs(l1.direction.dot(l2.direction)) < 1e-10

    def _lines_intersect(self, l1: Line3D, l2: Line3D) -> bool:
        cross = l1.direction.cross(l2.direction)
        if cross.norm() < 1e-10:
            return False

        diff = Vector3D(l2.point.x - l1.point.x, l2.point.y - l1.point.y, l2.point.z - l1.point.z)
        return abs(diff.dot(cross)) < 1e-10

    def _line_parallel_plane(self, line: Line3D, plane: Plane3D) -> bool:
        return abs(plane.normal.dot(line.direction)) < 1e-10

    def _line_perpendicular_plane(self, line: Line3D, plane: Plane3D) -> bool:
        return abs(abs(plane.normal.dot(line.direction)) - 1.0) < 1e-10

    def _planes_parallel(self, p1: Plane3D, p2: Plane3D) -> bool:
        return p1.normal.cross(p2.normal).norm() < 1e-10

    def _planes_perpendicular(self, p1: Plane3D, p2: Plane3D) -> bool:
        return abs(p1.normal.dot(p2.normal)) < 1e-10


class CollisionDetector:
    def __init__(self):
        self.reasoner = GeometricReasoner()

    def check_collision(self, obj1: Any, obj2: Any) -> bool:
        return self.reasoner.check_relation(obj1, obj2, SpatialRelation.INTERSECTS)

    def check_multiple_collisions(self, objects: List[Any]) -> List[Tuple[int, int]]:
        collisions = []
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                if self.check_collision(objects[i], objects[j]):
                    collisions.append((i, j))
        return collisions

    def compute_penetration_depth(self, obj1: Any, obj2: Any) -> float:
        if isinstance(obj1, Sphere3D) and isinstance(obj2, Sphere3D):
            dist = obj1.center.distance_to(obj2.center)
            return max(0.0, obj1.radius + obj2.radius - obj1.center.distance_to(obj2.center))

        elif isinstance(obj1, Box3D) and isinstance(obj2, Box3D):
            intersection = obj1.intersection_with_box(obj2)
            if intersection:
                dims = intersection.dimensions
                return min(dims.x, dims.y, dims.z)
            return 0.0

        return 0.0


class GeometricOptimizer:
    @staticmethod
    def fit_plane_to_points(points: List[Point3D]) -> Plane3D:
        if len(points) < 3:
            return Plane3D(Vector3D(0, 0, 1), 0)

        points_arr = np.array([[p.x, p.y, p.z] for p in points])
        centroid = np.mean(points_arr, axis=0)
        centered = points_arr - centroid

        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        normal = Vector3D(*Vt[-1])
        distance = normal.dot(Vector3D(*centroid))

        return Plane3D(normal, distance)

    @staticmethod
    def fit_sphere_to_points(points: List[Point3D]) -> Sphere3D:
        if len(points) < 4:
            return Sphere3D(Point3D(0, 0, 0), 0)

        A = []
        b = []
        for p in points:
            A.append([2*p.x, 2*p.y, 2*p.z, 1])
            b.append(p.x**2 + p.y**2 + p.z**2)

        A = np.array(A)
        b = np.array(b)
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        center = Point3D(x[0], x[1], x[2])
        radius = math.sqrt(max(0, x[3] + x[0]**2 + x[1]**2 + x[2]**2))

        return Sphere3D(center, radius)

    @staticmethod
    def fit_cylinder_to_points(points: List[Point3D], axis: Vector3D) -> Tuple[Point3D, float, float]:
        pass


class GeometricVisualizer:
    @staticmethod
    def draw_primitives(
        primitives: List[Any],
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if not primitives:
            return vis

        positions = []
        for p in primitives:
            if isinstance(p, Point3D):
                positions.append([p.x, p.y])
            elif isinstance(p, Sphere3D):
                positions.append([p.center.x, p.center.y])
            elif isinstance(p, Box3D):
                positions.append([p.center.x, p.center.y])

        if not positions:
            return vis

        pos_array = np.array(positions)
        min_coords = np.min(pos_array, axis=0)
        max_coords = np.max(pos_array, axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1

        scale = min(image_size[0], image_size[1]) * 0.8 / max(range_coords)

        for p in primitives:
            if isinstance(p, Point3D):
                x = int((p.x - 0) * scale + image_size[0] * 0.5)
                y = int(image_size[1] - (p.y - 0) * scale - image_size[1] * 0.5)
                cv2.circle(vis, (x, y), 3, (255, 255, 255), -1)

            elif isinstance(p, Sphere3D):
                x = int((p.center.x - 0) * scale + image_size[0] * 0.5)
                y = int(image_size[1] - (p.center.y - 0) * scale - image_size[1] * 0.5)
                r = int(p.radius * 100)
                cv2.circle(vis, (x, y), r, (0, 255, 0), 1)

            elif isinstance(p, Box3D):
                x = int((p.center.x - 0) * scale + image_size[0] * 0.5)
                y = int(image_size[1] - (p.center.y - 0) * scale - image_size[1] * 0.5)
                w = int(p.dimensions.x * 50)
                h = int(p.dimensions.y * 50)
                cv2.rectangle(vis, (x - w//2, y - h//2), (x + w//2, y + h//2), (255, 0, 0), 1)

        return vis


class GeometricMetrics:
    @staticmethod
    def volume(obj: Any) -> float:
        if isinstance(obj, Sphere3D):
            return 4.0/3.0 * math.pi * obj.radius**3
        elif isinstance(obj, Box3D):
            return obj.dimensions.x * obj.dimensions.y * obj.dimensions.z
        return 0.0

    @staticmethod
    def surface_area(obj: Any) -> float:
        if isinstance(obj, Sphere3D):
            return 4 * math.pi * obj.radius**2
        elif isinstance(obj, Box3D):
            dims = obj.dimensions
            return 2 * (dims.x * dims.y + dims.y * dims.z + dims.z * dims.x)
        return 0.0

    @staticmethod
    def moment_of_inertia(obj: Any, axis: Vector3D) -> float:
        if isinstance(obj, Sphere3D):
            return 2.0/5.0 * obj.radius**2
        elif isinstance(obj, Box3D):
            dims = obj.dimensions
            axis = axis.normalized()
            return (obj.dimensions.y**2 + obj.dimensions.z**2) * axis.x**2 + \
                   (obj.dimensions.x**2 + obj.dimensions.z**2) * axis.y**2 + \
                   (obj.dimensions.x**2 + obj.dimensions.y**2) * axis.z**2
        return 0.0

    @staticmethod
    def distance(obj1: Any, obj2: Any) -> float:
        if isinstance(obj1, Point3D) and isinstance(obj2, Point3D):
            return obj1.distance_to(obj2)
        elif isinstance(obj1, Sphere3D) and isinstance(obj2, Sphere3D):
            return max(0.0, obj1.center.distance_to(obj2.center) - obj1.radius - obj2.radius)
        elif isinstance(obj1, Box3D) and isinstance(obj2, Box3D):
            inter = obj1.intersection_with_box(obj2)
            if inter:
                return 0.0
            return min(
                abs(obj1.min_corner.x - obj2.max_corner.x),
                abs(obj1.max_corner.x - obj2.min_corner.x),
                abs(obj1.min_corner.y - obj2.max_corner.y),
                abs(obj1.max_corner.y - obj2.min_corner.y),
                abs(obj1.min_corner.z - obj2.max_corner.z),
                abs(obj1.max_corner.z - obj2.min_corner.z)
            )
        return 0.0


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
