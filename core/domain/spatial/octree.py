from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass(frozen=True, slots=True)
class SpatialAABB:
    """Axis-Aligned Bounding Box in 3D Euclidean space."""
    min_point: Tuple[float, float, float]
    max_point: Tuple[float, float, float]

    def contains(self, point: Tuple[float, float, float]) -> bool:
        return (
            self.min_point[0] <= point[0] <= self.max_point[0]
            and self.min_point[1] <= point[1] <= self.max_point[1]
            and self.min_point[2] <= point[2] <= self.max_point[2]
        )

    def intersects(self, other: SpatialAABB) -> bool:
        return (
            self.min_point[0] <= other.max_point[0] and self.max_point[0] >= other.min_point[0]
            and self.min_point[1] <= other.max_point[1] and self.max_point[1] >= other.min_point[1]
            and self.min_point[2] <= other.max_point[2] and self.max_point[2] >= other.min_point[2]
        )


@dataclass
class OctreePoint:
    position: Tuple[float, float, float]
    data: int  # index or entity ID


class OctreeNode:
    def __init__(self, boundary: SpatialAABB, capacity: int = 8, depth: int = 0, max_depth: int = 8) -> None:
        self.boundary = boundary
        self.capacity = capacity
        self.depth = depth
        self.max_depth = max_depth
        self.points: List[OctreePoint] = []
        self.children: Optional[List[OctreeNode]] = None

    def subdivide(self) -> None:
        x_min, y_min, z_min = self.boundary.min_point
        x_max, y_max, z_max = self.boundary.max_point
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        cz = (z_min + z_max) / 2.0

        self.children = [
            OctreeNode(SpatialAABB((x_min, y_min, z_min), (cx, cy, cz)), self.capacity, self.depth + 1, self.max_depth),
            OctreeNode(SpatialAABB((cx, y_min, z_min), (x_max, cy, cz)), self.capacity, self.depth + 1, self.max_depth),
            OctreeNode(SpatialAABB((x_min, cy, z_min), (cx, y_max, cz)), self.capacity, self.depth + 1, self.max_depth),
            OctreeNode(SpatialAABB((cx, cy, z_min), (x_max, y_max, cz)), self.capacity, self.depth + 1, self.max_depth),
            OctreeNode(SpatialAABB((x_min, y_min, cz), (cx, cy, z_max)), self.capacity, self.depth + 1, self.max_depth),
            OctreeNode(SpatialAABB((cx, y_min, cz), (x_max, cy, z_max)), self.capacity, self.depth + 1, self.max_depth),
            OctreeNode(SpatialAABB((x_min, cy, cz), (cx, y_max, z_max)), self.capacity, self.depth + 1, self.max_depth),
            OctreeNode(SpatialAABB((cx, cy, cz), (x_max, y_max, z_max)), self.capacity, self.depth + 1, self.max_depth),
        ]

        # Distribute existing points to children
        for pt in self.points:
            for child in self.children:
                if child.boundary.contains(pt.position):
                    child.insert(pt)
                    break
        self.points.clear()

    def insert(self, pt: OctreePoint) -> bool:
        if not self.boundary.contains(pt.position):
            return False

        if self.children is not None:
            for child in self.children:
                if child.insert(pt):
                    return True
            return False

        self.points.append(pt)
        if len(self.points) > self.capacity and self.depth < self.max_depth:
            self.subdivide()

        return True

    def query_range(self, aabb: SpatialAABB, results: List[OctreePoint]) -> None:
        if not self.boundary.intersects(aabb):
            return

        for pt in self.points:
            if aabb.contains(pt.position):
                results.append(pt)

        if self.children is not None:
            for child in self.children:
                child.query_range(aabb, results)

    def query_radius(self, center: Tuple[float, float, float], radius: float, results: List[OctreePoint]) -> None:
        aabb = SpatialAABB(
            (center[0] - radius, center[1] - radius, center[2] - radius),
            (center[0] + radius, center[1] + radius, center[2] + radius),
        )
        candidates: List[OctreePoint] = []
        self.query_range(aabb, candidates)

        r2 = radius * radius
        for pt in candidates:
            dx = pt.position[0] - center[0]
            dy = pt.position[1] - center[1]
            dz = pt.position[2] - center[2]
            if dx * dx + dy * dy + dz * dz <= r2:
                results.append(pt)


class SpatialOctreeIndex:
    """High-performance 3D Octree Index for spatial point clouds and target querying."""

    def __init__(self, bounds: SpatialAABB, capacity: int = 16, max_depth: int = 8) -> None:
        self.root = OctreeNode(bounds, capacity=capacity, max_depth=max_depth)
        self.count = 0

    def insert(self, x: float, y: float, z: float, entity_id: int) -> bool:
        success = self.root.insert(OctreePoint((x, y, z), entity_id))
        if success:
            self.count += 1
        return success

    def radius_search(self, center: Tuple[float, float, float], radius: float) -> List[Tuple[int, float]]:
        """Find entity IDs within radius distance, returning list of (entity_id, distance)."""
        pts: List[OctreePoint] = []
        self.root.query_radius(center, radius, pts)

        results = []
        for p in pts:
            dist = math.dist(center, p.position)
            results.append((p.data, dist))
        return sorted(results, key=lambda x: x[1])
