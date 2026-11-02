from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math
import time
import heapq


@dataclass(frozen=True, slots=True)
class Voxel:
    x: int
    y: int
    z: int

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __eq__(self, other):
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def neighbors(self, connectivity: int = 6) -> List["Voxel"]:
        if connectivity == 6:
            dirs = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        elif connectivity == 18:
            dirs = [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)
                    if not (dx==0 and dy==0 and dz==0) and (abs(dx)+abs(dy)+abs(dz) <= 2)]
        else:
            dirs = [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)
                    if not (dx==0 and dy==0 and dz==0)]
        return [Voxel(self.x+dx, self.y+dy, self.z+dz) for dx,dy,dz in dirs]


@dataclass(frozen=True, slots=True)
class VoxelGrid:
    voxels: Dict[Voxel, float]
    resolution: float
    origin: Tuple[float, float, float]

    def get(self, voxel: Voxel) -> float:
        return self.voxels.get(voxel, 0.0)

    def set(self, voxel: Voxel, value: float):
        self.voxels[voxel] = value


class OctreeNode:
    def __init__(self, min_bounds: np.ndarray, max_bounds: np.ndarray, max_depth: int, depth: int = 0):
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.max_depth = max_depth
        self.depth = depth
        self.children: List[Optional["OctreeNode"]] = [None] * 8
        self.value: float = 0.0
        self.is_leaf = True
        self.point_count = 0

    def get_center(self) -> np.ndarray:
        return (self.min_bounds + self.max_bounds) / 2

    def get_size(self) -> np.ndarray:
        return self.max_bounds - self.min_bounds

    def contains(self, point: np.ndarray) -> bool:
        return np.all(point >= self.min_bounds) and np.all(point < self.max_bounds)

    def get_child_index(self, point: np.ndarray) -> int:
        center = self.get_center()
        index = 0
        if point[0] >= center[0]:
            index |= 1
        if point[1] >= center[1]:
            index |= 2
        if point[2] >= center[2]:
            index |= 4
        return index

    def create_child(self, index: int) -> "OctreeNode":
        center = self.get_center()
        size = self.get_size() / 2
        min_bounds = self.min_bounds.copy()
        max_bounds = self.max_bounds.copy()

        if index & 1:
            min_bounds[0] = center[0]
        else:
            max_bounds[0] = center[0]
        if index & 2:
            min_bounds[1] = center[1]
        else:
            max_bounds[1] = center[1]
        if index & 4:
            min_bounds[2] = center[2]
        else:
            max_bounds[2] = center[2]

        return OctreeNode(min_bounds, max_bounds, self.max_depth, self.depth + 1)

    def insert(self, point: np.ndarray, value: float):
        if self.depth >= self.max_depth or self.point_count < 1:
            self.value = value
            self.point_count += 1
            return

        self.is_leaf = False
        index = self.get_child_index(point)
        if self.children[index] is None:
            self.children[index] = self.create_child(index)
        self.children[index].insert(point, value)
        self.point_count += 1


class Octree:
    def __init__(self, bounds: Tuple[np.ndarray, np.ndarray], max_depth: int = 10):
        self.root = OctreeNode(bounds[0], bounds[1], max_depth)

    def insert(self, point: np.ndarray, value: float):
        self.root.insert(point, value)

    def query_radius(self, center: np.ndarray, radius: float) -> List[Tuple[np.ndarray, float]]:
        results = []
        self._query_radius(self.root, center, radius, results)
        return results

    def _query_radius(self, node: OctreeNode, center: np.ndarray, radius: float, results: List):
        node_center = node.get_center()
        node_radius = np.linalg.norm(node.get_size()) / 2
        dist = np.linalg.norm(center - node_center)

        if dist > radius + node_radius:
            return

        if node.is_leaf:
            if node.point_count > 0 and np.linalg.norm(center - node_center) <= radius:
                results.append((node_center, node.value))
            return

        for child in node.children:
            if child is not None:
                self._query_radius(child, center, radius, results)

    def get_all_voxels(self) -> List[Tuple[np.ndarray, float]]:
        results = []
        self._collect_leafs(self.root, results)
        return results

    def _collect_leafs(self, node: OctreeNode, results: List):
        if node.is_leaf and node.point_count > 0:
            results.append((node.get_center(), node.value))
        else:
            for child in node.children:
                if child is not None:
                    self._collect_leafs(child, results)


class SparseVoxelMap:
    def __init__(self, voxel_size: float = 0.1):
        self.voxel_size = voxel_size
        self.voxels: Dict[Tuple[int, int, int], float] = {}
        self.colors: Dict[Tuple[int, int, int], np.ndarray] = {}
        self.weights: Dict[Tuple[int, int, int], float] = {}

    def _world_to_voxel(self, point: np.ndarray) -> Tuple[int, int, int]:
        return tuple(np.floor(point / self.voxel_size).astype(int))

    def add_point(self, point: np.ndarray, value: float = 1.0, color: np.ndarray = None):
        key = self._world_to_voxel(point)
        if key in self.voxels:
            w = self.weights[key]
            self.voxels[key] = (self.voxels[key] * w + value) / (w + 1)
            self.weights[key] = w + 1
            if color is not None:
                self.colors[key] = (self.colors[key] * w + color) / (w + 1)
        else:
            self.voxels[key] = value
            self.weights[key] = 1
            if color is not None:
                self.colors[key] = color

    def get_value(self, point: np.ndarray) -> float:
        key = self._world_to_voxel(point)
        return self.voxels.get(key, 0.0)

    def get_points(self) -> np.ndarray:
        if not self.voxels:
            return np.empty((0, 3))
        centers = np.array([np.array(k) * self.voxel_size for k in self.voxels.keys()])
        return centers

    def get_colors(self) -> np.ndarray:
        if not self.colors:
            return np.empty((0, 3))
        return np.array(list(self.colors.values()))

    def raycast(self, origin: np.ndarray, direction: np.ndarray, max_dist: float = 100.0, step: float = None) -> Tuple[Optional[np.ndarray], Optional[float]]:
        if step is None:
            step = self.voxel_size / 2
        direction = direction / np.linalg.norm(direction)
        dist = 0.0
        while dist < max_dist:
            point = origin + direction * dist
            key = self._world_to_voxel(point)
            if key in self.voxels and self.voxels[key] > 0.5:
                return point, dist
            dist += step
        return None, None


class TSDFVolume:
    def __init__(self, size: Tuple[int, int, int], voxel_size: float, truncation: float = 0.04):
        self.size = np.array(size)
        self.voxel_size = voxel_size
        self.truncation = truncation
        self.origin = np.array([0.0, 0.0, 0.0])
        self.tsdf = np.zeros(size, dtype=np.float32)
        self.weights = np.zeros(size, dtype=np.float32)
        self.colors = np.zeros((*size, 3), dtype=np.uint8)

    def _world_to_voxel(self, point: np.ndarray) -> np.ndarray:
        return np.floor((point - self.origin) / self.voxel_size).astype(int)

    def integrate(self, depth: np.ndarray, color: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray):
        h, w = depth.shape
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        for v in range(h):
            for u in range(w):
                d = depth[v, u]
                if d <= 0 or d > 5.0:
                    continue

                x = (u - cx) * d / fx
                y = (v - cy) * d / fy
                z = d
                point_cam = np.array([x, y, z, 1])
                point_world = pose @ point_cam
                point_world = point_world[:3]

                voxel = self._world_to_voxel(point_world)
                if np.any(voxel < 0) or np.any(voxel >= self.size):
                    continue

                voxel_center = (voxel + 0.5) * self.voxel_size + self.origin
                dist = np.linalg.norm(point_world - voxel_center)
                tsdf_val = dist / self.truncation
                tsdf_val = np.clip(tsdf_val, -1, 1)

                idx = tuple(voxel)
                w = self.weights[idx]
                new_w = w + 1
                self.tsdf[idx] = (self.tsdf[idx] * w + tsdf_val) / new_w
                self.weights[idx] = new_w

                if color is not None:
                    c = color[v, u]
                    self.colors[idx] = (self.colors[idx] * w + c) / new_w

    def get_surface(self) -> Tuple[np.ndarray, np.ndarray]:
        vertices = []
        faces = []
        for x in range(self.size[0] - 1):
            for y in range(self.size[1] - 1):
                for z in range(self.size[2] - 1):
                    cube = [self.tsdf[x + dx, y + dy, z + dz] for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
                    if max(cube) > 0 and min(cube) < 0:
                        self._marching_cubes(x, y, z, vertices, faces)
        return np.array(vertices), np.array(faces)

    def _marching_cubes(self, x: int, y: int, z: int, vertices: List, faces: List):
        edge_table = [0] * 256
        tri_table = [] * 256


class KdTree:
    def __init__(self, points: np.ndarray, leaf_size: int = 10):
        self.points = points
        self.leaf_size = leaf_size
        self.tree = self._build(points, 0)

    def _build(self, points: np.ndarray, depth: int) -> dict:
        if len(points) <= self.leaf_size:
            return {'leaf': True, 'points': points}

        axis = depth % 3
        sorted_idx = np.argsort(points[:, axis])
        mid = len(points) // 2

        return {
            'leaf': False,
            'axis': axis,
            'split': points[sorted_idx[mid], axis],
            'left': self._build(points[sorted_idx[:mid]], depth + 1),
            'right': self._build(points[sorted_idx[mid+1:]], depth + 1),
        }

    def query_radius(self, query: np.ndarray, radius: float) -> List[int]:
        results = []
        self._query_radius(self.tree, query, radius, results)
        return results

    def _query_radius(self, node: dict, query: np.ndarray, radius: float, results: List[int]):
        if node['leaf']:
            dists = np.linalg.norm(node['points'] - query, axis=1)
            results.extend(np.where(dists <= radius)[0])
            return

        axis = node['axis']
        split = node['split']
        diff = query[axis] - split

        if diff <= 0:
            self._query_radius(node['left'], query, radius, results)
            if diff + radius >= 0:
                self._query_radius(node['right'], query, radius, results)
        else:
            self._query_radius(node['right'], query, radius, results)
            if diff - radius <= 0:
                self._query_radius(node['left'], query, radius, results)

    def query_knn(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        best = []
        self._query_knn(self.tree, query, k, best)
        return sorted(best, key=lambda x: x[1])[:k]

    def _query_knn(self, node: dict, query: np.ndarray, k: int, best: List[Tuple[int, float]]):
        if node['leaf']:
            dists = np.linalg.norm(node['points'] - query, axis=1)
            for i, d in enumerate(dists):
                if len(best) < k:
                    heapq.heappush(best, (-d, i))
                elif d < -best[0][0]:
                    heapq.heappushpop(best, (-d, i))
            return

        axis = node['axis']
        split = node['split']
        diff = query[axis] - split

        if diff <= 0:
            self._query_knn(node['left'], query, k, best)
            if len(best) < k or (diff**2) < -best[0][0]:
                self._query_knn(node['right'], query, k, best)
        else:
            self._query_knn(node['right'], query, k, best)
            if len(best) < k or (diff**2) < -best[0][0]:
                self._query_knn(node['left'], query, k, best)


class FastMarching:
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.shape = grid.shape
        self.dist = np.full_like(grid, np.inf)
        self.visited = np.zeros_like(grid, dtype=bool)

    def solve(self, start: Tuple[int, ...]) -> np.ndarray:
        import heapq
        heap = []
        self.dist[start] = 0
        heapq.heappush(heap, (0, start))

        while heap:
            dist, idx = heapq.heappop(heap)
            if self.visited[idx]:
                continue
            self.visited[idx] = True

            for neighbor in self._get_neighbors(idx):
                if self.visited[neighbor]:
                    continue
                new_dist = dist + 1.0
                if new_dist < self.dist[neighbor]:
                    self.dist[neighbor] = new_dist
                    heapq.heappush(heap, (new_dist, neighbor))

        return self.dist

    def _get_neighbors(self, idx: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        neighbors = []
        for dim in range(len(idx)):
            for delta in (-1, 1):
                new_idx = list(idx)
                new_idx[dim] += delta
                if 0 <= new_idx[dim] < self.shape[dim]:
                    neighbors.append(tuple(new_idx))
        return neighbors


class DistanceField:
    def __init__(self, occupancy: np.ndarray):
        self.occupancy = occupancy
        self.shape = occupancy.shape
        self.distance = np.zeros_like(occupancy, dtype=np.float32)

    def compute_signed_distance(self) -> np.ndarray:
        from scipy.ndimage import distance_transform_edt
        self.distance = distance_transform_edt(self.occupancy == 0)
        self.distance[self.occupancy > 0] = -distance_transform_edt(self.occupancy > 0)[self.occupancy > 0]
        return self.distance

    def compute_euclidean_distance(self) -> np.ndarray:
        from scipy.ndimage import distance_transform_edt
        self.distance = distance_transform_edt(self.occupancy == 0)
        return self.distance

    def get_gradient(self) -> np.ndarray:
        grad = np.zeros((*self.shape, len(self.shape)))
        for i in range(len(self.shape)):
            slices = [slice(None)] * len(self.shape)
            slices[i] = slice(1, -1)
            slices_p = slices.copy()
            slices_p[i] = slice(2, None)
            slices_n = slices.copy()
            slices_n[i] = slice(0, -2)
            grad[(slice(None),) * i + (slice(1, -1),) + (slice(None),) * (len(self.shape) - i - 1)] = \
                (self.distance[tuple(slices_p)] - self.distance[tuple(slices_n)]) / 2
        return grad


class VoxelTraversal:
    @staticmethod
    def dda_3d(start: np.ndarray, end: np.ndarray, voxel_size: float) -> List[Tuple[int, int, int]]:
        start_v = np.floor(start / voxel_size).astype(int)
        end_v = np.floor(end / voxel_size).astype(int)

        step = np.sign(end_v - start_v).astype(int)
        t_max = np.zeros(3)
        t_delta = np.zeros(3)

        for i in range(3):
            if step[i] != 0:
                next_boundary = (start_v[i] + (step[i] > 0)) * voxel_size
                t_max[i] = (next_boundary - start[i]) / (end[i] - start[i]) if end[i] != start[i] else float('inf')
                t_delta[i] = voxel_size / abs(end[i] - start[i]) if end[i] != start[i] else float('inf')
            else:
                t_max[i] = float('inf')
                t_delta[i] = float('inf')

        current = start_v.copy()
        voxels = [tuple(current)]

        while not np.array_equal(current, end_v):
            min_dim = np.argmin(t_max)
            current[min_dim] += step[min_dim]
            t_max[min_dim] += t_delta[min_dim]
            voxels.append(tuple(current))

        return voxels


class VoxelHashing:
    def __init__(self, bucket_size: int = 1024):
        self.bucket_size = bucket_size
        self.buckets: Dict[int, List[Tuple[Tuple[int, int, int], float]]] = {}

    def _hash(self, x: int, y: int, z: int) -> int:
        return (x * 73856093) ^ (y * 19349663) ^ (z * 83492791) % self.bucket_size

    def insert(self, x: int, y: int, z: int, value: float):
        h = self._hash(x, y, z)
        if h not in self.buckets:
            self.buckets[h] = []
        self.buckets[h].append(((x, y, z), value))

    def get(self, x: int, y: int, z: int) -> float:
        h = self._hash(x, y, z)
        if h in self.buckets:
            for (vx, vy, vz), val in self.buckets[h]:
                if vx == x and vy == y and vz == z:
                    return val
        return 0.0


class MarchingCubes:
    def __init__(self):
        self.edge_table = self._build_edge_table()
        self.tri_table = self._build_tri_table()

    def _build_edge_table(self) -> List[int]:
        table = [0] * 256
        return table

    def _build_tri_table(self) -> List[List[int]]:
        table = [[] for _ in range(256)]
        return table

    def extract_mesh(self, volume: np.ndarray, level: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        vertices = []
        faces = []

        for x in range(volume.shape[0] - 1):
            for y in range(volume.shape[1] - 1):
                for z in range(volume.shape[2] - 1):
                    cube = [volume[x + dx, y + dy, z + dz] for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
                    if max(cube) > level and min(cube) < level:
                        self._process_cube(x, y, z, volume, level, vertices, faces)

        return np.array(vertices), np.array(faces)

    def _process_cube(self, x: int, y: int, z: int, volume: np.ndarray, level: float, vertices: List, faces: List):
        pass


class PointCloudVoxelization:
    @staticmethod
    def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
        coords = np.floor(points / voxel_size).astype(np.int32)
        _, indices = np.unique(coords, axis=0, return_index=True)
        return points[indices]

    @staticmethod
    def voxel_grid_filter(points: np.ndarray, voxel_size: float, colors: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        coords = np.floor(points / voxel_size).astype(np.int32)
        _, indices = np.unique(coords, axis=0, return_index=True)
        downsampled = points[indices]
        downsampled_colors = colors[indices] if colors is not None else None
        return downsampled, downsampled_colors

    @staticmethod
    def voxel_centroid(points: np.ndarray, voxel_size: float) -> np.ndarray:
        coords = np.floor(points / voxel_size).astype(np.int32)
        unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
        centroids = np.zeros((len(unique_coords), 3))
        counts = np.zeros(len(unique_coords))
        np.add.at(centroids, inverse, points)
        np.add.at(counts, inverse, 1)
        centroids = centroids / counts[:, None]
        return centroids

    @staticmethod
    def voxel_bounds(points: np.ndarray, voxel_size: float) -> Dict[Tuple, Tuple[np.ndarray, np.ndarray]]:
        coords = np.floor(points / voxel_size).astype(np.int32)
        bounds = {}
        for i, c in enumerate(coords):
            key = tuple(c)
            if key not in bounds:
                bounds[key] = (points[i].copy(), points[i].copy())
            else:
                bounds[key] = (np.minimum(bounds[key][0], points[i]), np.maximum(bounds[key][1], points[i]))
        return bounds


class VoxelRaycasting:
    @staticmethod
    def cast_ray(grid: np.ndarray, origin: np.ndarray, direction: np.ndarray, max_dist: float = 100.0, step: float = 0.1) -> Tuple[Optional[np.ndarray], Optional[float]]:
        direction = direction / np.linalg.norm(direction)
        dist = 0.0
        while dist < max_dist:
            point = origin + direction * dist
            voxel = np.floor(point).astype(int)
            if np.all(voxel >= 0) and np.all(voxel < np.array(grid.shape)):
                if grid[tuple(voxel)] > 0.5:
                    return point, dist
            dist += step
        return None, None

    @staticmethod
    def voxel_traversal(start: np.ndarray, end: np.ndarray) -> List[np.ndarray]:
        start_v = np.floor(start).astype(int)
        end_v = np.floor(end).astype(int)

        step = np.sign(end_v - start_v)
        t_max = np.zeros(3)
        t_delta = np.zeros(3)

        for i in range(3):
            if step[i] != 0:
                next_boundary = start_v[i] + (step[i] > 0)
                t_max[i] = (next_boundary - start[i]) / (end[i] - start[i]) if end[i] != start[i] else float('inf')
                t_delta[i] = 1.0 / abs(end[i] - start[i]) if end[i] != start[i] else float('inf')
            else:
                t_max[i] = float('inf')
                t_delta[i] = float('inf')

        current = start_v.copy()
        voxels = [current.copy()]

        while not np.array_equal(current, end_v):
            min_dim = np.argmin(t_max)
            current[min_dim] += step[min_dim]
            t_max[min_dim] += t_delta[min_dim]
            voxels.append(current.copy())

        return voxels


class VoxelMorphology:
    @staticmethod
    def dilate(grid: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        from scipy.ndimage import binary_dilation
        return binary_dilation(grid, structure=np.ones((kernel_size,) * grid.ndim))

    @staticmethod
    def erode(grid: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        from scipy.ndimage import binary_erosion
        return binary_erosion(grid, structure=np.ones((kernel_size,) * grid.ndim))

    @staticmethod
    def open(grid: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        return VoxelMorphology.dilate(VoxelMorphology.erode(grid, kernel_size), kernel_size)

    @staticmethod
    def close(grid: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        return VoxelMorphology.erode(VoxelMorphology.dilate(grid, kernel_size), kernel_size)

    @staticmethod
    def skeletonize(grid: np.ndarray) -> np.ndarray:
        from scipy.ndimage import morphology
        return morphology.skeletonize(grid)

    @staticmethod
    def distance_transform(grid: np.ndarray) -> np.ndarray:
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(grid)


class VoxelSegmentation:
    @staticmethod
    def connected_components(grid: np.ndarray, connectivity: int = 6) -> Tuple[np.ndarray, int]:
        from scipy.ndimage import label
        if connectivity == 6:
            structure = np.ones((3,) * grid.ndim)
        else:
            structure = np.ones((3,) * grid.ndim)
        return label(grid, structure=structure)

    @staticmethod
    def region_growing(grid: np.ndarray, seeds: List[Tuple[int, ...]], threshold: float = 0.5) -> np.ndarray:
        labeled = np.zeros_like(grid, dtype=np.int32)
        for i, seed in enumerate(seeds):
            if grid[seed] < threshold:
                continue
            labeled[seed] = i + 1
            queue = [seed]
            while queue:
                current = queue.pop(0)
                for dim in range(grid.ndim):
                    for delta in (-1, 1):
                        neighbor = list(current)
                        neighbor[dim] += delta
                        if np.all(np.array(neighbor) >= 0) and np.all(np.array(neighbor) < np.array(grid.shape)):
                            if labeled[tuple(neighbor)] == 0 and grid[tuple(neighbor)] >= threshold:
                                labeled[tuple(neighbor)] = i + 1
                                queue.append(tuple(neighbor))
        return labeled

    @staticmethod
    def watershed(grid: np.ndarray, markers: np.ndarray = None) -> np.ndarray:
        from scipy.ndimage import watershed_ift
        if markers is None:
            from scipy.ndimage import label
            markers, _ = label(grid)
        return watershed_ift(-grid, markers)