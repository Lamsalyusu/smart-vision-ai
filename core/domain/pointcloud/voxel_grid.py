from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict


class PointType(Enum):
    XYZ = "xyz"
    XYZI = "xyzi"
    XYZRGB = "xyzrgb"
    XYZRGBA = "xyzrgba"
    XYZNORMAL = "xyznormal"
    XYZRGBNORMAL = "xyzrgbnormal"


class VoxelType(Enum):
    OCCUPANCY = "occupancy"
    TSDF = "tsdf"
    ESDF = "esdf"
    COLOR = "color"
    SEMANTIC = "semantic"
    WEIGHT = "weight"


@dataclass(frozen=True, slots=True)
class Voxel:
    x: int
    y: int
    z: int
    value: float = 1.0
    weight: float = 1.0
    color: Optional[Tuple[int, int, int]] = None
    semantic_label: int = 0


@dataclass(frozen=True, slots=True)
class VoxelGrid:
    voxels: Dict[Tuple[int, int, int], Voxel]
    resolution: float
    origin: np.ndarray
    size: Tuple[int, int, int]

    def get_voxel(self, x: int, y: int, z: int) -> Optional[Voxel]:
        return self.voxels.get((x, y, z))

    def set_voxel(self, x: int, y: int, z: int, voxel: Voxel):
        self.voxels[(x, y, z)] = voxel

    def world_to_grid(self, point: np.ndarray) -> Tuple[int, int, int]:
        grid = ((point - self.origin) / self.resolution).astype(int)
        return tuple(grid)

    def grid_to_world(self, x: int, y: int, z: int) -> np.ndarray:
        return self.origin + np.array([x, y, z]) * self.resolution


class VoxelGridBuilder:
    @staticmethod
    def from_point_cloud(
        points: np.ndarray,
        resolution: float,
        origin: np.ndarray = None,
        colors: np.ndarray = None,
        semantic_labels: np.ndarray = None
    ) -> VoxelGrid:
        if origin is None:
            origin = np.min(points, axis=0)

        max_coords = np.max(points, axis=0)
        size = tuple(np.ceil((max_coords - origin) / resolution).astype(int) + 1)

        voxels = {}
        for i, point in enumerate(points):
            grid = tuple(((point - origin) / resolution).astype(int))
            if grid not in voxels:
                color = None
                if colors is not None:
                    color = tuple(colors[i].astype(int))
                label = 0
                if semantic_labels is not None:
                    label = int(semantic_labels[i])
                voxels[grid] = Voxel(
                    x=grid[0], y=grid[1], z=grid[2],
                    color=color,
                    semantic_label=label
                )

        return VoxelGrid(
            voxels=voxels,
            resolution=resolution,
            origin=origin,
            size=size
        )

    @staticmethod
    def from_depth_image(
        depth: np.ndarray,
        intrinsics: np.ndarray,
        resolution: float,
        origin: np.ndarray = None,
        colors: np.ndarray = None
    ) -> VoxelGrid:
        h, w = depth.shape
        points = []
        colors_list = []

        for v in range(h):
            for u in range(w):
                d = depth[v, u]
                if d > 0:
                    x = (u - intrinsics[0, 2]) * d / intrinsics[0, 0]
                    y = (v - intrinsics[1, 2]) * d / intrinsics[1, 1]
                    z = d
                    points.append([x, y, z])
                    if colors is not None:
                        colors_list.append(colors[v, u])

        points = np.array(points)
        return VoxelGridBuilder.from_point_cloud(
            points, resolution=0.01, origin=np.array([0, 0, 0]),
            colors=np.array(colors_list) if colors_list else None
        )


class VoxelGridFilter:
    @staticmethod
    def statistical_outlier_removal(grid: VoxelGrid, nb_neighbors: int = 20, std_ratio: float = 2.0) -> VoxelGrid:
        from scipy.spatial import KDTree

        points = np.array([list(v) for v in grid.voxels.keys()])
        if len(points) == 0:
            return grid

        tree = KDTree(points)
        distances = np.zeros(len(points))

        for i, pt in enumerate(points):
            dists, _ = tree.query(pt, k=min(nb_neighbors + 1, len(points)))
            distances[i] = np.mean(dists[1:])

        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + std_ratio * std_dist

        mask = distances < threshold
        filtered_points = points[mask]

        new_voxels = {}
        for pt in filtered_points:
            key = tuple(pt)
            if key in grid.voxels:
                new_voxels[key] = grid.voxels[key]

        return VoxelGrid(
            voxels=new_voxels,
            resolution=grid.resolution,
            origin=grid.origin,
            size=grid.size
        )

    @staticmethod
    def radius_outlier_removal(grid: VoxelGrid, radius: float, min_neighbors: int = 5) -> VoxelGrid:
        from scipy.spatial import KDTree

        points = np.array([list(v) for v in grid.voxels.keys()])
        if len(points) == 0:
            return grid

        tree = KDTree(points)
        mask = np.zeros(len(points), dtype=bool)

        for i, pt in enumerate(points):
            indices = tree.query_ball_point(pt, radius)
            if len(indices) >= min_neighbors:
                mask[i] = True

        filtered_points = points[mask]

        new_voxels = {}
        for pt in filtered_points:
            key = tuple(pt)
            if key in grid.voxels:
                new_voxels[key] = grid.voxels[key]

        return VoxelGrid(
            voxels=new_voxels,
            resolution=grid.resolution,
            origin=grid.origin,
            size=grid.size
        )

    @staticmethod
    def bilateral_filter(grid: VoxelGrid, sigma_s: float = 0.1, sigma_r: float = 0.05, iterations: int = 3) -> VoxelGrid:
        from scipy.spatial import KDTree

        points = np.array([list(v) for v in grid.voxels.keys()])
        values = np.array([grid.voxels[tuple(p)].value for p in points])

        tree = KDTree(points)
        filtered_values = values.copy()

        for _ in range(iterations):
            for i, pt in enumerate(points):
                indices = tree.query_ball_point(pt, sigma_s * 3)
                if len(indices) < 3:
                    continue

                weights = np.zeros(len(indices))
                for j, idx in enumerate(indices):
                    spatial_dist = np.linalg.norm(points[idx] - points[i])
                    range_dist = abs(values[idx] - values[i])
                    weights[j] = math.exp(-spatial_dist**2 / (2 * sigma_s**2)) * math.exp(-range_dist**2 / (2 * sigma_r**2))

                weights = weights / (np.sum(weights) + 1e-6)
                filtered_values[i] = np.sum(values[indices] * weights)

        new_voxels = {}
        for i, pt in enumerate(points):
            key = tuple(pt)
            if key in grid.voxels:
                voxel = grid.voxels[key]
                new_voxels[key] = Voxel(
                    x=voxel.x, y=voxel.y, z=voxel.z,
                    value=filtered_values[i],
                    weight=voxel.weight,
                    color=voxel.color,
                    semantic_label=voxel.semantic_label
                )

        return VoxelGrid(
            voxels=new_voxels,
            resolution=grid.resolution,
            origin=grid.origin,
            size=grid.size
        )


class VoxelGridMorphology:
    @staticmethod
    def dilate(grid: VoxelGrid, kernel_size: int = 3) -> VoxelGrid:
        new_voxels = grid.voxels.copy()
        offsets = [(dx, dy, dz) for dx in range(-kernel_size//2, kernel_size//2 + 1)
                   for dy in range(-kernel_size//2, kernel_size//2 + 1)
                   for dz in range(-kernel_size//2, kernel_size//2 + 1)]

        for (x, y, z), voxel in grid.voxels.items():
            for dx, dy, dz in offsets:
                nx, ny, nz = x + dx, y + dy, z + dz
                if (nx, ny, nz) not in new_voxels:
                    new_voxels[(nx, ny, nz)] = Voxel(x=nx, y=ny, z=nz)

        return VoxelGrid(
            voxels=new_voxels,
            resolution=grid.resolution,
            origin=grid.origin,
            size=grid.size
        )

    @staticmethod
    def erode(grid: VoxelGrid, kernel_size: int = 3) -> VoxelGrid:
        offsets = [(dx, dy, dz) for dx in range(-kernel_size//2, kernel_size//2 + 1)
                   for dy in range(-kernel_size//2, kernel_size//2 + 1)
                   for dz in range(-kernel_size//2, kernel_size//2 + 1)]

        new_voxels = {}
        for (x, y, z), voxel in grid.voxels.items():
            keep = True
            for dx, dy, dz in [(dx, dy, dz) for dx in range(-1, 2) for dy in range(-1, 2) for dz in range(-1, 2)]:
                if (x + dx, y + dy, z + dz) not in grid.voxels:
                    keep = False
                    break

            if keep:
                new_voxels[(x, y, z)] = voxel

        return VoxelGrid(
            voxels=new_voxels,
            resolution=grid.resolution,
            origin=grid.origin,
            size=grid.size
        )

    @staticmethod
    def close(grid: VoxelGrid, kernel_size: int = 3) -> VoxelGrid:
        dilated = VoxelGridMorphology.dilate(grid, kernel_size)
        return VoxelGridMorphology.erode(dilated, kernel_size)

    @staticmethod
    def open(grid: VoxelGrid, kernel_size: int = 3) -> VoxelGrid:
        eroded = VoxelGridMorphology.erode(grid, kernel_size)
        return VoxelGridMorphology.dilate(eroded, kernel_size)


class VoxelGridInterpolation:
    @staticmethod
    def trilinear_interpolation(grid: VoxelGrid, point: np.ndarray) -> float:
        resolution = grid.resolution
        origin = grid.origin

        x = (point[0] - grid.origin[0]) / grid.resolution
        y = (point[1] - grid.origin[1]) / grid.resolution
        z = (point[2] - grid.origin[2]) / grid.resolution

        x0, y0, z0 = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

        dx = x - x0
        dy = y - y0
        dz = z - z0

        def get_val(x, y, z):
            voxel = grid.voxels.get((x, y, z))
            return voxel.value if voxel else 0.0

        c000 = get_val(x0, y0, z0)
        c001 = get_val(x0, y0, z1)
        c010 = get_val(x0, y1, z0)
        c011 = get_val(x0, y1, z1)
        c100 = get_val(x1, y0, z0)
        c101 = get_val(x1, y0, z1)
        c110 = get_val(x1, y1, z0)
        c111 = get_val(x1, y1, z1)

        c00 = c000 * (1 - dx) + c100 * dx
        c01 = c001 * (1 - dx) + c101 * dx
        c10 = c010 * (1 - dx) + c110 * dx
        c11 = c011 * (1 - dx) + c111 * dx

        c0 = c00 * (1 - dy) + c10 * dy
        c1 = c01 * (1 - dy) + c11 * dy

        return c0 * (1 - dz) + c1 * dz

    @staticmethod
    def nearest_neighbor(grid: VoxelGrid, point: np.ndarray) -> float:
        grid_pos = tuple(((point - grid.origin) / grid.resolution).astype(int))
        voxel = grid.voxels.get(grid_pos)
        return voxel.value if voxel else 0.0


class VoxelGridRaycasting:
    @staticmethod
    def cast_ray(
        grid: VoxelGrid,
        origin: np.ndarray,
        direction: np.ndarray,
        max_distance: float = 100.0
    ) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        direction = direction / np.linalg.norm(direction)
        t = 0.0
        step = grid.resolution * 0.5

        while t < max_distance:
            point = origin + direction * t
            grid_pos = tuple(((point - grid.origin) / grid.resolution).astype(int))

            if grid_pos in grid.voxels:
                hit_point = np.array(grid_pos) * grid.resolution + grid.origin
                return True, hit_point, t

            t += step

        return False, None, None

    @staticmethod
    def cast_ray_all(
        grid: VoxelGrid,
        origin: np.ndarray,
        direction: np.ndarray,
        max_distance: float = 100.0
    ) -> List[Tuple[np.ndarray, float]]:
        direction = direction / np.linalg.norm(direction)
        t = 0.0
        step = grid.resolution * 0.5
        hits = []

        while t < max_distance:
            point = origin + direction * t
            grid_pos = tuple(((point - grid.origin) / grid.resolution).astype(int))

            if grid_pos in grid.voxels:
                hit_point = np.array(grid_pos) * grid.resolution + grid.origin
                hits.append((hit_point, t))

            t += step

        return hits


class VoxelGridMeshing:
    @staticmethod
    def marching_cubes(grid: VoxelGrid, isolevel: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        vertices = []
        triangles = []

        size_x, size_y, size_z = grid.size

        for i in range(size_x - 1):
            for j in range(size_y - 1):
                for k in range(size_z - 1):
                    cube_vals = []
                    for dx in [0, 1]:
                        for dy in [0, 1]:
                            for dz in [0, 1]:
                                voxel = grid.voxels.get((i + dx, j + dy, k + dz))
                                cube_vals.append(voxel.value if voxel else 0.0)

                    cube_index = 0
                    for idx, val in enumerate(cube_vals):
                        if val > 0.5:
                            cube_index |= (1 << idx)

                    if cube_index == 0 or cube_index == 255:
                        continue

                    edges = VoxelGridMeshing._marching_cubes_edges(cube_index, (i, j, k), grid.resolution)
                    for edge in edges:
                        v1, v2 = edge
                        v1_idx = len(vertices)
                        v2_idx = len(vertices) + 1
                        vertices.append(v1)
                        vertices.append(v2)
                        triangles.append([v1_idx, v2_idx, v1_idx + 1])

        return np.array(vertices), np.array(triangles)

    @staticmethod
    def _marching_cubes_edges(cube_index: int, grid_pos: Tuple[int, int, int], resolution: float) -> List[Tuple[np.ndarray, np.ndarray]]:
        edge_table = {
            1: [(0, 0, 0), (1, 0, 0)],
        }
        return edge_table.get(cube_index, [])

    @staticmethod
    def surface_nets(grid: VoxelGrid, isolevel: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        vertices = []
        triangles = []

        size_x, size_y, size_z = grid.size

        for i in range(size_x - 1):
            for j in range(size_y - 1):
                for k in range(size_z - 1):
                    cube_vals = []
                    for dx in [0, 1]:
                        for dy in [0, 1]:
                            for dz in [0, 1]:
                                voxel = grid.voxels.get((i + dx, j + dy, k + dz))
                                cube_vals.append(voxel.value if voxel else 0.0)

                    if max(cube_vals) < 0.5 or min(cube_vals) > 0.5:
                        continue

                    center = np.array([
                        (grid.origin[0] + (i + 0.5) * grid.resolution),
                        (grid.origin[1] + (j + 0.5) * grid.resolution),
                        (grid.origin[2] + (k + 0.5) * grid.resolution)
                    ])

                    vertices.append(center)
                    v_idx = len(vertices) - 1

        return np.array(vertices), np.array(triangles)


class VoxelGridVisualizer:
    @staticmethod
    def draw_voxel_grid(
        grid: VoxelGrid,
        image_size: Tuple[int, int] = (800, 600),
        slice_axis: int = 2,
        slice_index: int = None
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if slice_index is None:
            slice_index = grid.size[slice_axis] // 2

        if slice_axis == 2:
            x_range = range(grid.size[0])
            y_range = range(grid.size[1])
            fixed_coord = slice_index
            coord_func = lambda x, y: (x, y, slice_index)
        elif slice_axis == 1:
            x_range = range(grid.size[0])
            y_range = range(grid.size[2])
            fixed_coord = slice_index
            coord_func = lambda x, z: (x, slice_index, z)
        else:
            x_range = range(grid.size[1])
            y_range = range(grid.size[2])
            fixed_coord = slice_index
            coord_func = lambda y, z: (slice_index, y, z)

        positions = []
        values = []

        for x in x_range:
            for y in y_range:
                pos = coord_func(x, y)
                voxel = grid.voxels.get(pos)
                if voxel and voxel.value > 0:
                    positions.append((x, y))
                    values.append(voxel.value)

        if not positions:
            return vis

        pos_array = np.array(positions)
        min_coords = np.min(pos_array, axis=0)
        max_coords = np.max(pos_array, axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1

        scale = min(image_size[0], image_size[1]) * 0.8 / max(range_coords)

        for (x, y), val in zip(positions, values):
            px = int((x - min_coords[0]) * scale + image_size[0] * 0.1)
            py = int(image_size[1] - (y - min_coords[1]) * scale - image_size[1] * 0.1)

            intensity = int(val * 255)
            cv2.circle(vis, (px, py), 2, (intensity, intensity, intensity), -1)

        return vis

    @staticmethod
    def draw_point_cloud_from_voxels(
        grid: VoxelGrid,
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        points = []
        colors = []

        for (x, y, z), voxel in grid.voxels.items():
            if voxel.value > 0:
                world = grid.origin + np.array([x, y, z]) * grid.resolution
                points.append(world[:2])

                if voxel.color:
                    colors.append(voxel.color)
                else:
                    colors.append((int(voxel.value * 255),) * 3)

        if not points:
            return vis

        points = np.array(points)
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1

        scale = min(image_size[0], image_size[1]) * 0.8 / max(range_coords)

        for point, color in zip(points, colors):
            x = int((point[0] - min_coords[0]) * scale + image_size[0] * 0.1)
            y = int(image_size[1] - (point[1] - min_coords[1]) * scale - image_size[1] * 0.1)

            if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                cv2.circle(vis, (x, y), 1, color, -1)

        return vis


class VoxelGridMetrics:
    @staticmethod
    def occupancy_ratio(grid: VoxelGrid) -> float:
        total = np.prod(grid.size)
        occupied = len(grid.voxels)
        return occupied / total if total > 0 else 0.0

    @staticmethod
    def surface_area(grid: VoxelGrid) -> float:
        area = 0.0
        face_area = grid.resolution**2

        for (x, y, z), voxel in grid.voxels.items():
            if voxel.value <= 0:
                continue

            neighbors = [
                (x+1, y, z), (x-1, y, z),
                (x, y+1, z), (x, y-1, z),
                (x, y, z+1), (x, y, z-1)
            ]

            for nx, ny, nz in neighbors:
                neighbor = grid.voxels.get((nx, ny, nz))
                if not neighbor or neighbor.value <= 0:
                    area += face_area

        return area

    @staticmethod
    def volume(grid: VoxelGrid) -> float:
        voxel_volume = grid.resolution**3
        occupied = sum(1 for v in grid.voxels.values() if v.value > 0)
        return occupied * voxel_volume

    @staticmethod
    def bounding_box(grid: VoxelGrid) -> Tuple[np.ndarray, np.ndarray]:
        if not grid.voxels:
            return np.zeros(3), np.zeros(3)

        coords = np.array(list(grid.voxels.keys()))
        min_coords = np.min(coords, axis=0) * grid.resolution + grid.origin
        max_coords = np.max(coords, axis=0) * grid.resolution + grid.origin

        return min_coords, max_coords

    @staticmethod
    def centroid(grid: VoxelGrid) -> np.ndarray:
        if not grid.voxels:
            return np.zeros(3)

        coords = np.array(list(grid.voxels.keys()))
        centroid_grid = np.mean(coords, axis=0)
        return centroid_grid * grid.resolution + grid.origin


class VoxelGridExporter:
    @staticmethod
    def export_ply(grid: VoxelGrid, path: str):
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")

            occupied = sum(1 for v in grid.voxels.values() if v.value > 0)
            f.write(f"element vertex {occupied}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            for (x, y, z), voxel in grid.voxels.items():
                if voxel.value > 0:
                    world = grid.origin + np.array([x, y, z]) * grid.resolution
                    color = voxel.color if voxel.color else (255, 255, 255)
                    f.write(f"{world[0]} {world[1]} {world[2]} {color[0]} {color[1]} {color[2]}\n")

    @staticmethod
    def export_binvox(grid: VoxelGrid, path: str):
        pass

    @staticmethod
    def export_obj(grid: VoxelGrid, path: str, isolevel: float = 0.5):
        vertices, triangles = VoxelGridMeshing.marching_cubes(grid, isolevel)

        with open(path, 'w') as f:
            f.write("# OBJ file from voxel grid\n")
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for t in triangles:
                f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")


class VoxelGridOperations:
    @staticmethod
    def merge(grids: List[VoxelGrid]) -> VoxelGrid:
        if not grids:
            return VoxelGrid(voxels={}, resolution=0.01, origin=np.zeros(3), size=(0, 0, 0))

        result = grids[0]
        for grid in grids[1:]:
            for pos, voxel in grid.voxels.items():
                if pos not in result.voxels or voxel.value > result.voxels[pos].value:
                    result.voxels[pos] = voxel

        return result

    @staticmethod
    def intersect(grids: List[VoxelGrid]) -> VoxelGrid:
        if not grids:
            return VoxelGrid(voxels={}, resolution=0.01, origin=np.zeros(3), size=(0, 0, 0))

        result = grids[0]
        for grid in grids[1:]:
            common_keys = set(result.voxels.keys()) & set(grid.voxels.keys())
            result.voxels = {k: result.voxels[k] for k in common_keys}

        return result

    @staticmethod
    def subtract(grid_a: VoxelGrid, grid_b: VoxelGrid) -> VoxelGrid:
        result = grid_a
        for key in grid_b.voxels:
            if key in result.voxels:
                del result.voxels[key]

        return result

    @staticmethod
    def union(grids: List[VoxelGrid]) -> VoxelGrid:
        return VoxelGridOperations.merge(grids)

    @staticmethod
    def transform(grid: VoxelGrid, transform: np.ndarray) -> VoxelGrid:
        new_voxels = {}
        for (x, y, z), voxel in grid.voxels.items():
            point = grid.origin + np.array([x, y, z]) * grid.resolution
            point_h = np.array([point[0], point[1], point[2], 1.0])
            transformed = transform @ point_h
            new_point = transformed[:3]
            new_grid = tuple(((new_point - grid.origin) / grid.resolution).astype(int))
            new_voxels[new_grid] = voxel

        return VoxelGrid(
            voxels=new_voxels,
            resolution=grid.resolution,
            origin=grid.origin,
            size=grid.size
        )


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet