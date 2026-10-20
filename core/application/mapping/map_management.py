from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict
import copy


class MapType(Enum):
    OCCUPANCY_GRID = "occupancy_grid"
    HEIGHT_MAP = "height_map"
    SEMANTIC = "semantic"
    TOPOLOGICAL = "topological"
    DENSITY = "density"
    COST = "cost"
    UNCERTAINTY = "uncertainty"


@dataclass(frozen=True, slots=True)
class MapConfig:
    resolution: float = 0.1
    size: Tuple[int, int] = (1000, 1000)
    origin: Tuple[float, float] = (0.0, 0.0)
    map_type: MapType = MapType.OCCUPANCY_GRID


@dataclass(frozen=True, slots=True)
class MapCell:
    x: int
    y: int
    value: float
    variance: float = 0.0
    update_count: int = 0
    last_update: float = 0.0


@dataclass(frozen=True, slots=True)
class OccupancyGridMap:
    grid: np.ndarray
    resolution: float
    origin: Tuple[float, float]
    width: int
    height: int
    log_odds: np.ndarray = field(default_factory=lambda: np.array([]))
    updated_cells: Set[Tuple[int, int]] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class HeightMap:
    grid: np.ndarray
    resolution: float
    origin: Tuple[float, float]
    width: int
    height: int
    variance: np.ndarray = field(default_factory=lambda: np.array([]))
    updated_cells: Set[Tuple[int, int]] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class SemanticMap:
    grid: np.ndarray
    resolution: float
    origin: Tuple[float, float]
    width: int
    height: int
    class_ids: Dict[int, str] = field(default_factory=dict)
    confidence: np.ndarray = field(default_factory=lambda: np.array([]))
    updated_cells: Set[Tuple[int, int]] = field(default_factory=set)


class MapBuilder:
    def __init__(self, config: MapConfig):
        self.config = config
        self.maps: Dict[str, Any] = {}

    def create_occupancy_grid(self, name: str = "default") -> OccupancyGridMap:
        grid = np.zeros((self.config.height, self.config.width), dtype=np.int8)
        log_odds = np.zeros((self.config.height, self.config.width), dtype=np.float32)
        return OccupancyGridMap(
            grid=grid,
            resolution=self.config.resolution,
            origin=self.config.origin,
            width=self.config.width,
            height=self.config.height,
            log_odds=log_odds
        )

    def create_height_map(self, name: str = "height") -> HeightMap:
        grid = np.full((self.config.height, self.config.width), np.nan, dtype=np.float32)
        variance = np.zeros((self.config.height, self.config.width), dtype=np.float32)
        return HeightMap(
            grid=grid,
            resolution=self.config.resolution,
            origin=self.config.origin,
            width=self.config.width,
            height=self.config.height,
            variance=variance
        )

    def create_semantic_map(self, name: str = "semantic", num_classes: int = 20) -> SemanticMap:
        grid = np.zeros((self.config.height, self.config.width), dtype=np.int32)
        confidence = np.zeros((self.config.height, self.config.width), dtype=np.float32)
        return SemanticMap(
            grid=grid,
            resolution=self.config.resolution,
            origin=self.config.origin,
            width=self.config.width,
            height=self.config.height,
            confidence=confidence
        )

    def create_map(self, name: str, map_type: MapType) -> Any:
        if map_type == MapType.OCCUPANCY_GRID:
            return self.create_occupancy_grid(name)
        elif map_type == MapType.HEIGHT_MAP:
            return self.create_height_map(name)
        elif map_type == MapType.SEMANTIC:
            return self.create_semantic_map(name)
        else:
            raise ValueError(f"Unknown map type: {map_type}")


class MapUpdater:
    def __init__(self, config: MapConfig):
        self.config = config
        self.log_odds_free = np.log(0.3 / 0.7)
        self.log_odds_occupied = np.log(0.7 / 0.3)
        self.log_odds_prior = 0.0

    def update_occupancy_grid(
        self,
        grid_map: OccupancyGridMap,
        sensor_pose: np.ndarray,
        sensor_data: np.ndarray,
        sensor_range: float,
        fov: float = 360.0
    ) -> OccupancyGridMap:
        if grid_map.log_odds.size == 0:
            grid_map = grid_map._replace(log_odds=np.zeros((grid_map.height, grid_map.width), dtype=np.float32))

        sensor_x, sensor_y = sensor_pose[0], sensor_pose[1]
        sensor_grid_x = int((sensor_x - grid_map.origin[0]) / grid_map.resolution)
        sensor_grid_y = int((sensor_y - grid_map.origin[1]) / grid_map.resolution)

        max_cells = int(sensor_range / grid_map.resolution)

        for angle_deg in np.linspace(0, 360, 360):
            if fov < 360:
                angle_offset = angle_deg - 180
                if abs(angle_offset) > fov / 2:
                    continue

            angle_rad = math.radians(angle_deg)
            ray_dx = math.cos(angle_rad)
            ray_dy = math.sin(angle_rad)

            hit = False
            for step in range(1, max_cells + 1):
                gx = sensor_grid_x + int(ray_dx * step)
                gy = sensor_grid_y + int(ray_dy * step)

                if not (0 <= gx < grid_map.width and 0 <= gy < grid_map.height):
                    break

                if gx == sensor_grid_x and gy == sensor_grid_y:
                    continue

                if (gx, gy) in grid_map.updated_cells:
                    continue

                if grid_map.grid[gy, gx] == 100:
                    grid_map.log_odds[gy, gx] = min(grid_map.log_odds[gy, gx] + self.log_odds_occupied, 100)
                    grid_map.grid[gy, gx] = 100
                    grid_map.updated_cells.add((gx, gy))
                    hit = True
                    break
                else:
                    grid_map.log_odds[gy, gx] += self.log_odds_free
                    grid_map.log_odds[gy, gx] = max(grid_map.log_odds[gy, gx], -100)
                    grid_map.updated_cells.add((gx, gy))

                if grid_map.log_odds[gy, gx] > 0:
                    grid_map.grid[gy, gx] = min(100, int(grid_map.log_odds[gy, gx]))

            if not hit:
                for step in range(1, max_cells + 1):
                    gx = sensor_grid_x + int(ray_dx * step)
                    gy = sensor_grid_y + int(ray_dy * step)

                    if not (0 <= gx < grid_map.width and 0 <= gy < grid_map.height):
                        break

                    if grid_map.grid[gy, gx] != 0:
                        break

                    grid_map.log_odds[gy, gx] += self.log_odds_free
                    grid_map.log_odds[gy, gx] = max(grid_map.log_odds[gy, gx], -100)
                    grid_map.updated_cells.add((gx, gy))

        return grid_map

    def update_height_map(
        self,
        height_map: HeightMap,
        sensor_pose: np.ndarray,
        depth_image: np.ndarray,
        camera_matrix: np.ndarray,
        max_depth: float = 100.0
    ) -> HeightMap:
        h, w = depth_image.shape
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

        sensor_x, sensor_y, sensor_z = sensor_pose[:3]
        sensor_grid_x = int((sensor_pose[0] - height_map.origin[0]) / height_map.resolution)
        sensor_grid_y = int((sensor_pose[1] - height_map.origin[1]) / height_map.resolution)

        for v in range(h):
            for u in range(w):
                depth = depth_image[v, u]
                if depth <= 0 or depth > max_depth:
                    continue

                x = (u - cx) * depth / fx
                y = (v - cy) * depth / fy
                z = depth

                world_x = sensor_pose[0] + x
                world_y = sensor_pose[1] + y
                world_z = sensor_pose[2] + z

                gx = int((world_x - height_map.origin[0]) / height_map.resolution)
                gy = int((world_y - height_map.origin[1]) / height_map.resolution)

                if 0 <= gx < height_map.width and 0 <= gy < height_map.height:
                    old_val = height_map.grid[gy, gx]
                    if np.isnan(old_val):
                        height_map.grid[gy, gx] = world_z
                        height_map.variance[gy, gx] = 0.01
                    else:
                        n = 1.0 / (height_map.variance[gy, gx] + 1e-6)
                        new_val = (old_val * n + world_z) / (n + 1)
                        height_map.grid[gy, gx] = new_val
                        height_map.variance[gy, gx] = 1.0 / (n + 1)

                    height_map.updated_cells.add((gx, gy))

        return height_map

    def update_semantic_map(
        self,
        semantic_map: SemanticMap,
        detections: List[Dict],
        camera_pose: np.ndarray,
        camera_matrix: np.ndarray
    ) -> SemanticMap:
        for det in detections:
            bbox = det['bbox']
            class_id = det['class_id']
            confidence = det['confidence']

            center_u = (bbox[0] + bbox[2]) / 2
            center_v = (bbox[1] + bbox[3]) / 2

            depth = det.get('depth', 5.0)
            fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
            cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

            x = (center_u - camera_matrix[0, 2]) * depth / camera_matrix[0, 0]
            y = (center_v - camera_matrix[1, 2]) * depth / camera_matrix[1, 1]
            z = depth

            world_x = camera_pose[0] + x
            world_y = camera_pose[1] + y

            gx = int((world_x - semantic_map.origin[0]) / semantic_map.resolution)
            gy = int((world_y - semantic_map.origin[1]) / semantic_map.resolution)

            if 0 <= gx < semantic_map.width and 0 <= gy < semantic_map.height:
                old_class = semantic_map.grid[gy, gx]
                old_conf = semantic_map.confidence[gy, gx]

                if confidence > old_conf:
                    semantic_map.grid[gy, gx] = class_id
                    semantic_map.confidence[gy, gx] = confidence
                    semantic_map.updated_cells.add((gx, gy))

        return semantic_map


class MapFusion:
    def __init__(self, config: MapConfig):
        self.config = config

    def fuse_maps(
        self,
        maps: List[Any],
        weights: Optional[List[float]] = None
    ) -> Any:
        if not maps:
            return None

        if weights is None:
            weights = [1.0 / len(maps)] * len(maps)

        base_map = maps[0]
        for i, map_obj in enumerate(maps[1:]):
            base_map = self._fuse_two_maps(base_map, map_obj, weights[0], weights[i+1])

        return base_map

    def _fuse_two_maps(self, map1: Any, map2: Any, w1: float, w2: float) -> Any:
        if hasattr(map1, 'grid') and hasattr(map2, 'grid'):
            new_grid = map1.grid * w1 + map2.grid * w2
            new_map = type(map1)(
                grid=new_grid,
                resolution=map1.resolution,
                origin=map1.origin,
                width=map1.width,
                height=map1.height
            )
            return new_map
        return map1


class MapVisualizer:
    @staticmethod
    def draw_occupancy_grid(
        grid_map: OccupancyGridMap,
        image_size: Tuple[int, int] = (800, 600),
        robot_pose: Optional[np.ndarray] = None
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        scale_x = image_size[0] / grid_map.width
        scale_y = image_size[1] / grid_map.height

        for y in range(grid_map.height):
            for x in range(grid_map.width):
                val = grid_map.grid[y, x]
                if val > 0:
                    intensity = min(255, val * 2)
                    px = int(x * scale_x)
                    py = int(image_size[1] - y * scale_y)
                    if 0 <= px < image_size[0] and 0 <= py < image_size[1]:
                        vis[py, px] = (intensity, intensity, intensity)

        if robot_pose is not None:
            rx = int(robot_pose[0] * 100)
            ry = int(robot_pose[1] * 100)
            cv2.circle(vis, (rx, ry), 5, (0, 0, 255), -1)

        return vis

    @staticmethod
    def draw_height_map(
        height_map: HeightMap,
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        valid = ~np.isnan(height_map.grid)
        if not np.any(valid):
            return vis

        min_h = np.nanmin(height_map.grid)
        max_h = np.max(height_map.grid)

        scale_x = image_size[0] / height_map.width
        scale_y = image_size[1] / height_map.height

        for y in range(height_map.height):
            for x in range(height_map.width):
                if not np.isnan(height_map.grid[y, x]):
                    h = height_map.grid[y, x]
                    intensity = int((height_map.grid[y, x] - min_h) / (max_h - min_h + 1e-6) * 255)
                    px = int(x * scale_x)
                    py = int(image_size[1] - y * scale_y)
                    if 0 <= px < image_size[0] and 0 <= py < image_size[1]:
                        cv2.circle(vis, (px, py), 1, (intensity, intensity, intensity), -1)

        return vis

    @staticmethod
    def draw_semantic_map(
        semantic_map: SemanticMap,
        image_size: Tuple[int, int] = (800, 600),
        color_map: Dict[int, Tuple[int, int, int]] = None
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if color_map is None:
            color_map = {i: (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for i in range(256)}

        scale_x = image_size[0] / semantic_map.width
        scale_y = image_size[1] / semantic_map.height

        for y in range(semantic_map.height):
            for x in range(semantic_map.width):
                class_id = semantic_map.grid[y, x]
                if class_id > 0:
                    color = color_map.get(class_id, (255, 255, 255))
                    px = int(x * scale_x)
                    py = int(image_size[1] - y * scale_y)
                    if 0 <= px < image_size[0] and 0 <= py < image_size[1]:
                        cv2.rectangle(vis, (px, py), (px + 2, py + 2), color, -1)

        return vis


class MapSerializer:
    @staticmethod
    def save_occupancy_grid(grid_map: OccupancyGridMap, path: str):
        np.savez_compressed(
            path,
            grid=grid_map.grid,
            resolution=grid_map.resolution,
            origin=grid_map.origin,
            width=grid_map.width,
            height=grid_map.height,
            log_odds=grid_map.log_odds
        )

    @staticmethod
    def load_occupancy_grid(path: str) -> OccupancyGridMap:
        data = np.load(path)
        return OccupancyGridMap(
            grid=data['grid'],
            resolution=data['resolution'].item(),
            origin=tuple(data['origin']),
            width=int(data['width']),
            height=int(data['height']),
            log_odds=data['log_odds']
        )

    @staticmethod
    def save_height_map(height_map: HeightMap, path: str):
        np.savez_compressed(
            path,
            grid=height_map.grid,
            resolution=height_map.resolution,
            origin=height_map.origin,
            width=height_map.width,
            height=height_map.height,
            variance=height_map.variance
        )

    @staticmethod
    def load_height_map(path: str) -> HeightMap:
        data = np.load(path)
        return HeightMap(
            grid=data['grid'],
            resolution=data['resolution'].item(),
            origin=tuple(data['origin']),
            width=int(data['width']),
            height=int(data['height']),
            variance=data['variance']
        )

    @staticmethod
    def save_semantic_map(semantic_map: SemanticMap, path: str):
        np.savez_compressed(
            path,
            grid=semantic_map.grid,
            resolution=semantic_map.resolution,
            origin=semantic_map.origin,
            width=semantic_map.width,
            height=semantic_map.height,
            confidence=semantic_map.confidence,
            class_ids=semantic_map.class_ids
        )

    @staticmethod
    def load_semantic_map(path: str) -> SemanticMap:
        data = np.load(path)
        return SemanticMap(
            grid=data['grid'],
            resolution=data['resolution'].item(),
            origin=tuple(data['origin']),
            width=int(data['width']),
            height=int(data['height']),
            confidence=data['confidence'],
            class_ids=data['class_ids'].item()
        )


class MapMetrics:
    @staticmethod
    def occupied_ratio(grid_map: OccupancyGridMap) -> float:
        occupied = np.sum(grid_map.grid > 0)
        total = grid_map.width * grid_map.height
        return float(occupied) / total if total > 0 else 0.0

    @staticmethod
    def free_ratio(grid_map: OccupancyGridMap) -> float:
        free = np.sum(grid_map.grid == 0)
        total = grid_map.width * grid_map.height
        return float(free) / total if total > 0 else 0.0

    @staticmethod
    def unknown_ratio(grid_map: OccupancyGridMap) -> float:
        unknown = np.sum(grid_map.grid < 0)
        total = grid_map.width * grid_map.height
        return float(unknown) / total if total > 0 else 0.0

    @staticmethod
    def map_coverage(height_map: HeightMap) -> float:
        valid = ~np.isnan(height_map.grid)
        return float(np.sum(valid)) / height_map.grid.size

    @staticmethod
    def semantic_diversity(semantic_map: SemanticMap) -> int:
        unique_classes = np.unique(semantic_map.grid[semantic_map.grid > 0])
        return len(unique_classes)

    @staticmethod
    def entropy(semantic_map: SemanticMap) -> float:
        valid = semantic_map.grid > 0
        if not np.any(valid):
            return 0.0

        classes, counts = np.unique(semantic_map.grid[valid], return_counts=True)
        probs = counts / np.sum(counts)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return float(entropy)


class MapManager:
    def __init__(self, config: MapConfig):
        self.config = config
        self.builder = MapBuilder(config)
        self.updater = MapUpdater(config)
        self.fusion = MapFusion(config)
        self.maps: Dict[str, Any] = {}
        self.active_map: Optional[str] = None

    def create_map(self, name: str, map_type: MapType) -> Any:
        map_obj = self.builder.create_map(name, map_type)
        self.maps[name] = map_obj
        if self.active_map is None:
            self.active_map = name
        return map_obj

    def get_map(self, name: str) -> Optional[Any]:
        return self.maps.get(name)

    def update_map(
        self,
        name: str,
        sensor_pose: np.ndarray,
        sensor_data: Any,
        sensor_type: str,
        **kwargs
    ) -> Any:
        if name not in self.maps:
            return None

        map_obj = self.maps[name]

        if isinstance(map_obj, OccupancyGridMap):
            return self.updater.update_occupancy_grid(map_obj, **kwargs)
        elif isinstance(map_obj, HeightMap):
            return self.updater.update_height_map(map_obj, **kwargs)
        elif isinstance(map_obj, SemanticMap):
            return self.updater.update_semantic_map(map_obj, **kwargs)

        return map_obj

    def fuse_maps(self, map_names: List[str], weights: List[float] = None) -> Any:
        maps = [self.maps[name] for name in map_names if name in self.maps]
        return self.fusion.fuse_maps(maps, weights)

    def save_map(self, name: str, path: str):
        if name not in self.maps:
            return False

        map_obj = self.maps[name]
        if isinstance(map_obj, OccupancyGridMap):
            MapSerializer.save_occupancy_grid(map_obj, path)
        elif isinstance(map_obj, HeightMap):
            MapSerializer.save_height_map(map_obj, path)
        elif isinstance(map_obj, SemanticMap):
            MapSerializer.save_semantic_map(map_obj, path)
        return True

    def load_map(self, name: str, path: str, map_type: MapType) -> Any:
        if map_type == MapType.OCCUPANCY_GRID:
            map_obj = MapSerializer.load_occupancy_grid(path)
        elif map_type == MapType.HEIGHT_MAP:
            map_obj = MapSerializer.load_height_map(path)
        elif map_type == MapType.SEMANTIC:
            map_obj = MapSerializer.load_semantic_map(path)
        else:
            return None

        self.maps[name] = map_obj
        if self.active_map is None:
            self.active_map = name
        return map_obj


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
