from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict
from scipy.spatial import KDTree


class SegmentationMethod(Enum):
    EUCLIDEAN_CLUSTERING = "euclidean_clustering"
    REGION_GROWING = "region_growing"
    CONDITIONAL_EUCLIDEAN = "conditional_euclidean"
    RANSAC_PLANE = "ransac_plane"
    RANSAC_CYLINDER = "ransac_cylinder"
    RANSAC_SPHERE = "ransac_sphere"
    SAC_SEGMENTATION = "sac_segmentation"
    PROGRESSIVE_MORPHOLOGICAL = "progressive_morphological"
    SUPERVOXEL = "supervoxel"
    NORMAL_BASED = "normal_based"
    DEEP_LEARNING = "deep_learning"


@dataclass(frozen=True, slots=True)
class PointCloud:
    points: np.ndarray
    normals: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    intensity: Optional[np.ndarray] = None
    curvature: Optional[np.ndarray] = None

    @property
    def num_points(self) -> int:
        return len(self.points)


@dataclass(frozen=True, slots=True)
class Segment:
    segment_id: int
    indices: np.ndarray
    centroid: np.ndarray
    bounding_box: Tuple[np.ndarray, np.ndarray]
    normal: Optional[np.ndarray] = None
    area: float = 0.0
    volume: float = 0.0
    point_count: int = 0


@dataclass(frozen=True, slots=True)
class PlaneModel:
    coefficients: np.ndarray
    indices: np.ndarray
    centroid: np.ndarray
    normal: np.ndarray
    area: float
    inliers: int


class BaseSegmenter(ABC):
    @abstractmethod
    def segment(self, cloud: PointCloud) -> List[Segment]:
        pass


class EuclideanClusterExtractor(BaseSegmenter):
    def __init__(
        self,
        tolerance: float = 0.05,
        min_cluster_size: int = 100,
        max_cluster_size: int = 25000
    ):
        self.tolerance = tolerance
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size

    def segment(self, cloud: PointCloud) -> List[Segment]:
        if cloud.num_points == 0:
            return []

        tree = KDTree(cloud.points)
        visited = np.zeros(cloud.num_points, dtype=bool)
        segments = []

        for i in range(cloud.num_points):
            if visited[i]:
                continue

            cluster_indices = []
            queue = [i]
            visited[i] = True

            while queue:
                idx = queue.pop(0)
                cluster_indices.append(idx)

                neighbors = tree.query_ball_point(cloud.points[idx], self.tolerance)
                for n_idx in neighbors:
                    if not visited[n_idx]:
                        visited[n_idx] = True
                        queue.append(n_idx)

            if self.min_cluster_size <= len(cluster_indices) <= self.max_cluster_size:
                cluster_points = cloud.points[cluster_indices]
                centroid = np.mean(cluster_points, axis=0)
                min_coords = np.min(cluster_points, axis=0)
                max_coords = np.max(cluster_points, axis=0)

                segment = Segment(
                    segment_id=len(segments),
                    indices=np.array(cluster_indices, dtype=np.int32),
                    centroid=centroid,
                    bounding_box=(min_coords, max_coords),
                    point_count=len(cluster_indices)
                )
                segments.append(segment)

        return segments


class RegionGrowingSegmenter(BaseSegmenter):
    def __init__(
        self,
        smoothness_threshold: float = 0.1,
        curvature_threshold: float = 0.05,
        min_cluster_size: int = 100,
        max_cluster_size: int = 25000,
        number_of_neighbours: int = 30
    ):
        self.smoothness_threshold = smoothness_threshold
        self.curvature_threshold = curvature_threshold
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.number_of_neighbours = number_of_neighbours

    def segment(self, cloud: PointCloud) -> List[Segment]:
        if cloud.normals is None:
            cloud = self._estimate_normals(cloud)
        if cloud.curvature is None:
            cloud = self._estimate_curvature(cloud)

        tree = KDTree(cloud.points)
        visited = np.zeros(cloud.num_points, dtype=bool)
        segments = []

        for i in range(cloud.num_points):
            if visited[i] or cloud.curvature[i] > self.curvature_threshold:
                continue

            cluster_indices = [i]
            visited[i] = True
            queue = [i]

            while queue:
                idx = queue.pop(0)
                neighbors = tree.query_ball_point(cloud.points[idx], 0.1)

                for n_idx in neighbors[:self.number_of_neighbours]:
                    if visited[n_idx]:
                        continue
                    if cloud.curvature[n_idx] > self.curvature_threshold:
                        continue

                    angle = math.acos(np.clip(np.dot(cloud.normals[idx], cloud.normals[n_idx]), -1, 1))
                    if angle < self.smoothness_threshold:
                        cluster_indices.append(n_idx)
                        visited[n_idx] = True
                        queue.append(n_idx)

            if self.min_cluster_size <= len(cluster_indices) <= self.max_cluster_size:
                cluster_points = cloud.points[cluster_indices]
                centroid = np.mean(cluster_points, axis=0)
                min_coords = np.min(cluster_points, axis=0)
                max_coords = np.max(cluster_points, axis=0)

                normal = np.mean(cloud.normals[cluster_indices], axis=0)
                normal = normal / (np.linalg.norm(normal) + 1e-6)

                segment = Segment(
                    segment_id=len(segments),
                    indices=np.array(cluster_indices, dtype=np.int32),
                    centroid=centroid,
                    bounding_box=(min_coords, max_coords),
                    normal=normal,
                    point_count=len(cluster_indices)
                )
                segments.append(segment)

        return segments

    def _estimate_normals(self, cloud: PointCloud) -> PointCloud:
        tree = KDTree(cloud.points)
        normals = np.zeros_like(cloud.points)

        for i, point in enumerate(cloud.points):
            _, indices = tree.query(point, k=30)
            neighbors = cloud.points[indices]
            centroid = np.mean(neighbors, axis=0)
            cov = (neighbors - centroid).T @ (neighbors - centroid)
            _, _, Vt = np.linalg.svd(cov)
            normals[i] = Vt[-1]

        return PointCloud(
            points=cloud.points,
            normals=normals,
            colors=cloud.colors,
            labels=cloud.labels,
            intensity=cloud.intensity,
            curvature=cloud.curvature
        )

    def _estimate_curvature(self, cloud: PointCloud) -> PointCloud:
        if cloud.normals is None:
            cloud = self._estimate_normals(cloud)

        tree = KDTree(cloud.points)
        curvature = np.zeros(cloud.num_points)

        for i, point in enumerate(cloud.points):
            _, indices = tree.query(point, k=30)
            neighbors = cloud.points[indices]
            centroid = np.mean(neighbors, axis=0)
            cov = (neighbors - centroid).T @ (neighbors - centroid)
            eigvals, _ = np.linalg.eigh(cov)
            eigvals = np.sort(eigvals)

            if eigvals[2] > 1e-6:
                curvature[i] = eigvals[0] / (eigvals[0] + eigvals[1] + eigvals[2])
            else:
                curvature[i] = 0

        return PointCloud(
            points=cloud.points,
            normals=cloud.normals,
            colors=cloud.colors,
            labels=cloud.labels,
            intensity=cloud.intensity,
            curvature=curvature
        )


class RansacPlaneSegmenter(BaseSegmenter):
    def __init__(
        self,
        max_iterations: int = 1000,
        distance_threshold: float = 0.01,
        min_inliers: int = 100,
        optimize_coefficients: bool = True
    ):
        self.max_iterations = max_iterations
        self.distance_threshold = distance_threshold
        self.min_inliers = min_inliers
        self.optimize_coefficients = optimize_coefficients

    def segment(self, cloud: PointCloud) -> List[Segment]:
        if cloud.num_points < 3:
            return []

        best_model = None
        best_inliers = []
        best_inlier_count = 0

        for _ in range(self.max_iterations):
            idx = np.random.choice(cloud.num_points, 3, replace=False)
            pts = cloud.points[idx]

            normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm

            d = -np.dot(normal, pts[0])

            distances = np.abs(np.dot(cloud.points, normal) + d)
            inliers = np.where(distances < self.distance_threshold)[0]

            if len(inliers) > best_inlier_count:
                best_inlier_count = len(inliers)
                best_inliers = inliers
                best_model = np.array([normal[0], normal[1], normal[2], d])

        if best_model is None or best_inlier_count < self.min_inliers:
            return []

        if self.optimize_coefficients:
            best_model = self._refine_model(cloud, best_inliers, best_model)

        plane_points = cloud.points[best_inliers]
        centroid = np.mean(plane_points, axis=0)
        min_coords = np.min(plane_points, axis=0)
        max_coords = np.max(plane_points, axis=0)

        return [Segment(
            segment_id=0,
            indices=best_inliers,
            centroid=centroid,
            bounding_box=(min_coords, max_coords),
            normal=best_model[:3],
            point_count=len(best_inliers)
        )]

    def _refine_model(self, cloud: PointCloud, inliers: np.ndarray, model: np.ndarray) -> np.ndarray:
        pts = cloud.points[inliers]
        centroid = np.mean(pts, axis=0)
        centered = pts - centroid
        _, _, Vt = np.linalg.svd(centered.T @ centered)
        normal = Vt[-1]
        d = -np.dot(normal, centroid)
        return np.array([normal[0], normal[1], normal[2], d])


class MultiPlaneSegmenter(BaseSegmenter):
    def __init__(
        self,
        plane_segmenter: RansacPlaneSegmenter,
        max_planes: int = 10,
        min_plane_points: int = 100
    ):
        self.plane_segmenter = plane_segmenter
        self.max_planes = max_planes
        self.min_plane_points = min_plane_points

    def segment(self, cloud: PointCloud) -> List[Segment]:
        segments = []
        remaining_cloud = cloud

        for i in range(self.max_planes):
            if remaining_cloud.num_points < self.min_plane_points:
                break

            plane_segments = self.plane_segmenter.segment(remaining_cloud)
            if not plane_segments:
                break

            plane = plane_segments[0]
            if plane.point_count < self.min_plane_points:
                break

            segments.append(plane)

            mask = np.ones(remaining_cloud.num_points, dtype=bool)
            mask[plane.indices] = False

            remaining_cloud = PointCloud(
                points=remaining_cloud.points[mask],
                normals=remaining_cloud.normals[mask] if remaining_cloud.normals is not None else None,
                colors=remaining_cloud.colors[mask] if remaining_cloud.colors is not None else None,
                labels=remaining_cloud.labels[mask] if remaining_cloud.labels is not None else None,
                intensity=remaining_cloud.intensity[mask] if remaining_cloud.intensity is not None else None,
                curvature=remaining_cloud.curvature[mask] if remaining_cloud.curvature is not None else None
            )

        return segments


class SACSegmentation:
    def __init__(
        self,
        model_type: str = "plane",
        method_type: str = "ransac",
        max_iterations: int = 1000,
        distance_threshold: float = 0.01,
        optimize_coefficients: bool = True
    ):
        self.model_type = model_type
        self.method_type = method_type
        self.max_iterations = max_iterations
        self.distance_threshold = distance_threshold
        self.optimize_coefficients = optimize_coefficients

    def segment_plane(self, cloud: PointCloud) -> Tuple[np.ndarray, np.ndarray]:
        return self.plane_segmenter.segment(cloud)

    def segment_sphere(self, cloud: PointCloud) -> Tuple[np.ndarray, np.ndarray]:
        return [], []

    def segment_cylinder(self, cloud: PointCloud) -> Tuple[np.ndarray, np.ndarray]:
        return [], []


class ProgressiveMorphologicalFilter:
    def __init__(
        self,
        max_window_size: int = 20,
        slope: float = 1.0,
        max_distance: float = 0.1,
        initial_distance: float = 0.5,
        cell_size: float = 0.5
    ):
        self.max_window_size = max_window_size
        self.slope = slope
        self.max_distance = max_distance
        self.initial_distance = initial_distance
        self.cell_size = cell_size

    def filter(self, cloud: PointCloud) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([]), np.array([])


class SupervoxelSegmentation:
    def __init__(
        self,
        voxel_resolution: float = 0.01,
        seed_resolution: float = 0.1,
        color_importance: float = 0.2,
        spatial_importance: float = 0.4,
        normal_importance: float = 1.0
    ):
        self.voxel_resolution = voxel_resolution
        self.seed_resolution = seed_resolution
        self.color_importance = color_importance
        self.spatial_importance = spatial_importance
        self.normal_importance = normal_importance

    def segment(self, cloud: PointCloud) -> List[Segment]:
        return []


class DeepLearningSegmenter(BaseSegmenter):
    def __init__(
        self,
        model_path: str = "segmentation_model.onnx",
        num_classes: int = 10,
        input_size: Tuple[int, int, int] = (4096, 3),
        device: str = "cpu"
    ):
        self.model_path = model_path
        self.num_classes = num_classes
        self.input_size = input_size
        self.device = device
        self.model = None

    def segment(self, cloud: PointCloud) -> List[Segment]:
        return []


class SegmentationMetrics:
    @staticmethod
    def iou(seg1: np.ndarray, seg2: np.ndarray, num_classes: int) -> np.ndarray:
        ious = []
        for c in range(num_classes):
            mask1 = seg1 == c
            mask2 = seg2 == c
            intersection = np.logical_and(mask1, mask2).sum()
            union = np.logical_or(mask1, mask2).sum()
            iou = intersection / (union + 1e-6)
            ious.append(iou)
        return np.array(ious)

    @staticmethod
    def miou(seg1: np.ndarray, seg2: np.ndarray, num_classes: int) -> float:
        ious = SegmentationMetrics.iou(seg1, seg2, num_classes)
        return float(np.nanmean(ious))

    @staticmethod
    def boundary_iou(seg1: np.ndarray, seg2: np.ndarray) -> float:
        return 0.0

    @staticmethod
    def under_segmentation_error(ground_truth: np.ndarray, predicted: np.ndarray) -> float:
        return 0.0

    @staticmethod
    def achievable_segmentation_accuracy(ground_truth: np.ndarray, predicted: np.ndarray) -> float:
        return 0.0


class SegmentationVisualizer:
    @staticmethod
    def draw_segments(
        cloud: PointCloud,
        segments: List[Segment],
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if not segments:
            return vis

        all_points = np.vstack([cloud.points[s.indices] for s in segments])
        min_coords = np.min(all_points[:, :2], axis=0)
        max_coords = np.max(all_points[:, :2], axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1

        scale = min(image_size[0], image_size[1]) * 0.8 / max(range_coords)

        colors = np.random.randint(0, 255, (len(segments), 3))

        for i, segment in enumerate(segments):
            points = cloud.points[segment.indices]
            x = ((points[:, 0] - min_coords[0]) * scale + image_size[0] * 0.1).astype(int)
            y = (image_size[1] - (points[:, 1] - min_coords[1]) * scale - image_size[1] * 0.1).astype(int)

            color = colors[i].tolist()
            for px, py in zip(x, y):
                if 0 <= px < image_size[0] and 0 <= py < image_size[1]:
                    vis[py, px] = color

        return vis

    @staticmethod
    def draw_segment_3d(
        cloud: PointCloud,
        segments: List[Segment],
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        all_points = np.vstack([cloud.points[s.indices] for s in segments]) if segments else np.array([[0,0,0]])
        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1

        scale = min(image_size[0], image_size[1]) * 0.8 / max(range_coords)

        colors = np.random.randint(0, 255, (len(segments), 3))

        for i, segment in enumerate(segments):
            points = cloud.points[segment.indices]
            x = ((points[:, 0] - min_coords[0]) * scale + image_size[0] * 0.1).astype(int)
            y = (image_size[1] - (points[:, 1] - min_coords[1]) * scale - image_size[1] * 0.1).astype(int)
            z = points[:, 2]

            color = colors[i].tolist()
            for px, py, pz in zip(x, y, z):
                if 0 <= px < image_size[0] and 0 <= py < image_size[1]:
                    intensity = int((pz - min_coords[2]) / (max_coords[2] - min_coords[2] + 1e-6) * 255)
                    vis[py, px] = (min(color[0], intensity), min(color[1], intensity), min(color[2], intensity))

        return vis


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
