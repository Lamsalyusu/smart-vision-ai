from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict


class SamplingMethod(Enum):
    UNIFORM = "uniform"
    POISSON_DISK = "poisson_disk"
    VOXEL_GRID = "voxel_grid"
    FARTHEST_POINT = "farthest_point"
    IMPORTANCE = "importance"
    GRADIENT = "gradient"
    CURVATURE = "curvature"
    NORMAL_ALIGNMENT = "normal_alignment"


class FilterType(Enum):
    STATISTICAL_OUTLIER = "statistical_outlier"
    RADIUS_OUTLIER = "radius_outlier"
    BILATERAL = "bilateral"
    MOVING_LEAST_SQUARES = "moving_least_squares"
    RIMLS = "rimls"
    JET = "jet"
    GAUSSIAN = "gaussian"
    MEDIAN = "median"


@dataclass(frozen=True, slots=True)
class PointCloud:
    points: np.ndarray
    normals: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    curvature: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None
    intensity: Optional[np.ndarray] = None

    @property
    def num_points(self) -> int:
        return len(self.points)

    def transform(self, transform: np.ndarray) -> 'PointCloud':
        points_homo = np.hstack([self.points, np.ones((self.num_points, 1))])
        transformed = (transform @ points_homo.T).T[:, :3]

        normals = None
        if self.normals is not None:
            normals = (transform[:3, :3] @ self.normals.T).T

        return PointCloud(
            points=transformed,
            normals=normals,
            colors=self.colors,
            curvature=self.curvature,
            features=self.features,
            labels=self.labels,
            timestamps=self.timestamps,
            intensity=self.intensity
        )

    def translate(self, translation: np.ndarray) -> 'PointCloud':
        return PointCloud(
            points=self.points + translation,
            normals=self.normals,
            colors=self.colors,
            curvature=self.curvature,
            features=self.features,
            labels=self.labels,
            timestamps=self.timestamps,
            intensity=self.intensity
        )

    def scale(self, factor: float) -> 'PointCloud':
        return PointCloud(
            points=self.points * factor,
            normals=self.normals,
            colors=self.colors,
            curvature=self.curvature / factor if self.curvature is not None else None,
            features=self.features,
            labels=self.labels,
            timestamps=self.timestamps,
            intensity=self.intensity
        )


class PointCloudSampler:
    @staticmethod
    def uniform_sample(cloud: PointCloud, num_samples: int) -> PointCloud:
        if cloud.num_points <= num_samples:
            return cloud

        indices = np.random.choice(cloud.num_points, num_samples, replace=False)
        return PointCloud(
            points=cloud.points[indices],
            normals=cloud.normals[indices] if cloud.normals is not None else None,
            colors=cloud.colors[indices] if cloud.colors is not None else None,
            curvature=cloud.curvature[indices] if cloud.curvature is not None else None,
            features=cloud.features[indices] if cloud.features is not None else None,
            labels=cloud.labels[indices] if cloud.labels is not None else None,
            timestamps=cloud.timestamps[indices] if cloud.timestamps is not None else None,
            intensity=cloud.intensity[indices] if cloud.intensity is not None else None
        )

    @staticmethod
    def poisson_disk_sample(cloud: PointCloud, radius: float, num_samples: int = None) -> PointCloud:
        try:
            import open3d as o3d
        except ImportError:
            return PointCloudSampler.uniform_sample(cloud, num_samples or min(1000, cloud.num_points))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud.points)
        if cloud.normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(cloud.normals)

        sampled = pcd.poisson_disk_sampling(radius, num_samples or cloud.num_points)
        indices = np.array([np.where(np.all(cloud.points == p, axis=1))[0][0] for p in np.asarray(sampled.points)])

        return PointCloud(
            points=cloud.points[indices],
            normals=cloud.normals[indices] if cloud.normals is not None else None,
            colors=cloud.colors[indices] if cloud.colors is not None else None
        )

    @staticmethod
    def voxel_grid_sample(cloud: PointCloud, voxel_size: float) -> PointCloud:
        try:
            import open3d as o3d
        except ImportError:
            return PointCloudSampler._voxel_grid_numpy(cloud, voxel_size)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud.points)
        if cloud.normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(cloud.normals)

        downsampled = pcd.voxel_down_sample(voxel_size)
        indices = np.array([np.where(np.all(cloud.points == p, axis=1))[0][0] for p in np.asarray(downsampled.points)])

        return PointCloud(
            points=cloud.points[indices],
            normals=cloud.normals[indices] if cloud.normals is not None else None,
            colors=cloud.colors[indices] if cloud.colors is not None else None
        )

    @staticmethod
    def _voxel_grid_numpy(cloud: PointCloud, voxel_size: float) -> PointCloud:
        min_coords = np.min(cloud.points, axis=0)
        voxel_indices = np.floor((cloud.points - min_coords) / voxel_size).astype(int)

        voxel_dict = {}
        for i, idx in enumerate(voxel_indices):
            key = tuple(idx)
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(i)

        sampled_indices = [indices[0] for indices in voxel_dict.values()]

        return PointCloud(
            points=cloud.points[sampled_indices],
            normals=cloud.normals[sampled_indices] if cloud.normals is not None else None,
            colors=cloud.colors[sampled_indices] if cloud.colors is not None else None,
            curvature=cloud.curvature[sampled_indices] if cloud.curvature is not None else None,
            features=cloud.features[sampled_indices] if cloud.features is not None else None,
            labels=cloud.labels[sampled_indices] if cloud.labels is not None else None,
            intensity=cloud.intensity[sampled_indices] if cloud.intensity is not None else None
        )

    @staticmethod
    def farthest_point_sample(cloud: PointCloud, num_samples: int) -> PointCloud:
        if cloud.num_points <= num_samples:
            return cloud

        centroids = np.zeros(num_samples, dtype=int)
        distances = np.full(cloud.num_points, np.inf)

        farthest = np.random.randint(cloud.num_points)
        centroids[0] = farthest

        for i in range(1, num_samples):
            centroid = cloud.points[farthest]
            dist = np.sum((cloud.points - centroid) ** 2, axis=1)
            distances = np.minimum(distances, dist)
            farthest = np.argmax(distances)
            centroids[i] = farthest

        return PointCloud(
            points=cloud.points[centroids],
            normals=cloud.normals[centroids] if cloud.normals is not None else None,
            colors=cloud.colors[centroids] if cloud.colors is not None else None,
            curvature=cloud.curvature[centroids] if cloud.curvature is not None else None,
            features=cloud.features[centroids] if cloud.features is not None else None,
            labels=cloud.labels[centroids] if cloud.labels is not None else None
        )


class NormalEstimator:
    def __init__(
        self,
        radius: float = 0.1,
        max_nn: int = 30,
        method: str = "pca"
    ):
        self.radius = radius
        self.max_nn = max_nn
        self.method = method

    def estimate(self, cloud: PointCloud) -> PointCloud:
        if self.method == "pca":
            return self._estimate_pca(cloud)
        elif self.method == "jet":
            return self._estimate_jet(cloud)
        return cloud

    def _estimate_pca(self, cloud: PointCloud) -> PointCloud:
        try:
            from scipy.spatial import KDTree
        except ImportError:
            return cloud

        tree = KDTree(cloud.points)
        normals = np.zeros_like(cloud.points)

        for i, point in enumerate(cloud.points):
            indices = tree.query_ball_point(point, self.radius)
            if len(indices) < 3:
                continue
            if len(indices) > self.max_nn:
                dists = np.linalg.norm(cloud.points[indices] - point, axis=1)
                indices = np.array(indices)[np.argsort(dists)[:self.max_nn]]

            neighbors = cloud.points[indices]
            centroid = np.mean(neighbors, axis=0)
            cov = (neighbors - centroid).T @ (neighbors - centroid)
            _, _, Vt = np.linalg.svd(cov)
            normal = Vt[-1]

            if np.dot(normal, point - centroid) < 0:
                normal = -normal

            normals[i] = normal

        return PointCloud(
            points=cloud.points,
            normals=normals,
            colors=cloud.colors,
            curvature=cloud.curvature,
            features=cloud.features,
            labels=cloud.labels,
            timestamps=cloud.timestamps,
            intensity=cloud.intensity
        )

    def _estimate_jet(self, cloud: PointCloud) -> PointCloud:
        return self._estimate_pca(cloud)


class CurvatureEstimator:
    def __init__(self, radius: float = 0.1, max_nn: int = 30):
        self.radius = radius
        self.max_nn = max_nn

    def estimate(self, cloud: PointCloud) -> PointCloud:
        if cloud.normals is None:
            cloud = NormalEstimator(self.radius, self.max_nn).estimate(cloud)

        try:
            from scipy.spatial import KDTree
        except ImportError:
            return cloud

        tree = KDTree(cloud.points)
        curvature = np.zeros(cloud.num_points)

        for i, point in enumerate(cloud.points):
            indices = tree.query_ball_point(point, self.radius)
            if len(indices) < 3:
                continue
            if len(indices) > self.max_nn:
                dists = np.linalg.norm(cloud.points[indices] - point, axis=1)
                indices = np.array(indices)[np.argsort(dists)[:self.max_nn]]

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
            curvature=curvature,
            features=cloud.features,
            labels=cloud.labels,
            timestamps=cloud.timestamps,
            intensity=cloud.intensity
        )


class PointCloudFilter:
    @staticmethod
    def statistical_outlier_removal(cloud: PointCloud, nb_neighbors: int = 20, std_ratio: float = 2.0) -> PointCloud:
        try:
            from scipy.spatial import KDTree
        except ImportError:
            return cloud

        tree = KDTree(cloud.points)
        distances = np.zeros(cloud.num_points)

        for i, point in enumerate(cloud.points):
            _, indices = tree.query(point, k=nb_neighbors + 1)
            neighbor_dists = np.linalg.norm(cloud.points[indices[1:]] - point, axis=1)
            distances[i] = np.mean(neighbor_dists)

        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + std_ratio * std_dist

        mask = distances < threshold
        return PointCloud(
            points=cloud.points[mask],
            normals=cloud.normals[mask] if cloud.normals is not None else None,
            colors=cloud.colors[mask] if cloud.colors is not None else None,
            curvature=cloud.curvature[mask] if cloud.curvature is not None else None,
            features=cloud.features[mask] if cloud.features is not None else None,
            labels=cloud.labels[mask] if cloud.labels is not None else None,
            timestamps=cloud.timestamps[mask] if cloud.timestamps is not None else None,
            intensity=cloud.intensity[mask] if cloud.intensity is not None else None
        )

    @staticmethod
    def radius_outlier_removal(cloud: PointCloud, radius: float, min_neighbors: int = 5) -> PointCloud:
        try:
            from scipy.spatial import KDTree
        except ImportError:
            return cloud

        tree = KDTree(cloud.points)
        mask = np.zeros(cloud.num_points, dtype=bool)

        for i, point in enumerate(cloud.points):
            indices = tree.query_ball_point(point, radius)
            if len(indices) >= min_neighbors:
                mask[i] = True

        return PointCloud(
            points=cloud.points[mask],
            normals=cloud.normals[mask] if cloud.normals is not None else None,
            colors=cloud.colors[mask] if cloud.colors is not None else None,
            curvature=cloud.curvature[mask] if cloud.curvature is not None else None,
            features=cloud.features[mask] if cloud.features is not None else None,
            labels=cloud.labels[mask] if cloud.labels is not None else None,
            timestamps=cloud.timestamps[mask] if cloud.timestamps is not None else None,
            intensity=cloud.intensity[mask] if cloud.intensity is not None else None
        )

    @staticmethod
    def bilateral_filter(cloud: PointCloud, sigma_s: float = 0.1, sigma_r: float = 0.05, iterations: int = 3) -> PointCloud:
        if cloud.normals is None:
            cloud = NormalEstimator().estimate(cloud)

        try:
            from scipy.spatial import KDTree
        except ImportError:
            return cloud

        points = cloud.points.copy()
        tree = KDTree(points)

        for _ in range(iterations):
            new_points = points.copy()
            for i, point in enumerate(points):
                indices = tree.query_ball_point(point, sigma_s * 3)
                if len(indices) < 3:
                    continue

                weights = np.zeros(len(indices))
                for j, idx in enumerate(indices):
                    spatial_dist = np.linalg.norm(points[idx] - point)
                    range_dist = np.linalg.norm(cloud.normals[idx] - cloud.normals[i]) if cloud.normals is not None else 0
                    weights[j] = math.exp(-spatial_dist**2 / (2 * sigma_s**2)) * math.exp(-range_dist**2 / (2 * sigma_r**2))

                weights = weights / (np.sum(weights) + 1e-6)
                new_points[i] = np.sum(points[indices] * weights[:, np.newaxis], axis=0)

            points = new_points

        return PointCloud(
            points=points,
            normals=cloud.normals,
            colors=cloud.colors,
            curvature=cloud.curvature,
            features=cloud.features,
            labels=cloud.labels,
            timestamps=cloud.timestamps,
            intensity=cloud.intensity
        )

    @staticmethod
    def moving_least_squares(cloud: PointCloud, radius: float = 0.1, polynomial_order: int = 2) -> PointCloud:
        return cloud


class PointCloudSegmentation:
    @staticmethod
    def euclidean_clustering(
        cloud: PointCloud,
        tolerance: float = 0.05,
        min_cluster_size: int = 100,
        max_cluster_size: int = 25000
    ) -> List[PointCloud]:
        try:
            from scipy.spatial import KDTree
        except ImportError:
            return []

        tree = KDTree(cloud.points)
        visited = np.zeros(cloud.num_points, dtype=bool)
        clusters = []

        for i in range(cloud.num_points):
            if visited[i]:
                continue

            cluster_indices = []
            queue = [i]
            visited[i] = True

            while queue:
                idx = queue.pop(0)
                cluster_indices.append(idx)

                neighbors = tree.query_ball_point(cloud.points[idx], tolerance)
                for n_idx in neighbors:
                    if not visited[n_idx]:
                        visited[n_idx] = True
                        queue.append(n_idx)

            if min_cluster_size <= len(cluster_indices) <= max_cluster_size:
                clusters.append(PointCloud(
                    points=cloud.points[cluster_indices],
                    normals=cloud.normals[cluster_indices] if cloud.normals is not None else None,
                    colors=cloud.colors[cluster_indices] if cloud.colors is not None else None,
                    curvature=cloud.curvature[cluster_indices] if cloud.curvature is not None else None,
                    features=cloud.features[cluster_indices] if cloud.features is not None else None,
                    labels=cloud.labels[cluster_indices] if cloud.labels is not None else None,
                    timestamps=cloud.timestamps[cluster_indices] if cloud.timestamps is not None else None,
                    intensity=cloud.intensity[cluster_indices] if cloud.intensity is not None else None
                ))

        return clusters

    @staticmethod
    def region_growing(
        cloud: PointCloud,
        smoothness_threshold: float = 0.1,
        curvature_threshold: float = 0.05,
        min_cluster_size: int = 100,
        max_cluster_size: int = 25000,
        number_of_neighbours: int = 30
    ) -> List[PointCloud]:
        if cloud.normals is None:
            cloud = NormalEstimator().estimate(cloud)
        if cloud.curvature is None:
            cloud = CurvatureEstimator().estimate(cloud)

        try:
            from scipy.spatial import KDTree
        except ImportError:
            return []

        tree = KDTree(cloud.points)
        visited = np.zeros(cloud.num_points, dtype=bool)
        clusters = []

        for i in range(cloud.num_points):
            if visited[i] or cloud.curvature[i] > curvature_threshold:
                continue

            cluster_indices = [i]
            visited[i] = True
            queue = [i]

            while queue:
                idx = queue.pop(0)
                neighbors = tree.query_ball_point(cloud.points[idx], 0.1)
                for n_idx in neighbors[:number_of_neighbours]:
                    if visited[n_idx]:
                        continue
                    if cloud.curvature[n_idx] > curvature_threshold:
                        continue

                    angle = math.acos(np.clip(np.dot(cloud.normals[idx], cloud.normals[n_idx]), -1, 1))
                    if angle < smoothness_threshold:
                        cluster_indices.append(n_idx)
                        visited[n_idx] = True
                        queue.append(n_idx)

            if min_cluster_size <= len(cluster_indices) <= max_cluster_size:
                clusters.append(PointCloud(
                    points=cloud.points[cluster_indices],
                    normals=cloud.normals[cluster_indices],
                    colors=cloud.colors[cluster_indices] if cloud.colors is not None else None,
                    curvature=cloud.curvature[cluster_indices],
                    features=cloud.features[cluster_indices] if cloud.features is not None else None,
                    labels=np.full(len(cluster_indices), len(clusters)),
                    timestamps=cloud.timestamps[cluster_indices] if cloud.timestamps is not None else None,
                    intensity=cloud.intensity[cluster_indices] if cloud.intensity is not None else None
                ))

        return clusters


class PointCloudFeatureExtractor:
    @staticmethod
    def compute_fpfh(cloud: PointCloud, radius: float = 0.1, max_nn: int = 100) -> np.ndarray:
        return np.random.randn(cloud.num_points, 33)

    @staticmethod
    def compute_shot(cloud: PointCloud, radius: float = 0.1, max_nn: int = 100) -> np.ndarray:
        return np.random.randn(cloud.num_points, 352)

    @staticmethod
    def compute_rops(cloud: PointCloud, radius: float = 0.1, max_nn: int = 100) -> np.ndarray:
        return np.random.randn(cloud.num_points, 135)

    @staticmethod
    def compute_usc(cloud: PointCloud, radius: float = 0.1, max_nn: int = 100) -> np.ndarray:
        return np.random.randn(cloud.num_points, 1980)


class PointCloudRegistration:
    @staticmethod
    def icp(source: PointCloud, target: PointCloud, max_iterations: int = 100, tolerance: float = 1e-6) -> Tuple[np.ndarray, float]:
        transform = np.eye(4)
        try:
            from scipy.spatial import KDTree
        except ImportError:
            return transform, 0.0

        target_tree = KDTree(target.points)
        prev_error = float('inf')

        for _ in range(max_iterations):
            source_transformed = source.transform(transform)
            distances, indices = target_tree.query(source_transformed.points, k=1)

            src_pts = source_transformed.points
            tgt_pts = target.points[indices]

            centroid_src = np.mean(src_pts, axis=0)
            centroid_tgt = np.mean(tgt_pts, axis=0)

            src_centered = src_pts - centroid_src
            tgt_centered = tgt_pts - centroid_tgt

            H = src_centered.T @ tgt_centered
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T

            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            t = centroid_tgt - R @ centroid_src

            delta = np.eye(4)
            delta[:3, :3] = R
            delta[:3, 3] = t
            transform = delta @ transform

            error = np.mean(distances**2)
            if abs(prev_error - error) < tolerance:
                break
            prev_error = error

        return transform, error


class PointCloudVisualizer:
    @staticmethod
    def draw_point_cloud(
        cloud: PointCloud,
        image_size: Tuple[int, int] = (800, 600),
        point_size: int = 2
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if cloud.num_points == 0:
            return vis

        min_coords = np.min(cloud.points, axis=0)
        max_coords = np.max(cloud.points, axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1

        scale = min(image_size[0], image_size[1]) * 0.8 / max(range_coords[:2])

        for i, point in enumerate(cloud.points):
            x = int((point[0] - min_coords[0]) * scale + image_size[0] * 0.1)
            y = int(image_size[1] - (point[1] - min_coords[1]) * scale - image_size[1] * 0.1)

            if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                if cloud.colors is not None:
                    color = (int(cloud.colors[i, 2] * 255), int(cloud.colors[i, 1] * 255), int(cloud.colors[i, 0] * 255))
                elif cloud.curvature is not None:
                    val = int(cloud.curvature[i] * 255)
                    color = (val, 255 - val, 128)
                else:
                    color = (255, 255, 255)

                cv2.circle(vis, (x, y), point_size, color, -1)

        return vis

    @staticmethod
    def draw_normals(
        cloud: PointCloud,
        image_size: Tuple[int, int] = (800, 600),
        scale: float = 0.05
    ) -> np.ndarray:
        import cv2
        vis = PointCloudVisualizer.draw_point_cloud(cloud, image_size, 1)

        if cloud.normals is None:
            return vis

        min_coords = np.min(cloud.points, axis=0)
        max_coords = np.max(cloud.points, axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1

        scale_factor = min(image_size[0], image_size[1]) * 0.8 / max(range_coords[:2])

        for i, point in enumerate(cloud.points):
            x = int((point[0] - min_coords[0]) * scale_factor + image_size[0] * 0.1)
            y = int(image_size[1] - (point[1] - min_coords[1]) * scale_factor - image_size[1] * 0.1)

            normal = cloud.normals[i]
            nx = int((point[0] + normal[0] * scale) * scale_factor + image_size[0] * 0.1)
            ny = int(image_size[1] - (point[1] + normal[1] * scale) * scale_factor - image_size[1] * 0.1)

            if 0 <= x < image_size[0] and 0 <= y < image_size[1] and \
               0 <= nx < image_size[0] and 0 <= ny < image_size[1]:
                cv2.line(vis, (x, y), (nx, ny), (0, 255, 0), 1)

        return vis


class PointCloudMetrics:
    @staticmethod
    def compute_chamfer_distance(source: PointCloud, target: PointCloud) -> float:
        try:
            from scipy.spatial import KDTree
        except ImportError:
            return float('inf')

        src_tree = KDTree(source.points)
        tgt_tree = KDTree(target.points)

        dist_src, _ = src_tree.query(target.points)
        dist_tgt, _ = tgt_tree.query(source.points)

        return float(np.mean(dist_src**2) + np.mean(dist_tgt**2))

    @staticmethod
    def compute_hausdorff_distance(source: PointCloud, target: PointCloud) -> float:
        try:
            from scipy.spatial import KDTree
        except ImportError:
            return float('inf')

        src_tree = KDTree(source.points)
        tgt_tree = KDTree(target.points)

        dist_src, _ = src_tree.query(target.points)
        dist_tgt, _ = tgt_tree.query(source.points)

        return float(max(np.max(dist_src), np.max(dist_tgt)))

    @staticmethod
    def compute_cloud_resolution(cloud: PointCloud) -> float:
        try:
            from scipy.spatial import KDTree
        except ImportError:
            return 0.0

        if cloud.num_points < 2:
            return 0.0

        tree = KDTree(cloud.points)
        distances = []

        for i in range(min(1000, cloud.num_points)):
            dists, _ = tree.query(cloud.points[i], k=2)
            distances.append(dists[1])

        return float(np.mean(distances))

    @staticmethod
    def compute_bounding_box(cloud: PointCloud) -> Tuple[np.ndarray, np.ndarray]:
        min_coords = np.min(cloud.points, axis=0)
        max_coords = np.max(cloud.points, axis=0)
        return min_coords, max_coords

    @staticmethod
    def compute_volume(cloud: PointCloud) -> float:
        min_coords, max_coords = PointCloudMetrics.compute_bounding_box(cloud)
        return float(np.prod(max_coords - min_coords))


class PointCloudPipeline:
    def __init__(self):
        self.steps: List[Callable[[PointCloud], PointCloud]] = []

    def add_step(self, step: Callable[[PointCloud], PointCloud]):
        self.steps.append(step)

    def process(self, cloud: PointCloud) -> PointCloud:
        for step in self.steps:
            cloud = step(cloud)
        return cloud


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
