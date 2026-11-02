from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict
import math
import heapq


@dataclass(frozen=True, slots=True)
class PointCloud:
    points: np.ndarray
    colors: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    intensities: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    timestamp: float = 0.0

    def __post_init__(self):
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("Points must be Nx3")
        if self.colors is not None and self.colors.shape != self.points.shape:
            raise ValueError("Colors must match points shape")
        if self.normals is not None and self.normals.shape != self.points.shape:
            raise ValueError("Normals must match points shape")

    @property
    def size(self) -> int:
        return self.points.shape[0]

    @property
    def centroid(self) -> np.ndarray:
        return np.mean(self.points, axis=0)

    @property
    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        return np.min(self.points, axis=0), np.max(self.points, axis=0)

    def transform(self, R: np.ndarray, t: np.ndarray) -> "PointCloud":
        new_points = (R @ self.points.T).T + t
        new_normals = (R @ self.normals.T).T if self.normals is not None else None
        return PointCloud(new_points, self.colors, new_normals, self.intensities, self.labels, self.timestamp)

    def crop(self, min_bound: np.ndarray, max_bound: np.ndarray) -> "PointCloud":
        mask = np.all(self.points >= min_bound, axis=1) & np.all(self.points <= max_bound, axis=1)
        return PointCloud(
            self.points[mask],
            self.colors[mask] if self.colors is not None else None,
            self.normals[mask] if self.normals is not None else None,
            self.intensities[mask] if self.intensities is not None else None,
            self.labels[mask] if self.labels is not None else None,
            self.timestamp,
        )

    def voxel_downsample(self, voxel_size: float) -> "PointCloud":
        coords = np.floor(self.points / voxel_size).astype(np.int32)
        _, indices = np.unique(coords, axis=0, return_index=True)
        return PointCloud(
            self.points[indices],
            self.colors[indices] if self.colors is not None else None,
            self.normals[indices] if self.normals is not None else None,
            self.intensities[indices] if self.intensities is not None else None,
            self.labels[indices] if self.labels is not None else None,
            self.timestamp,
        )


class KDTree:
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


class PointCloudProcessor:
    @staticmethod
    def estimate_normals(
        cloud: PointCloud,
        k: int = 30,
        radius: float = 0.1,
    ) -> PointCloud:
        kdtree = KDTree(cloud.points)
        normals = np.zeros_like(cloud.points)

        for i, p in enumerate(cloud.points):
            neighbors = kdtree.query_knn(p, k)
            if len(neighbors) < 3:
                normals[i] = np.array([0, 0, 1])
                continue

            neighbor_pts = cloud.points[[idx for idx, _ in neighbors]]
            centroid = np.mean(neighbor_pts, axis=0)
            centered = neighbor_pts - centroid
            cov = centered.T @ centered / len(neighbor_pts)
            eigvals, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, 0]
            if normal[2] < 0:
                normal = -normal
            normals[i] = normal

        return PointCloud(
            cloud.points, cloud.colors, normals, cloud.intensities, cloud.labels, cloud.timestamp
        )

    @staticmethod
    def compute_fpfh(
        cloud: PointCloud,
        radius: float = 0.05,
    ) -> np.ndarray:
        if cloud.normals is None:
            cloud = PointCloudProcessor.estimate_normals(cloud)

        kdtree = KDTree(cloud.points)
        fpfh = np.zeros((cloud.size, 33))

        for i, p in enumerate(cloud.points):
            neighbors = kdtree.query_radius(p, radius)
            if len(neighbors) < 3:
                continue

            hist = np.zeros(33)
            for j in neighbors:
                if j == i:
                    continue
                diff = cloud.points[j] - p
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    continue
                diff = diff / dist

                n1 = cloud.normals[i]
                n2 = cloud.normals[j]

                u = n1
                v = n2
                w = np.cross(u, v)
                w_norm = np.linalg.norm(w)
                if w_norm < 1e-6:
                    continue
                w = w / w_norm

                alpha = np.arctan2(np.dot(v, w), np.dot(v, u))
                phi = np.arctan2(np.dot(u, diff), np.dot(diff, w))
                theta = np.arctan2(np.dot(n2, w), np.dot(n2, u))

                idx1 = int((alpha + np.pi) / (2 * np.pi) * 11)
                idx2 = int((phi + np.pi) / (2 * np.pi) * 11)
                idx3 = int((theta + np.pi) / (2 * np.pi) * 11)
                idx = idx1 * 11 * 11 + idx2 * 11 + idx3
                hist[idx] += 1

            if hist.sum() > 0:
                hist = hist / hist.sum()
            fpfh[i] = hist

        return fpfh

    @staticmethod
    def ransac_plane(
        cloud: PointCloud,
        distance_threshold: float = 0.01,
        max_iterations: int = 1000,
    ) -> tuple[np.ndarray, np.ndarray]:
        best_plane = None
        best_inliers = None
        best_count = 0

        n = cloud.size
        if n < 3:
            return np.array([0, 0, 1, 0]), np.array([])

        for _ in range(max_iterations):
            idx = np.random.choice(n, 3, replace=False)
            pts = cloud.points[idx]

            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm

            d = -np.dot(normal, pts[0])
            distances = np.abs(cloud.points @ normal + d)
            inliers = np.where(distances < distance_threshold)[0]

            if len(inliers) > best_count:
                best_count = len(inliers)
                best_plane = np.array([normal[0], normal[1], normal[2], d])
                best_inliers = inliers

        return best_plane if best_plane is not None else np.array([0, 0, 1, 0]), best_inliers if best_inliers is not None else np.array([])

    @staticmethod
    def euclidean_cluster(
        cloud: PointCloud,
        tolerance: float = 0.02,
        min_size: int = 100,
        max_size: int = 100000,
    ) -> List[PointCloud]:
        kdtree = KDTree(cloud.points)
        visited = np.zeros(cloud.size, dtype=bool)
        clusters = []

        for i in range(cloud.size):
            if visited[i]:
                continue

            cluster_indices = []
            stack = [i]
            visited[i] = True

            while stack:
                idx = stack.pop()
                cluster_indices.append(idx)

                neighbors = kdtree.query_radius(cloud.points[idx], tolerance)
                for n_idx in neighbors:
                    if not visited[n_idx]:
                        visited[n_idx] = True
                        stack.append(n_idx)

            if min_size <= len(cluster_indices) <= max_size:
                clusters.append(PointCloud(
                    cloud.points[cluster_indices],
                    cloud.colors[cluster_indices] if cloud.colors is not None else None,
                    cloud.normals[cluster_indices] if cloud.normals is not None else None,
                    cloud.intensities[cluster_indices] if cloud.intensities is not None else None,
                    cloud.labels[cluster_indices] if cloud.labels is not None else None,
                    cloud.timestamp,
                ))

        return clusters

    @staticmethod
    def dbscan_cluster(
        cloud: PointCloud,
        eps: float = 0.05,
        min_samples: int = 10,
    ) -> List[PointCloud]:
        kdtree = KDTree(cloud.points)
        labels = np.full(cloud.size, -1, dtype=np.int32)
        cluster_id = 0

        for i in range(cloud.size):
            if labels[i] != -1:
                continue

            neighbors = kdtree.query_radius(cloud.points[i], eps)
            if len(neighbors) < min_samples:
                labels[i] = -2
                continue

            labels[i] = cluster_id
            seed_set = [n for n in neighbors if n != i]

            for q in seed_set:
                if labels[q] == -2:
                    labels[q] = cluster_id
                if labels[q] != -1:
                    continue

                labels[q] = cluster_id
                q_neighbors = kdtree.query_radius(cloud.points[q], eps)
                if len(q_neighbors) >= min_samples:
                    seed_set.extend([n for n in q_neighbors if labels[n] <= 0])

            cluster_id += 1

        clusters = []
        for cid in range(cluster_id):
            indices = np.where(labels == cid)[0]
            if len(indices) > 0:
                clusters.append(PointCloud(
                    cloud.points[indices],
                    cloud.colors[indices] if cloud.colors is not None else None,
                    cloud.normals[indices] if cloud.normals is not None else None,
                    cloud.intensities[indices] if cloud.intensities is not None else None,
                    cloud.labels[indices] if cloud.labels is not None else None,
                    cloud.timestamp,
                ))

        return clusters


class ICP:
    def __init__(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        max_correspondence_distance: float = 0.1,
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_correspondence_distance = max_correspondence_distance

    def align(
        self,
        source: PointCloud,
        target: PointCloud,
        initial_transform: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        R = np.eye(3)
        t = np.zeros(3)

        if initial_transform:
            R, t = initial_transform

        prev_error = float('inf')
        target_kdtree = KDTree(target.points)

        for iteration in range(self.max_iterations):
            transformed = (R @ source.points.T).T + t

            correspondences = []
            for i, p in enumerate(transformed):
                neighbors = target_kdtree.query_knn(p, 1)
                if neighbors:
                    idx, dist = neighbors[0]
                    if dist < self.max_correspondence_distance:
                        correspondences.append((i, idx, dist))

            if len(correspondences) < 3:
                break

            src_pts = source.points[[c[0] for c in correspondences]]
            tgt_pts = target.points[[c[1] for c in correspondences]]

            centroid_src = np.mean(src_pts, axis=0)
            centroid_tgt = np.mean(tgt_pts, axis=0)

            src_centered = src_pts - centroid_src
            tgt_centered = tgt_pts - centroid_tgt

            H = src_centered.T @ tgt_centered
            U, _, Vt = np.linalg.svd(H)
            R_new = Vt.T @ U.T

            if np.linalg.det(R_new) < 0:
                Vt[-1, :] *= -1
                R_new = Vt.T @ U.T

            t_new = centroid_tgt - R_new @ centroid_src

            R = R_new @ R
            t = R_new @ t + t_new

            error = np.mean([c[2] for c in correspondences])
            if abs(prev_error - error) < self.tolerance:
                break
            prev_error = error

        final_error = np.mean([c[2] for c in correspondences]) if correspondences else 0
        return R, t, final_error


class NDT:
    def __init__(
        self,
        grid_size: float = 1.0,
        max_iterations: int = 30,
        tolerance: float = 1e-4,
        step_size: float = 0.1,
    ):
        self.grid_size = grid_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.step_size = step_size
        self.grid: Dict[tuple, Dict] = {}

    def _build_grid(self, cloud: PointCloud):
        self.grid.clear()
        coords = np.floor(cloud.points / self.grid_size).astype(np.int32)

        for i, coord in enumerate(coords):
            key = tuple(coord)
            if key not in self.grid:
                self.grid[key] = {'points': [], 'indices': []}
            self.grid[key]['points'].append(cloud.points[i])
            self.grid[key]['indices'].append(i)

        for key in self.grid:
            pts = np.array(self.grid[key]['points'])
            if len(pts) > 1:
                centroid = np.mean(pts, axis=0)
                centered = pts - centroid
                cov = centered.T @ centered / len(pts)
                self.grid[key]['centroid'] = centroid
                self.grid[key]['cov'] = cov
                self.grid[key]['inv_cov'] = np.linalg.inv(cov + 1e-6 * np.eye(3))
            else:
                self.grid[key]['centroid'] = pts[0]
                self.grid[key]['cov'] = np.eye(3) * 1e-3
                self.grid[key]['inv_cov'] = np.eye(3) * 1e3

    def align(
        self,
        source: PointCloud,
        target: PointCloud,
        initial_transform: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        self._build_grid(target)

        R = np.eye(3)
        t = np.zeros(3)

        if initial_transform:
            R, t = initial_transform

        for iteration in range(self.max_iterations):
            gradient = np.zeros(6)
            hessian = np.zeros((6, 6))
            count = 0

            for p in source.points:
                p_trans = R @ p + t
                coord = tuple(np.floor(p_trans / self.grid_size).astype(np.int32))

                if coord not in self.grid:
                    continue

                cell = self.grid[coord]
                diff = p_trans - cell['centroid']
                inv_cov = cell['inv_cov']

                J = np.zeros((3, 6))
                J[:3, :3] = -self._skew(p)
                J[:3, 3:] = np.eye(3)

                g = J.T @ inv_cov @ diff
                H = J.T @ inv_cov @ J

                gradient += g
                hessian += H
                count += 1

            if count < 3:
                break

            try:
                dx = np.linalg.solve(hessian + 1e-6 * np.eye(6), -gradient)
            except np.linalg.LinAlgError:
                break

            if np.linalg.norm(dx) < self.tolerance:
                break

            omega = dx[:3]
            trans = dx[3:]

            theta = np.linalg.norm(omega)
            if theta < 1e-6:
                R_delta = np.eye(3) + self._skew(omega)
            else:
                axis = omega / theta
                K = self._skew(axis)
                R_delta = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K

            R = R_delta @ R
            t = R_delta @ t + trans

        error = 0
        for p in source.points:
            p_trans = R @ p + t
            coord = tuple(np.floor(p_trans / self.grid_size).astype(np.int32))
            if coord in self.grid:
                diff = p_trans - self.grid[coord]['centroid']
                error += diff @ self.grid[coord]['inv_cov'] @ diff

        return R, t, error / max(count, 1)

    def _skew(self, v: np.ndarray) -> np.ndarray:
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])


class PointCloudRegistration:
    def __init__(self):
        self.icp = ICP()
        self.ndt = NDT()

    def register_icp(self, source: PointCloud, target: PointCloud, **kwargs) -> tuple:
        return self.icp.align(source, target, **kwargs)

    def register_ndt(self, source: PointCloud, target: PointCloud, **kwargs) -> tuple:
        return self.ndt.align(source, target, **kwargs)

    def register_fpfh_ransac(
        self,
        source: PointCloud,
        target: PointCloud,
        distance_threshold: float = 0.05,
        max_iterations: int = 100000,
    ) -> tuple:
        src_fpfh = PointCloudProcessor.compute_fpfh(source)
        tgt_fpfh = PointCloudProcessor.compute_fpfh(target)

        matches = self._match_fpfh(src_fpfh, tgt_fpfh)

        if len(matches) < 4:
            return np.eye(3), np.zeros(3), 0

        best_R, best_t, best_score = None, None, -1

        for _ in range(min(max_iterations, len(matches) * 10)):
            idx = np.random.choice(len(matches), 4, replace=False)
            sample = [matches[i] for i in idx]

            src_pts = source.points[[s[0] for s in sample]]
            tgt_pts = target.points[[s[1] for s in sample]]

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

            inliers = 0
            for s_idx, t_idx in matches:
                p_trans = R @ source.points[s_idx] + t
                dist = np.linalg.norm(p_trans - target.points[t_idx])
                if dist < distance_threshold:
                    inliers += 1

            if inliers > best_score:
                best_score = inliers
                best_R, best_t = R, t

        return best_R if best_R is not None else np.eye(3), best_t if best_t is not None else np.zeros(3), best_score

    def _match_fpfh(self, src_fpfh: np.ndarray, tgt_fpfh: np.ndarray) -> List[tuple]:
        matches = []
        for i, src in enumerate(src_fpfh):
            dists = np.linalg.norm(tgt_fpfh - src, axis=1)
            best = np.argmin(dists)
            if dists[best] < 0.5:
                matches.append((i, best))
        return matches


class Octree:
    def __init__(
        self,
        min_bound: np.ndarray,
        max_bound: np.ndarray,
        max_depth: int = 10,
        min_points: int = 1,
    ):
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.max_depth = max_depth
        self.min_points = min_points
        self.root = self._Node(min_bound, max_bound, 0)
        self.points: Optional[np.ndarray] = None

    class _Node:
        def __init__(self, min_b: np.ndarray, max_b: np.ndarray, depth: int):
            self.min_bound = min_b
            self.max_bound = max_b
            self.depth = depth
            self.children: List[Optional["Octree._Node"]] = [None] * 8
            self.point_indices: List[int] = []
            self.is_leaf = True

    def build(self, points: np.ndarray):
        self.points = points
        self.root = self._Node(self.min_bound, self.max_bound, 0)
        for i in range(len(points)):
            self._insert(self.root, points[i], i)

    def _insert(self, node: _Node, point: np.ndarray, index: int):
        if not node.is_leaf:
            child_idx = self._get_child_index(node, point)
            if node.children[child_idx] is None:
                node.children[child_idx] = self._create_child(node, child_idx)
            self._insert(node.children[child_idx], point, index)
            return

        node.point_indices.append(index)

        if len(node.point_indices) > self.min_points and node.depth < self.max_depth:
            node.is_leaf = False
            for idx in node.point_indices:
                child_idx = self._get_child_index(node, self.points[idx])
                if node.children[child_idx] is None:
                    node.children[child_idx] = self._create_child(node, child_idx)
                self._insert(node.children[child_idx], self.points[idx], idx)
            node.point_indices = []

    def _create_child(self, node: _Node, child_idx: int) -> _Node:
        mid = (node.min_bound + node.max_bound) / 2
        min_b = node.min_bound.copy()
        max_b = node.max_bound.copy()

        if child_idx & 1:
            min_b[0] = mid[0]
        else:
            max_b[0] = mid[0]
        if child_idx & 2:
            min_b[1] = mid[1]
        else:
            max_b[1] = mid[1]
        if child_idx & 4:
            min_b[2] = mid[2]
        else:
            max_b[2] = mid[2]

        return self._Node(min_b, max_b, node.depth + 1)

    def _get_child_index(self, node: _Node, point: np.ndarray) -> int:
        mid = (node.min_bound + node.max_bound) / 2
        idx = 0
        if point[0] >= mid[0]:
            idx |= 1
        if point[1] >= mid[1]:
            idx |= 2
        if point[2] >= mid[2]:
            idx |= 4
        return idx

    def query_radius(self, center: np.ndarray, radius: float) -> List[int]:
        results = []
        self._query_radius(self.root, center, radius, results)
        return results

    def _query_radius(self, node: _Node, center: np.ndarray, radius: float, results: List[int]):
        if node is None:
            return

        closest = np.clip(center, node.min_bound, node.max_bound)
        dist = np.linalg.norm(center - closest)
        if dist > radius:
            return

        if node.is_leaf:
            for idx in node.point_indices:
                if np.linalg.norm(self.points[idx] - center) <= radius:
                    results.append(idx)
        else:
            for child in node.children:
                self._query_radius(child, center, radius, results)


class VoxelGrid:
    def __init__(self, voxel_size: float):
        self.voxel_size = voxel_size
        self.voxels: Dict[tuple, Dict] = {}

    def add_points(self, points: np.ndarray, colors: Optional[np.ndarray] = None):
        coords = np.floor(points / self.voxel_size).astype(np.int32)

        for i, coord in enumerate(coords):
            key = tuple(coord)
            if key not in self.voxels:
                self.voxels[key] = {'points': [], 'colors': []}
            self.voxels[key]['points'].append(points[i])
            if colors is not None:
                self.voxels[key]['colors'].append(colors[i])

    def get_downsampled(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        downsampled_points = []
        downsampled_colors = []

        for voxel in self.voxels.values():
            pts = np.array(voxel['points'])
            centroid = np.mean(pts, axis=0)
            downsampled_points.append(centroid)
            if voxel['colors']:
                cols = np.array(voxel['colors'])
                downsampled_colors.append(np.mean(cols, axis=0))

        return np.array(downsampled_points), np.array(downsampled_colors) if downsampled_colors else None