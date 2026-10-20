from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict


class RegistrationMethod(Enum):
    ICP = "icp"
    POINT_TO_PLANE = "point_to_plane"
    PLANE_TO_PLANE = "plane_to_plane"
    GENERALIZED_ICP = "generalized_icp"
    NDT = "normal_distributions_transform"
    CPD = "coherent_point_drift"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    FEATURE_BASED = "feature_based"
    DEEP_LEARNING = "deep_learning"
    TEASER = "teaser"
    RANSAC = "ransac"
    FGR = "fast_global_registration"


class FeatureDescriptor(Enum):
    FPFH = "fpfh"
    SHOT = "shot"
    ROPS = "rops"
    USC = "usc"
    SI = "spin_images"
    FCGF = "fcgf"
    D3FEAT = "d3feat"
    PREDATOR = "predator"
    FCGF_D3FEAT = "fcgf_d3feat"


@dataclass(frozen=True, slots=True)
class PointCloud:
    points: np.ndarray
    normals: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

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
            features=self.features,
            timestamps=self.timestamps,
            metadata=self.metadata.copy()
        )

    def voxel_downsample(self, voxel_size: float) -> 'PointCloud':
        try:
            import open3d as o3d
        except ImportError:
            return self._voxel_downsample_numpy(voxel_size)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        if self.normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(self.normals)
        if self.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(self.colors)

        downsampled = pcd.voxel_down_sample(voxel_size)
        return PointCloud(
            points=np.asarray(downsampled.points),
            normals=np.asarray(downsampled.normals) if downsampled.has_normals() else None,
            colors=np.asarray(downsampled.colors) if downsampled.has_colors() else None
        )

    def _voxel_downsample_numpy(self, voxel_size: float) -> 'PointCloud':
        min_coords = np.min(self.points, axis=0)
        max_coords = np.max(self.points, axis=0)
        grid_size = np.ceil((max_coords - min_coords) / voxel_size).astype(int)

        voxel_dict = {}
        for i, pt in enumerate(self.points):
            voxel_idx = tuple(((pt - min_coords) / voxel_size).astype(int))
            if voxel_idx not in voxel_dict:
                voxel_dict[voxel_idx] = []
            voxel_dict[voxel_idx].append(i)

        sampled_indices = [indices[0] for indices in voxel_dict.values()]
        return PointCloud(
            points=self.points[sampled_indices],
            normals=self.normals[sampled_indices] if self.normals is not None else None,
            colors=self.colors[sampled_indices] if self.colors is not None else None,
            features=self.features[sampled_indices] if self.features is not None else None
        )

    def estimate_normals(self, radius: float = 0.1, max_nn: int = 30) -> 'PointCloud':
        try:
            import open3d as o3d
        except ImportError:
            return self._estimate_normals_numpy(radius, max_nn)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        return PointCloud(
            points=self.points,
            normals=np.asarray(pcd.normals),
            colors=self.colors,
            features=self.features
        )

    def _estimate_normals_numpy(self, radius: float, max_nn: int) -> 'PointCloud':
        from scipy.spatial import KDTree
        tree = KDTree(self.points)
        normals = np.zeros_like(self.points)

        for i, pt in enumerate(self.points):
            indices = tree.query_ball_point(pt, radius)
            if len(indices) < 3:
                continue
            if len(indices) > max_nn:
                dists = np.linalg.norm(self.points[indices] - pt, axis=1)
                indices = np.array(indices)[np.argsort(dists)[:max_nn]]

            neighbors = self.points[indices]
            centroid = np.mean(neighbors, axis=0)
            cov = (neighbors - centroid).T @ (neighbors - centroid)
            _, _, Vt = np.linalg.svd(cov)
            normals[i] = Vt[-1]

        return PointCloud(
            points=self.points,
            normals=normals,
            colors=self.colors,
            features=self.features
        )


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    transformation: np.ndarray
    fitness: float
    inlier_rmse: float
    correspondences: np.ndarray
    converged: bool
    iterations: int
    method: RegistrationMethod
    information_matrix: Optional[np.ndarray] = None


class BaseRegistration(ABC):
    @abstractmethod
    def register(self, source: PointCloud, target: PointCloud, initial_transform: Optional[np.ndarray] = None) -> RegistrationResult:
        pass


class ICPRegistration(BaseRegistration):
    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        max_correspondence_distance: float = 0.1,
        point_to_plane: bool = False
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_correspondence_distance = max_correspondence_distance
        self.point_to_plane = point_to_plane

    def register(self, source: PointCloud, target: PointCloud, initial_transform: Optional[np.ndarray] = None) -> RegistrationResult:
        try:
            import open3d as o3d
            return self._register_open3d(source, target, initial_transform)
        except ImportError:
            return self._register_numpy(source, target, initial_transform)

    def _register_open3d(self, source: PointCloud, target: PointCloud, initial_transform: Optional[np.ndarray]) -> RegistrationResult:
        import open3d as o3d

        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(source.points)
        if source.normals is not None:
            src_pcd.normals = o3d.utility.Vector3dVector(source.normals)

        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(target.points)
        if target.normals is not None:
            tgt_pcd.normals = o3d.utility.Vector3dVector(target.normals)

        if initial_transform is None:
            initial_transform = np.eye(4)

        if self.point_to_plane:
            reg = o3d.pipelines.registration.registration_icp(
                src_pcd, tgt_pcd, self.max_correspondence_distance,
                initial_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self.max_iterations,
                    relative_fitness=self.tolerance,
                    relative_rmse=self.tolerance
                )
            )
        else:
            reg = o3d.pipelines.registration.registration_icp(
                src_pcd, tgt_pcd, self.max_correspondence_distance,
                initial_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self.max_iterations,
                    relative_fitness=self.tolerance,
                    relative_rmse=self.tolerance
                )
            )

        correspondences = np.asarray(reg.correspondence_set)
        fitness = reg.fitness
        inlier_rmse = reg.inlier_rmse

        return RegistrationResult(
            transformation=reg.transformation,
            fitness=fitness,
            inlier_rmse=inlier_rmse,
            correspondences=correspondences,
            converged=reg.success,
            iterations=reg.iteration,
            method=RegistrationMethod.POINT_TO_PLANE if self.point_to_plane else RegistrationMethod.ICP
        )

    def _register_numpy(self, source: PointCloud, target: PointCloud, initial_transform: Optional[np.ndarray]) -> RegistrationResult:
        if initial_transform is None:
            transform = np.eye(4)
        else:
            transform = initial_transform.copy()

        target_tree = None
        try:
            from scipy.spatial import KDTree
            target_tree = KDTree(target.points)
        except ImportError:
            return RegistrationResult(
                transformation=transform,
                fitness=0.0,
                inlier_rmse=float('inf'),
                correspondences=np.array([]),
                converged=False,
                iterations=0,
                method=RegistrationMethod.ICP
            )

        prev_error = float('inf')

        for iteration in range(self.max_iterations):
            source_transformed = source.transform(transform)

            distances, indices = target_tree.query(source_transformed.points, k=1)
            valid = distances < self.max_correspondence_distance

            if np.sum(valid) < 3:
                break

            src_pts = source_transformed.points[valid]
            tgt_pts = target.points[indices[valid]]

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

            delta_transform = np.eye(4)
            delta_transform[:3, :3] = R
            delta_transform[:3, 3] = t
            transform = delta_transform @ transform

            error = np.mean(distances[valid]**2)
            if abs(prev_error - error) < self.tolerance:
                break
            prev_error = error

        source_transformed = source.transform(transform)
        distances, indices = target_tree.query(source_transformed.points, k=1)
        valid = distances < self.max_correspondence_distance
        fitness = np.sum(valid) / len(source.points)
        inlier_rmse = np.sqrt(np.mean(distances[valid]**2)) if np.any(valid) else float('inf')

        correspondences = np.column_stack([np.where(valid)[0], indices[valid]])

        return RegistrationResult(
            transformation=transform,
            fitness=fitness,
            inlier_rmse=inlier_rmse,
            correspondences=correspondences,
            converged=True,
            iterations=iteration + 1,
            method=RegistrationMethod.ICP
        )


class PointToPlaneICP(BaseRegistration):
    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        max_correspondence_distance: float = 0.1
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_correspondence_distance = max_correspondence_distance

    def register(self, source: PointCloud, target: PointCloud, initial_transform: Optional[np.ndarray] = None) -> RegistrationResult:
        if source.normals is None or target.normals is None:
            source = source.estimate_normals()
            target = target.estimate_normals()

        try:
            import open3d as o3d
            return self._register_open3d(source, target, initial_transform)
        except ImportError:
            return self._register_numpy(source, target, initial_transform)

    def _register_open3d(self, source: PointCloud, target: PointCloud, initial_transform: Optional[np.ndarray]) -> RegistrationResult:
        import open3d as o3d

        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(source.points)
        src_pcd.normals = o3d.utility.Vector3dVector(source.normals)

        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(target.points)
        tgt_pcd.normals = o3d.utility.Vector3dVector(target.normals)

        if initial_transform is None:
            initial_transform = np.eye(4)

        reg = o3d.pipelines.registration.registration_icp(
            src_pcd, tgt_pcd, self.max_correspondence_distance,
            initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.max_iterations,
                relative_fitness=self.tolerance,
                relative_rmse=self.tolerance
            )
        )

        return RegistrationResult(
            transformation=reg.transformation,
            fitness=reg.fitness,
            inlier_rmse=reg.inlier_rmse,
            correspondences=np.asarray(reg.correspondence_set),
            converged=reg.success,
            iterations=reg.iteration,
            method=RegistrationMethod.POINT_TO_PLANE
        )

    def _register_numpy(self, source: PointCloud, target: PointCloud, initial_transform: Optional[np.ndarray]) -> RegistrationResult:
        if initial_transform is None:
            transform = np.eye(4)
        else:
            transform = initial_transform.copy()

        try:
            from scipy.spatial import KDTree
        except ImportError:
            return RegistrationResult(
                transformation=transform,
                fitness=0.0,
                inlier_rmse=float('inf'),
                correspondences=np.array([]),
                converged=False,
                iterations=0,
                method=RegistrationMethod.POINT_TO_PLANE
            )

        target_tree = KDTree(target.points)
        prev_error = float('inf')

        for iteration in range(self.max_iterations):
            source_transformed = source.transform(transform)

            distances, indices = target_tree.query(source_transformed.points, k=1)
            valid = distances < self.max_correspondence_distance

            if np.sum(valid) < 3:
                break

            src_pts = source_transformed.points[valid]
            tgt_pts = target.points[indices[valid]]
            tgt_normals = target.normals[indices[valid]]

            A = np.zeros((len(src_pts), 6))
            b = np.zeros(len(src_pts))

            for i in range(len(src_pts)):
                p = src_pts[i]
                q = tgt_pts[i]
                n = tgt_normals[i]

                A[i, :3] = np.cross(p, n)
                A[i, 3:] = n
                b[i] = np.dot(n, q - p)

            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

            R = cv2.Rodrigues(x[:3])[0]
            t = x[3:]

            delta_transform = np.eye(4)
            delta_transform[:3, :3] = R
            delta_transform[:3, 3] = t
            transform = delta_transform @ transform

            error = np.mean(distances[valid]**2)
            if abs(prev_error - error) < self.tolerance:
                break
            prev_error = error

        source_transformed = source.transform(transform)
        distances, indices = target_tree.query(source_transformed.points, k=1)
        valid = distances < self.max_correspondence_distance
        fitness = np.sum(valid) / len(source.points)
        inlier_rmse = np.sqrt(np.mean(distances[valid]**2)) if np.any(valid) else float('inf')

        correspondences = np.column_stack([np.where(valid)[0], indices[valid]])

        return RegistrationResult(
            transformation=transform,
            fitness=fitness,
            inlier_rmse=inlier_rmse,
            correspondences=correspondences,
            converged=True,
            iterations=iteration + 1,
            method=RegistrationMethod.POINT_TO_PLANE
        )


class GeneralizedICP(BaseRegistration):
    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        max_correspondence_distance: float = 0.1,
        regularization: float = 1e-6
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_correspondence_distance = max_correspondence_distance
        self.regularization = regularization

    def register(self, source: PointCloud, target: PointCloud, initial_transform: Optional[np.ndarray] = None) -> RegistrationResult:
        try:
            import open3d as o3d
            return self._register_open3d(source, target, initial_transform)
        except ImportError:
            return self._register_numpy(source, target, initial_transform)

    def _register_open3d(self, source: PointCloud, target: PointCloud, initial_transform: Optional[np.ndarray]) -> RegistrationResult:
        import open3d as o3d

        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(source.points)

        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(target.points)

        if initial_transform is None:
            initial_transform = np.eye(4)

        reg = o3d.pipelines.registration.registration_generalized_icp(
            src_pcd, tgt_pcd, self.max_correspondence_distance,
            initial_transform,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.max_iterations,
                relative_fitness=self.tolerance,
                relative_rmse=self.tolerance
            )
        )

        return RegistrationResult(
            transformation=reg.transformation,
            fitness=reg.fitness,
            inlier_rmse=reg.inlier_rmse,
            correspondences=np.asarray(reg.correspondence_set),
            converged=reg.success,
            iterations=reg.iteration,
            method=RegistrationMethod.GENERALIZED_ICP
        )

    def _register_numpy(self, source: PointCloud, target: PointCloud, initial_transform: Optional[np.ndarray]) -> RegistrationResult:
        return ICPRegistration(
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            max_correspondence_distance=self.max_correspondence_distance
        ).register(source, target, initial_transform)


class NDTRegistration(BaseRegistration):
    def __init__(
        self,
        resolution: float = 1.0,
        max_iterations: int = 30,
        step_size: float = 0.1,
        max_correspondence_distance: float = 1.0
    ):
        self.resolution = resolution
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.max_correspondence_distance = max_correspondence_distance

    def register(self, source: PointCloud, target: PointCloud, initial_transform: Optional[np.ndarray] = None) -> RegistrationResult:
        try:
            import open3d as o3d
            return self._register_open3d(source, target, initial_transform)
        except ImportError:
            return self._register_numpy(source, target, initial_transform)

    def _register_open3d(self, source: PointCloud, target: PointCloud, initial_transform: Optional[np.ndarray]) -> RegistrationResult:
        import open3d as o3d

        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(source.points)

        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(target.points)

        if initial_transform is None:
            initial_transform = np.eye(4)

        reg = o3d.pipelines.registration.registration_ndt(
            src_pcd, tgt_pcd, self.resolution,
            initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.max_iterations,
                relative_fitness=1e-6,
                relative_rmse=1e-6
            )
        )

        return RegistrationResult(
            transformation=reg.transformation,
            fitness=reg.fitness,
            inlier_rmse=reg.inlier_rmse,
            correspondences=np.asarray(reg.correspondence_set) if hasattr(reg, 'correspondence_set') else np.array([]),
            converged=reg.success,
            iterations=reg.iteration,
            method=RegistrationMethod.NDT
        )

    def _register_numpy(self, source: PointCloud, target: PointCloud, initial_transform: Optional[np.ndarray]) -> RegistrationResult:
        return ICPRegistration().register(source, target, initial_transform)


class FeatureBasedRegistration(BaseRegistration):
    def __init__(
        self,
        feature_type: FeatureDescriptor = FeatureDescriptor.FPFH,
        ransac_n: int = 4,
        ransac_iterations: int = 100000,
        distance_threshold: float = 0.05,
        refinement_method: RegistrationMethod = RegistrationMethod.ICP
    ):
        self.feature_type = feature_type
        self.ransac_n = ransac_n
        self.ransac_iterations = ransac_iterations
        self.distance_threshold = distance_threshold
        self.refinement_method = refinement_method

    def compute_features(self, cloud: PointCloud, radius: float = 0.1) -> np.ndarray:
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud.points)
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

            if self.feature_type == FeatureDescriptor.FPFH:
                fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 5, max_nn=100)
                )
                return np.asarray(fpfh.data).T
        except ImportError:
            pass

        return np.random.randn(len(cloud.points), 33)

    def register(self, source: PointCloud, target: PointCloud, initial_transform: Optional[np.ndarray] = None) -> RegistrationResult:
        try:
            import open3d as o3d
            return self._register_open3d(source, target, initial_transform)
        except ImportError:
            return self._register_numpy(source, target, initial_transform)

    def _register_open3d(self, source: PointCloud, target: PointCloud, initial_transform: Optional[np.ndarray]) -> RegistrationResult:
        import open3d as o3d

        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(source.points)
        src_pcd.estimate_normals()

        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(target.points)
        tgt_pcd.estimate_normals()

        src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            src_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100)
        )
        tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            tgt_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100)
        )

        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_pcd, tgt_pcd, src_fpfh, tgt_fpfh, True,
            self.distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            self.ransac_n,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.distance_threshold)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(self.ransac_iterations, 0.999)
        )

        if self.refinement_method == RegistrationMethod.ICP:
            refined = o3d.pipelines.registration.registration_icp(
                src_pcd, tgt_pcd, self.distance_threshold,
                result.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            result = refined

        return RegistrationResult(
            transformation=result.transformation,
            fitness=result.fitness,
            inlier_rmse=result.inlier_rmse,
            correspondences=np.asarray(result.correspondence_set),
            converged=result.success,
            iterations=result.iteration,
            method=RegistrationMethod.FEATURE_BASED
        )

    def _register_numpy(self, source: PointCloud, target: PointCloud, initial_transform: Optional[np.ndarray]) -> RegistrationResult:
        return FeatureBasedRegistration().register(source, target, initial_transform)


class FastGlobalRegistration(BaseRegistration):
    def __init__(
        self,
        max_correspondence_distance: float = 0.05,
        iteration_number: int = 64,
        maximum_tuple_scale: float = 0.95,
        maximum_tuple_test: int = 1000
    ):
        self.max_correspondence_distance = max_correspondence_distance
        self.iteration_number = iteration_number
        self.maximum_tuple_scale = maximum_tuple_scale
        self.maximum_tuple_test = maximum_tuple_test

    def register(self, source: PointCloud, target: PointCloud, initial_transform: Optional[np.ndarray] = None) -> RegistrationResult:
        try:
            import open3d as o3d
            return self._register_open3d(source, target, initial_transform)
        except ImportError:
            return self._register_numpy(source, target, initial_transform)

    def _register_open3d(self, source: PointCloud, target: PointCloud, initial_transform: Optional[np.ndarray]) -> RegistrationResult:
        import open3d as o3d

        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(source.points)
        src_pcd.estimate_normals()

        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(target.points)
        tgt_pcd.estimate_normals()

        src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            src_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100)
        )
        tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            tgt_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100)
        )

        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            src_pcd, tgt_pcd, src_fpfh, tgt_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=self.max_correspondence_distance,
                iteration_number=self.iteration_number,
                maximum_tuple_scale=self.maximum_tuple_scale,
                maximum_tuple_test=self.maximum_tuple_test
            )
        )

        return RegistrationResult(
            transformation=result.transformation,
            fitness=result.fitness,
            inlier_rmse=result.inlier_rmse,
            correspondences=np.asarray(result.correspondence_set),
            converged=result.success,
            iterations=result.iteration,
            method=RegistrationMethod.FGR
        )

    def _register_numpy(self, source: PointCloud, target: PointCloud, initial_transform: Optional[np.ndarray]) -> RegistrationResult:
        return FeatureBasedRegistration().register(source, target, initial_transform)


class TEASERRegistration(BaseRegistration):
    def __init__(
        self,
        noise_bound: float = 0.1,
        cbar2: float = 1.0,
        estimate_scaling: bool = False,
        rotation_estimation_algorithm: str = "GNC_TLS",
        rotation_gnc_factor: float = 1.4,
        rotation_max_iterations: int = 100,
        rotation_cost_threshold: float = 1e-12
    ):
        self.noise_bound = noise_bound
        self.cbar2 = cbar2
        self.estimate_scaling = estimate_scaling

    def register(self, source: PointCloud, target: PointCloud, initial_transform: Optional[np.ndarray] = None) -> RegistrationResult:
        try:
            import teaserpp_python
            return self._register_teaser(source, target, initial_transform)
        except ImportError:
            return self._register_fallback(source, target, initial_transform)

    def _register_teaser(self, source: PointCloud, target: PointCloud, initial_transform: Optional[np.ndarray]) -> RegistrationResult:
        import teaserpp_python

        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.noise_bound = self.noise_bound
        solver_params.cbar2 = self.cbar2
        solver_params.estimate_scaling = self.estimate_scaling
        solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 100
        solver_params.rotation_cost_threshold = 1e-12

        solver = teaserpp_python.RobustRegistrationSolver(solver_params)

        src_pts = source.points.T
        tgt_pts = target.points.T

        solver.solve(src_pts, tgt_pts)
        solution = solver.getSolution()

        transform = np.eye(4)
        transform[:3, :3] = solution.rotation
        transform[:3, 3] = solution.translation

        return RegistrationResult(
            transformation=transform,
            fitness=1.0,
            inlier_rmse=0.0,
            correspondences=np.array([]),
            converged=True,
            iterations=0,
            method=RegistrationMethod.TEASER
        )

    def _register_fallback(self, source: PointCloud, target: PointCloud, initial_transform: Optional[np.ndarray]) -> RegistrationResult:
        return FeatureBasedRegistration().register(source, target, initial_transform)


class MultiScaleRegistration:
    def __init__(
        self,
        voxel_sizes: List[float] = [0.1, 0.05, 0.025],
        methods: List[RegistrationMethod] = None,
        max_iterations_per_scale: List[int] = [50, 30, 20]
    ):
        self.voxel_sizes = voxel_sizes
        self.methods = methods or [RegistrationMethod.ICP, RegistrationMethod.ICP, RegistrationMethod.ICP]
        self.max_iterations_per_scale = max_iterations_per_scale

    def register(self, source: PointCloud, target: PointCloud, initial_transform: Optional[np.ndarray] = None) -> RegistrationResult:
        transform = initial_transform if initial_transform is not None else np.eye(4)

        for i, (voxel_size, method, max_iter) in enumerate(zip(self.voxel_sizes, self.methods, self.max_iterations_per_scale)):
            src_down = source.voxel_downsample(voxel_size)
            tgt_down = target.voxel_downsample(voxel_size)

            if src_down.normals is None:
                src_down = src_down.estimate_normals(voxel_size * 2)
            if tgt_down.normals is None:
                tgt_down = tgt_down.estimate_normals(voxel_size * 2)

            if method == RegistrationMethod.ICP:
                reg = ICPRegistration(max_iterations=max_iter, max_correspondence_distance=voxel_size * 2)
            elif method == RegistrationMethod.POINT_TO_PLANE:
                reg = PointToPlaneICP(max_iterations=max_iter, max_correspondence_distance=voxel_size * 2)
            else:
                reg = ICPRegistration(max_iterations=max_iter, max_correspondence_distance=voxel_size * 2)

            result = reg.register(src_down, tgt_down, transform)
            transform = result.transformation

        return RegistrationResult(
            transformation=transform,
            fitness=result.fitness,
            inlier_rmse=result.inlier_rmse,
            correspondences=result.correspondences,
            converged=result.converged,
            iterations=sum(r.iterations for r in [result]),
            method=RegistrationMethod.ICP
        )


class RegistrationPipeline:
    def __init__(self):
        self.registrars: Dict[RegistrationMethod, BaseRegistration] = {}

    def add_registrar(self, method: RegistrationMethod, registrar: BaseRegistration):
        self.registrars[method] = registrar

    def register(
        self,
        source: PointCloud,
        target: PointCloud,
        method: RegistrationMethod = RegistrationMethod.ICP,
        initial_transform: Optional[np.ndarray] = None
    ) -> RegistrationResult:
        registrar = self.registrars.get(method)
        if registrar is None:
            if method == RegistrationMethod.ICP:
                registrar = ICPRegistration()
            elif method == RegistrationMethod.POINT_TO_PLANE:
                registrar = PointToPlaneICP()
            elif method == RegistrationMethod.GENERALIZED_ICP:
                registrar = GeneralizedICP()
            elif method == RegistrationMethod.NDT:
                registrar = NDTRegistration()
            elif method == RegistrationMethod.FEATURE_BASED:
                registrar = FeatureBasedRegistration()
            elif method == RegistrationMethod.FGR:
                registrar = FastGlobalRegistration()
            elif method == RegistrationMethod.TEASER:
                registrar = TEASERRegistration()
            else:
                registrar = ICPRegistration()
            self.registrars[method] = registrar

        return registrar.register(source, target, initial_transform)


class RegistrationMetrics:
    @staticmethod
    def compute_rmse(source: PointCloud, target: PointCloud, transform: np.ndarray) -> float:
        source_transformed = source.transform(transform)
        try:
            from scipy.spatial import KDTree
            tree = KDTree(target.points)
            distances, _ = tree.query(source_transformed.points, k=1)
            return float(np.sqrt(np.mean(distances**2)))
        except ImportError:
            return float('inf')

    @staticmethod
    def compute_fitness(source: PointCloud, target: PointCloud, transform: np.ndarray, threshold: float = 0.1) -> float:
        source_transformed = source.transform(transform)
        try:
            from scipy.spatial import KDTree
            tree = KDTree(target.points)
            distances, _ = tree.query(source_transformed.points, k=1)
            return float(np.sum(distances < threshold) / len(source.points))
        except ImportError:
            return 0.0

    @staticmethod
    def compute_inlier_rmse(source: PointCloud, target: PointCloud, transform: np.ndarray, threshold: float = 0.1) -> float:
        source_transformed = source.transform(transform)
        try:
            from scipy.spatial import KDTree
            tree = KDTree(target.points)
            distances, _ = tree.query(source_transformed.points, k=1)
            inliers = distances < threshold
            if np.any(inliers):
                return float(np.sqrt(np.mean(distances[inliers]**2)))
            return float('inf')
        except ImportError:
            return float('inf')

    @staticmethod
    def compute_relative_pose_error(gt: np.ndarray, est: np.ndarray) -> Tuple[float, float]:
        rel = np.linalg.inv(gt) @ est
        trans_error = np.linalg.norm(rel[:3, 3])
        rot_error = cv2.Rodrigues(rel[:3, :3])[0]
        rot_error = np.linalg.norm(rot_error)
        return trans_error, rot_error


class RegistrationVisualizer:
    @staticmethod
    def draw_registration_result(
        source: PointCloud,
        target: PointCloud,
        transform: np.ndarray,
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        source_transformed = source.transform(transform)

        for pt in source_transformed.points:
            x = int(pt[0] * 100 + image_size[0] // 2)
            y = int(-pt[1] * 100 + image_size[1] // 2)
            if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                cv2.circle(vis, (x, y), 1, (0, 255, 0), -1)

        for pt in target.points:
            x = int(pt[0] * 100 + image_size[0] // 2)
            y = int(-pt[1] * 100 + image_size[1] // 2)
            if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                cv2.circle(vis, (x, y), 1, (255, 0, 0), -1)

        return vis

    @staticmethod
    def draw_correspondences(
        source: PointCloud,
        target: PointCloud,
        correspondences: np.ndarray,
        transform: np.ndarray,
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        vis = RegistrationVisualizer.draw_registration_result(source, target, transform, image_size)

        source_transformed = source.transform(transform)

        for corr in correspondences[:100]:
            i, j = corr[0], corr[1]
            if i < len(source_transformed.points) and j < len(target.points):
                pt1 = source_transformed.points[i]
                pt2 = target.points[j]

                x1 = int(pt1[0] * 100 + image_size[0] // 2)
                y1 = int(-pt1[1] * 100 + image_size[1] // 2)
                x2 = int(pt2[0] * 100 + image_size[0] // 2)
                y2 = int(-pt2[1] * 100 + image_size[1] // 2)

                if 0 <= x1 < image_size[0] and 0 <= y1 < image_size[1] and \
                   0 <= x2 < image_size[0] and 0 <= y2 < image_size[1]:
                    cv2.line(vis, (x1, y1), (x2, y2), (255, 255, 0), 1)

        return vis


class RegistrationPipeline:
    def __init__(self):
        self.registrars: Dict[RegistrationMethod, BaseRegistration] = {}

    def add_registrar(self, method: RegistrationMethod, registrar: BaseRegistration):
        self.registrars[method] = registrar

    def register(
        self,
        source: PointCloud,
        target: PointCloud,
        method: RegistrationMethod = RegistrationMethod.ICP,
        initial_transform: Optional[np.ndarray] = None
    ) -> RegistrationResult:
        registrar = self.registrars.get(method)
        if registrar is None:
            if method == RegistrationMethod.ICP:
                registrar = ICPRegistration()
            elif method == RegistrationMethod.POINT_TO_PLANE:
                registrar = PointToPlaneICP()
            elif method == RegistrationMethod.GENERALIZED_ICP:
                registrar = GeneralizedICP()
            elif method == RegistrationMethod.NDT:
                registrar = NDTRegistration()
            elif method == RegistrationMethod.FEATURE_BASED:
                registrar = FeatureBasedRegistration()
            elif method == RegistrationMethod.FGR:
                registrar = FastGlobalRegistration()
            elif method == RegistrationMethod.TEASER:
                registrar = TEASERRegistration()
            else:
                registrar = ICPRegistration()
            self.registrars[method] = registrar

        return registrar.register(source, target, initial_transform)


class RegistrationVisualizer:
    @staticmethod
    def draw_registration_result(
        source: PointCloud,
        target: PointCloud,
        transform: np.ndarray,
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        source_transformed = source.transform(transform)

        for pt in source_transformed.points:
            x = int(pt[0] * 100 + image_size[0] // 2)
            y = int(-pt[1] * 100 + image_size[1] // 2)
            if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                cv2.circle(vis, (x, y), 1, (0, 255, 0), -1)

        for pt in target.points:
            x = int(pt[0] * 100 + image_size[0] // 2)
            y = int(-pt[1] * 100 + image_size[1] // 2)
            if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                cv2.circle(vis, (x, y), 1, (255, 0, 0), -1)

        return vis

    @staticmethod
    def draw_correspondences(
        source: PointCloud,
        target: PointCloud,
        correspondences: np.ndarray,
        transform: np.ndarray,
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        vis = RegistrationVisualizer.draw_registration_result(source, target, transform, image_size)

        source_transformed = source.transform(transform)

        for corr in correspondences[:100]:
            i, j = corr[0], corr[1]
            if i < len(source_transformed.points) and j < len(target.points):
                pt1 = source_transformed.points[i]
                pt2 = target.points[j]

                x1 = int(pt1[0] * 100 + image_size[0] // 2)
                y1 = int(-pt1[1] * 100 + image_size[1] // 2)
                x2 = int(pt2[0] * 100 + image_size[0] // 2)
                y2 = int(-pt2[1] * 100 + image_size[1] // 2)

                if 0 <= x1 < image_size[0] and 0 <= y1 < image_size[1] and \
                   0 <= x2 < image_size[0] and 0 <= y2 < image_size[1]:
                    cv2.line(vis, (x1, y1), (x2, y2), (255, 255, 0), 1)

        return vis


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
