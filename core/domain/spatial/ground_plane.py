from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PlaneEstimate:
    normal: tuple[float, float, float]
    offset: float
    inlier_indices: tuple[int, ...]
    root_mean_squared_error: float


def fit_ground_plane_ransac(
    points: Sequence[Sequence[float]],
    distance_threshold: float,
    iterations: int = 200,
    min_inliers: int = 3,
    random_seed: int = 0,
) -> PlaneEstimate:
    """Fit a metric plane robustly and orient its normal toward positive Y."""
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or cloud.shape[0] < 3:
        raise ValueError("point cloud must contain at least three 3D points")
    if not np.isfinite(cloud).all():
        raise ValueError("point cloud coordinates must be finite")
    if not math.isfinite(distance_threshold) or distance_threshold <= 0:
        raise ValueError("distance threshold must be finite and positive")
    if iterations < 1 or min_inliers < 3 or min_inliers > len(cloud):
        raise ValueError("iterations and minimum inlier count are invalid")

    random = np.random.default_rng(random_seed)
    best_inliers = np.empty(0, dtype=np.int64)
    best_error = float("inf")
    for _ in range(iterations):
        sample = cloud[random.choice(len(cloud), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        normal_length = np.linalg.norm(normal)
        if normal_length <= np.finfo(np.float64).eps:
            continue
        normal /= normal_length
        offset = -float(np.dot(normal, sample[0]))
        distances = np.abs(cloud @ normal + offset)
        inliers = np.flatnonzero(distances <= distance_threshold)
        error = float(distances[inliers].sum())
        if len(inliers) > len(best_inliers) or (
            len(inliers) == len(best_inliers) and error < best_error
        ):
            best_inliers = inliers
            best_error = error

    if len(best_inliers) < min_inliers:
        raise ValueError("RANSAC could not find a plane with enough inliers")

    inlier_points = cloud[best_inliers]
    center = inlier_points.mean(axis=0)
    _, _, right_vectors = np.linalg.svd(inlier_points - center, full_matrices=False)
    normal = right_vectors[-1]
    normal_length = np.linalg.norm(normal)
    if normal_length <= np.finfo(np.float64).eps:
        raise ValueError("inlier points do not define a stable plane")
    normal /= normal_length
    if normal[1] < 0 or (abs(normal[1]) <= 1e-12 and normal[np.argmax(np.abs(normal))] < 0):
        normal = -normal
    offset = -float(np.dot(normal, center))

    residuals = np.abs(cloud @ normal + offset)
    refined_inliers = np.flatnonzero(residuals <= distance_threshold)
    if len(refined_inliers) < min_inliers:
        raise ValueError("refined plane has too few inliers")
    rmse = float(np.sqrt(np.mean(residuals[refined_inliers] ** 2)))
    return PlaneEstimate(
        normal=tuple(float(value) for value in normal),
        offset=offset,
        inlier_indices=tuple(int(index) for index in refined_inliers),
        root_mean_squared_error=rmse,
    )