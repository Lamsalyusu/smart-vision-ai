from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _point_array(points: Sequence[Sequence[float]], name: str, minimum: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < minimum:
        raise ValueError(f"{name} must contain at least {minimum} 2D points")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return values


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    mean_distance = np.linalg.norm(points - center, axis=1).mean()
    if mean_distance <= np.finfo(np.float64).eps:
        raise ValueError("point configuration is degenerate")
    scale = np.sqrt(2.0) / mean_distance
    transform = np.array(
        [
            [scale, 0.0, -scale * center[0]],
            [0.0, scale, -scale * center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    normalized = (transform @ homogeneous.T).T
    return normalized[:, :2], transform


def estimate_homography(
    source_points: Sequence[Sequence[float]],
    target_points: Sequence[Sequence[float]],
) -> np.ndarray:
    """Estimate source-to-target planar mapping with normalized DLT."""
    source = _point_array(source_points, "source points", 4)
    target = _point_array(target_points, "target points", 4)
    if source.shape[0] != target.shape[0]:
        raise ValueError("source and target point counts must match")

    source_normalized, source_transform = _normalize_points(source)
    target_normalized, target_transform = _normalize_points(target)
    rows = []
    for (x, y), (u, v) in zip(source_normalized, target_normalized):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    design = np.asarray(rows, dtype=np.float64)
    if np.linalg.matrix_rank(design) < 8:
        raise ValueError("point correspondences do not determine a homography")

    _, _, right_vectors = np.linalg.svd(design, full_matrices=True)
    normalized_homography = right_vectors[-1].reshape(3, 3)
    homography = np.linalg.inv(target_transform) @ normalized_homography @ source_transform
    norm = np.linalg.norm(homography)
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("homography solution is degenerate")
    return homography / norm


def solve_zhang_intrinsics(homographies: Sequence[Sequence[Sequence[float]]]) -> np.ndarray:
    """Recover a 3x3 camera intrinsic matrix from planar-view homographies."""
    if len(homographies) < 3:
        raise ValueError("Zhang calibration requires at least three homographies")

    constraints = []
    for homography_values in homographies:
        homography = np.asarray(homography_values, dtype=np.float64)
        if homography.shape != (3, 3) or not np.isfinite(homography).all():
            raise ValueError("each homography must be a finite 3x3 matrix")

        def constraint(first_column: int, second_column: int) -> np.ndarray:
            first = homography[:, first_column]
            second = homography[:, second_column]
            return np.array(
                [
                    first[0] * second[0],
                    first[0] * second[1] + first[1] * second[0],
                    first[1] * second[1],
                    first[2] * second[0] + first[0] * second[2],
                    first[2] * second[1] + first[1] * second[2],
                    first[2] * second[2],
                ],
                dtype=np.float64,
            )

        first = constraint(0, 1)
        constraints.extend((first, constraint(0, 0) - constraint(1, 1)))

    design = np.vstack(constraints)
    if np.linalg.matrix_rank(design) < 5:
        raise ValueError("planar views do not sufficiently constrain camera intrinsics")
    _, _, right_vectors = np.linalg.svd(design, full_matrices=True)
    b = right_vectors[-1]
    if b[0] < 0:
        b = -b

    b11, b12, b22, b13, b23, b33 = b
    denominator = b11 * b22 - b12 * b12
    if b11 <= 0 or denominator <= 0:
        raise ValueError("calibration views produced non-physical intrinsics")

    center_y = (b12 * b13 - b11 * b23) / denominator
    scale = b33 - (b13 * b13 + center_y * (b12 * b13 - b11 * b23)) / b11
    if scale <= 0:
        raise ValueError("calibration views produced non-positive focal length")

    focal_x = np.sqrt(scale / b11)
    focal_y = np.sqrt(scale * b11 / denominator)
    skew = -b12 * focal_x * focal_x * focal_y / scale
    center_x = skew * center_y / focal_y - b13 * focal_x * focal_x / scale
    intrinsics = np.array(
        [[focal_x, skew, center_x], [0.0, focal_y, center_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if not np.isfinite(intrinsics).all():
        raise ValueError("calibration produced non-finite intrinsics")
    return intrinsics


def estimate_fundamental_matrix(
    first_points: Sequence[Sequence[float]],
    second_points: Sequence[Sequence[float]],
) -> np.ndarray:
    """Estimate a rank-two fundamental matrix using normalized 8-point DLT."""
    first = _point_array(first_points, "first-image points", 8)
    second = _point_array(second_points, "second-image points", 8)
    if first.shape[0] != second.shape[0]:
        raise ValueError("point correspondence counts must match")

    first_normalized, first_transform = _normalize_points(first)
    second_normalized, second_transform = _normalize_points(second)
    x1, y1 = first_normalized.T
    x2, y2 = second_normalized.T
    design = np.column_stack((x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones_like(x1)))
    if np.linalg.matrix_rank(design) < 8:
        raise ValueError("point correspondences do not determine a fundamental matrix")

    _, _, right_vectors = np.linalg.svd(design, full_matrices=True)
    normalized_fundamental = right_vectors[-1].reshape(3, 3)
    left, singular_values, right = np.linalg.svd(normalized_fundamental)
    singular_values[-1] = 0.0
    rank_two = left @ np.diag(singular_values) @ right
    fundamental = second_transform.T @ rank_two @ first_transform
    norm = np.linalg.norm(fundamental)
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("fundamental matrix solution is degenerate")
    return fundamental / norm


def symmetric_epipolar_distance(
    fundamental_matrix: Sequence[Sequence[float]],
    first_point: Sequence[float],
    second_point: Sequence[float],
) -> float:
    matrix = np.asarray(fundamental_matrix, dtype=np.float64)
    first = np.asarray(first_point, dtype=np.float64)
    second = np.asarray(second_point, dtype=np.float64)
    if matrix.shape != (3, 3) or first.shape != (2,) or second.shape != (2,):
        raise ValueError("expected a 3x3 matrix and two 2D points")
    if not np.isfinite(matrix).all() or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("epipolar inputs must be finite")

    first_h = np.append(first, 1.0)
    second_h = np.append(second, 1.0)
    line_in_second = matrix @ first_h
    line_in_first = matrix.T @ second_h
    error = float(second_h @ line_in_second)
    norm_second = float(np.linalg.norm(line_in_second[:2]))
    norm_first = float(np.linalg.norm(line_in_first[:2]))
    if norm_second <= np.finfo(np.float64).eps or norm_first <= np.finfo(np.float64).eps:
        raise ValueError("epipolar line is degenerate")
    return abs(error) * np.sqrt(1.0 / norm_second**2 + 1.0 / norm_first**2)