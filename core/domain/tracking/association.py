import math
from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]


def mahalanobis_squared(
    residual: Sequence[float], covariance: Sequence[Sequence[float]]
) -> float:
    """Return r^T covariance^-1 r using a Cholesky solve, without matrix inversion."""
    dimension = len(residual)
    if dimension == 0 or len(covariance) != dimension:
        raise ValueError("residual and covariance dimensions must match and be non-empty")
    if any(len(row) != dimension for row in covariance):
        raise ValueError("covariance must be square and match the residual dimension")
    if not all(math.isfinite(value) for value in residual):
        raise ValueError("residual values must be finite")

    lower = [[0.0] * dimension for _ in range(dimension)]
    for row_index in range(dimension):
        for column_index in range(dimension):
            value = covariance[row_index][column_index]
            if not math.isfinite(value):
                raise ValueError("covariance values must be finite")
            if not math.isclose(
                value,
                covariance[column_index][row_index],
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError("covariance must be symmetric")

        for column_index in range(row_index + 1):
            value = covariance[row_index][column_index] - sum(
                lower[row_index][prior] * lower[column_index][prior]
                for prior in range(column_index)
            )
            if row_index == column_index:
                if value <= 0:
                    raise ValueError("covariance must be positive definite")
                lower[row_index][column_index] = math.sqrt(value)
            else:
                lower[row_index][column_index] = value / lower[column_index][column_index]

    whitened = [0.0] * dimension
    for row_index in range(dimension):
        remainder = residual[row_index] - sum(
            lower[row_index][column_index] * whitened[column_index]
            for column_index in range(row_index)
        )
        whitened[row_index] = remainder / lower[row_index][row_index]

    distance_squared = sum(value * value for value in whitened)
    if not math.isfinite(distance_squared):
        raise ValueError("Mahalanobis distance is not finite")
    return distance_squared


def calculate_iou(
    bbox1: Sequence[float],
    bbox2: Sequence[float],
) -> float:
    """Compute Intersection over Union (IoU) between bounding boxes [x1, y1, x2, y2]."""
    if len(bbox1) != 4 or len(bbox2) != 4:
        raise ValueError("bounding boxes must have four coordinates: [x1, y1, x2, y2]")
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area1 = max(0.0, bbox1[2] - bbox1[0]) * max(0.0, bbox1[3] - bbox1[1])
    area2 = max(0.0, bbox2[2] - bbox2[0]) * max(0.0, bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return float(intersection / union) if union > 0.0 else 0.0


def iou_cost_matrix(
    boxes_a: Sequence[Sequence[float]],
    boxes_b: Sequence[Sequence[float]],
) -> np.ndarray:
    """Compute 1.0 - IoU cost matrix between two sets of bounding boxes."""
    n_a = len(boxes_a)
    n_b = len(boxes_b)
    if n_a == 0 or n_b == 0:
        return np.empty((n_a, n_b), dtype=np.float64)

    matrix = np.zeros((n_a, n_b), dtype=np.float64)
    for i, box_a in enumerate(boxes_a):
        for j, box_b in enumerate(boxes_b):
            matrix[i, j] = 1.0 - calculate_iou(box_a, box_b)
    return matrix


def linear_assignment(cost_matrix: np.ndarray, max_cost: float = 1e5) -> np.ndarray:
    """Return minimum-cost bipartite pairs using Jonker-Volgenant (Munkres) assignment.

    Filters out matched pairs whose cost meets or exceeds max_cost.
    """
    matrix = np.asarray(cost_matrix)
    if matrix.ndim != 2:
        raise ValueError("cost matrix must be two-dimensional")
    if not np.isfinite(matrix).all():
        raise ValueError("cost matrix must contain only finite values")

    n_rows, n_cols = matrix.shape
    if n_rows == 0 or n_cols == 0:
        return np.empty((0, 2), dtype=int)

    row_indices, column_indices = linear_sum_assignment(matrix)
    valid = matrix[row_indices, column_indices] < max_cost
    return np.column_stack((row_indices[valid], column_indices[valid])).astype(int)


def fuse_reid_mahalanobis_cost(
    reid_distance_matrix: np.ndarray,
    mahalanobis_distance_matrix: np.ndarray,
    alpha: float = 0.6,
    mahalanobis_gate: float = 9.488,  # Chi-square 95% threshold for 4-DOF
    gated_cost: float = 1e5,
) -> np.ndarray:
    """Fuse appearance cosine distance and motion Mahalanobis distance.

    Applies hard gating on Mahalanobis distance before weighted linear combination.
    """
    if reid_distance_matrix.shape != mahalanobis_distance_matrix.shape:
        raise ValueError("Re-ID and Mahalanobis distance matrices must have matching shapes")
    fused = alpha * reid_distance_matrix + (1.0 - alpha) * mahalanobis_distance_matrix
    # Gating
    gated_mask = mahalanobis_distance_matrix > mahalanobis_gate
    fused[gated_mask] = gated_cost
    return fused
