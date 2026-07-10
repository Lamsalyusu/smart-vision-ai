import math
from collections.abc import Sequence


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