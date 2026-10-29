import math
import unittest

import numpy as np

from core.domain.spatial.calibration import (
    estimate_fundamental_matrix,
    estimate_homography,
    solve_zhang_intrinsics,
    symmetric_epipolar_distance,
)


def make_planar_homographies(intrinsics: np.ndarray) -> list[np.ndarray]:
    views = []
    angles = [(-0.2, 0.1), (0.15, 0.25), (0.3, -0.2), (-0.25, -0.15)]
    for angle_x, angle_y in angles:
        rotation_x = np.array(
            [
                [1, 0, 0],
                [0, math.cos(angle_x), -math.sin(angle_x)],
                [0, math.sin(angle_x), math.cos(angle_x)],
            ]
        )
        rotation_y = np.array(
            [
                [math.cos(angle_y), 0, math.sin(angle_y)],
                [0, 1, 0],
                [-math.sin(angle_y), 0, math.cos(angle_y)],
            ]
        )
        rotation = rotation_y @ rotation_x
        views.append(
            intrinsics
            @ np.column_stack((rotation[:, 0], rotation[:, 1], np.array([0.2, -0.1, 4.0])))
        )
    return views


def make_stereo_correspondences() -> tuple[np.ndarray, np.ndarray]:
    intrinsics = np.array([[700.0, 0, 320], [0, 710.0, 240], [0, 0, 1]])
    first_points = []
    second_points = []
    for depth in (4.0, 5.5, 7.0):
        for x in (-1.2, -0.4, 0.5, 1.1):
            for y in (-0.7, 0.2, 0.8):
                point = np.array([x, y, depth])
                first = intrinsics @ point
                second = intrinsics @ (point + np.array([-0.6, 0.0, 0.0]))
                first_points.append(first[:2] / first[2])
                second_points.append(second[:2] / second[2])
    return np.asarray(first_points), np.asarray(second_points)


class HomographyEstimationTests(unittest.TestCase):
    def test_normalized_dlt_recovers_planar_mapping(self) -> None:
        source = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.3, 0.7]], dtype=float)
        target = source * 2 + np.array([4, -3])

        homography = estimate_homography(source, target)
        projected = (homography @ np.column_stack((source, np.ones(len(source)))).T).T
        projected = projected[:, :2] / projected[:, 2, None]

        np.testing.assert_allclose(projected, target, atol=1e-8)

    def test_rejects_collinear_point_correspondences(self) -> None:
        points = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)

        with self.assertRaisesRegex(ValueError, "do not determine"):
            estimate_homography(points, points)


class MultiViewCalibrationTests(unittest.TestCase):
    def test_zhang_solver_recovers_intrinsics(self) -> None:
        expected = np.array([[800.0, 0, 320], [0, 820.0, 240], [0, 0, 1]])

        recovered = solve_zhang_intrinsics(make_planar_homographies(expected))

        np.testing.assert_allclose(recovered, expected, rtol=1e-6, atol=1e-4)

    def test_zhang_solver_requires_multiple_views(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least three"):
            solve_zhang_intrinsics([np.eye(3), np.eye(3)])


class EpipolarGeometryTests(unittest.TestCase):
    def test_fundamental_matrix_fits_stereo_correspondences(self) -> None:
        first, second = make_stereo_correspondences()

        fundamental = estimate_fundamental_matrix(first, second)

        self.assertEqual(np.linalg.matrix_rank(fundamental, tol=1e-8), 2)
        distances = [
            symmetric_epipolar_distance(fundamental, point_a, point_b)
            for point_a, point_b in zip(first, second)
        ]
        self.assertLess(max(distances), 1e-6)

    def test_epipolar_distance_rejects_invalid_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "3x3"):
            symmetric_epipolar_distance(np.eye(2), (0, 0), (1, 1))


if __name__ == "__main__":
    unittest.main()