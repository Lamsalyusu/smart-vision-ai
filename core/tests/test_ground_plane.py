import unittest

import numpy as np

from core.domain.spatial import fit_ground_plane_ransac


class GroundPlaneRansacTests(unittest.TestCase):
    def test_recovers_y_up_plane_with_outliers(self) -> None:
        random = np.random.default_rng(42)
        x_values, z_values = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(1, 6, 8))
        y_values = 1.5 + random.normal(0, 0.003, x_values.shape)
        ground_points = np.column_stack((x_values.ravel(), y_values.ravel(), z_values.ravel()))
        outliers = random.uniform(-4, 4, size=(16, 3))
        cloud = np.vstack((ground_points, outliers))

        estimate = fit_ground_plane_ransac(
            cloud,
            distance_threshold=0.02,
            iterations=300,
            random_seed=7,
        )

        self.assertGreaterEqual(len(estimate.inlier_indices), len(ground_points))
        self.assertGreater(estimate.normal[1], 0.999)
        self.assertAlmostEqual(estimate.offset, -1.5, places=2)
        self.assertLess(estimate.root_mean_squared_error, 0.01)

    def test_rejects_collinear_point_cloud(self) -> None:
        points = [(index, 0, 0) for index in range(5)]

        with self.assertRaisesRegex(ValueError, "enough inliers"):
            fit_ground_plane_ransac(points, distance_threshold=0.01, iterations=10)

    def test_rejects_invalid_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "threshold"):
            fit_ground_plane_ransac([(0, 0, 0), (1, 0, 0), (0, 0, 1)], 0)


if __name__ == "__main__":
    unittest.main()