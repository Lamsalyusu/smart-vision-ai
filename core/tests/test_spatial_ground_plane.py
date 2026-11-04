import unittest

import numpy as np

from core.domain.spatial.ground_plane import PlaneEstimate, fit_ground_plane_ransac


class GroundPlaneRANSACTests(unittest.TestCase):
    def test_fits_flat_xy_plane(self):
        # Points on z=0 plane with small noise
        points = np.array(
            [
                [0.0, 0.0, 0.01],
                [1.0, 0.0, -0.01],
                [0.0, 1.0, 0.02],
                [1.0, 1.0, -0.02],
                [0.5, 0.5, 0.0],
            ],
            dtype=np.float64,
        )
        result = fit_ground_plane_ransac(
            points, distance_threshold=0.05, iterations=100, random_seed=42
        )
        self.assertIsInstance(result, PlaneEstimate)
        # Normal should be approximately vertical (z-dominant for a ground plane)
        self.assertAlmostEqual(result.normal[0], 0.0, places=1)
        self.assertAlmostEqual(result.normal[1], 0.0, places=1)
        self.assertAlmostEqual(abs(result.normal[2]), 1.0, places=1)
        self.assertLess(result.root_mean_squared_error, 0.05)

    def test_rejects_insufficient_points(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        with self.assertRaises(ValueError):
            fit_ground_plane_ransac(points, distance_threshold=0.1)

    def test_rejects_non_finite_coordinates(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]],
            dtype=np.float64,
        )
        with self.assertRaises(ValueError):
            fit_ground_plane_ransac(points, distance_threshold=0.1)

    def test_rejects_invalid_threshold(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        )
        with self.assertRaises(ValueError):
            fit_ground_plane_ransac(points, distance_threshold=-1.0)

    def test_rejects_invalid_iterations(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        )
        with self.assertRaises(ValueError):
            fit_ground_plane_ransac(points, distance_threshold=0.1, iterations=0)

    def test_rejects_invalid_min_inliers(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        )
        with self.assertRaises(ValueError):
            fit_ground_plane_ransac(
                points, distance_threshold=0.1, min_inliers=10
            )

    def test_raises_when_no_plane_found(self):
        # Random points with no planar structure
        points = np.random.default_rng(0).uniform(0, 10, size=(20, 3))
        with self.assertRaises(ValueError):
            fit_ground_plane_ransac(
                points, distance_threshold=0.001, iterations=50, min_inliers=10
            )


if __name__ == "__main__":
    unittest.main()
