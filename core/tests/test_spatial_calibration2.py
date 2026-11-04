import unittest

import numpy as np

from core.domain.spatial.calibration import (
    estimate_fundamental_matrix,
    estimate_homography,
    sampson_epipolar_distance,
    solve_zhang_intrinsics,
    symmetric_epipolar_distance,
)


class HomographyTests(unittest.TestCase):
    def test_estimate_homography_from_four_points(self):
        source = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        target = [(10.0, 10.0), (20.0, 10.0), (10.0, 20.0), (20.0, 20.0)]
        H = estimate_homography(source, target)
        self.assertEqual(H.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(H)))

    def test_estimate_homography_requires_at_least_four_points(self):
        with self.assertRaises(ValueError):
            estimate_homography([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])

    def test_estimate_homography_requires_matching_counts(self):
        with self.assertRaises(ValueError):
            estimate_homography([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)], [(0.0, 0.0), (1.0, 0.0)])

    def test_estimate_homography_rejects_non_finite(self):
        with self.assertRaises(ValueError):
            estimate_homography(
                [(float("nan"), 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
                [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
            )


class ZhangIntrinsicsTests(unittest.TestCase):
    def _make_homography(self, fx, fy, cx, cy):
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    def test_solve_zhang_intrinsics_from_three_homographies(self):
        # Homographies from three different camera poses viewing a plane.
        # H = K * [r1, r2, t] for plane at Z=0
        K_true = np.array([[800.0, 0.0, 400.0], [0.0, 800.0, 300.0], [0.0, 0.0, 1.0]])

        def make_homography(R, t):
            return K_true @ np.column_stack([R[:, 0], R[:, 1], t])

        # Pose 1: identity rotation, camera at (0,0,1)
        h1 = make_homography(np.eye(3), np.array([0.0, 0.0, 1.0]))
        # Pose 2: 10 degree rotation about Y
        R2 = np.array([[0.985, 0.0, 0.174], [0.0, 1.0, 0.0], [-0.174, 0.0, 0.985]])
        h2 = make_homography(R2, np.array([0.0, 0.0, 1.2]))
        # Pose 3: 10 degree rotation about X
        R3 = np.array([[1.0, 0.0, 0.0], [0.0, 0.985, -0.174], [0.0, 0.174, 0.985]])
        h3 = make_homography(R3, np.array([0.0, 0.0, 1.5]))
        K = solve_zhang_intrinsics([h1, h2, h3])
        self.assertEqual(K.shape, (3, 3))
        self.assertGreater(K[0, 0], 0)
        self.assertGreater(K[1, 1], 0)

    def test_solve_zhang_requires_at_least_three(self):
        with self.assertRaises(ValueError):
            solve_zhang_intrinsics([np.eye(3), np.eye(3)])

    def test_solve_zhang_rejects_non_finite(self):
        bad_h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, float("nan")]])
        with self.assertRaises(ValueError):
            solve_zhang_intrinsics([bad_h, np.eye(3), np.eye(3)])


class FundamentalMatrixTests(unittest.TestCase):
    def test_estimate_fundamental_matrix_from_eight_points(self):
        # Non-collinear points in general position
        pts1 = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 0.5), (0.5, 2.0), (2.0, 2.0), (0.3, 0.7)]
        pts2 = [(0.1, 0.1), (1.1, 0.0), (0.0, 1.1), (1.1, 1.1), (2.1, 0.6), (0.6, 2.1), (2.1, 2.1), (0.4, 0.8)]
        F = estimate_fundamental_matrix(pts1, pts2)
        self.assertEqual(F.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(F)))

    def test_estimate_fundamental_requires_at_least_eight(self):
        with self.assertRaises(ValueError):
            estimate_fundamental_matrix([(0.0, 0.0)] * 7, [(0.0, 0.0)] * 7)

    def test_estimate_fundamental_requires_matching_counts(self):
        with self.assertRaises(ValueError):
            estimate_fundamental_matrix([(0.0, 0.0)] * 8, [(0.0, 0.0)] * 5)


class EpipolarDistanceTests(unittest.TestCase):
    def test_symmetric_epipolar_distance(self):
        F = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        d = symmetric_epipolar_distance(F, (1.0, 2.0), (3.0, 4.0))
        self.assertIsInstance(d, float)
        self.assertTrue(np.isfinite(d))

    def test_symmetric_epipolar_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            symmetric_epipolar_distance(np.eye(2), (1.0, 2.0), (3.0, 4.0))

    def test_sampson_epipolar_distance(self):
        F = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        d = sampson_epipolar_distance(F, (1.0, 2.0), (3.0, 4.0))
        self.assertIsInstance(d, float)
        self.assertTrue(np.isfinite(d))

    def test_sampson_rejects_degenerate(self):
        with self.assertRaises(ValueError):
            sampson_epipolar_distance(np.zeros((3, 3)), (1.0, 2.0), (3.0, 4.0))


if __name__ == "__main__":
    unittest.main()
