import unittest

import numpy as np

from core.domain.spatial.triangulation import (
    CameraProjectionMatrix,
    StereoTriangulationEngine,
)


class EssentialMatrixTests(unittest.TestCase):
    def test_compute_essential_matrix(self):
        F = np.eye(3)
        K1 = np.eye(3)
        K2 = np.eye(3)
        E = StereoTriangulationEngine.compute_essential_matrix(F, K1, K2)
        self.assertTrue(np.allclose(E, np.eye(3)))

    def test_decompose_essential_matrix_returns_four_solutions(self):
        # Construct a valid essential matrix from known R, t
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        t = np.array([[1.0], [0.0], [0.0]])
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        E = R @ np.array([[0, -t[2, 0], t[1, 0]], [t[2, 0], 0, -t[0, 0]], [-t[1, 0], t[0, 0], 0]])
        solutions = StereoTriangulationEngine.decompose_essential_matrix(E)
        self.assertEqual(len(solutions), 4)
        for R_candidate, t_candidate in solutions:
            self.assertEqual(R_candidate.shape, (3, 3))
            self.assertEqual(t_candidate.shape, (3, 1))


class TriangulationDLTTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        self.cam1 = CameraProjectionMatrix("cam1", self.K @ P1)
        R = np.eye(3)
        t = np.array([[1.0], [0.0], [0.0]])
        P2 = np.hstack([R, t])
        self.cam2 = CameraProjectionMatrix("cam2", self.K @ P2)

    def test_triangulate_point_two_views(self):
        # Point at (1, 2, 5) projects roughly to same u,v in both cameras
        # given baseline along X
        points_2d = [(640.0, 480.0), (140.0, 480.0)]
        X = StereoTriangulationEngine.triangulate_point_dlt(
            points_2d, [self.cam1, self.cam2]
        )
        self.assertEqual(X.shape, (3,))
        self.assertTrue(np.all(np.isfinite(X)))

    def test_triangulate_requires_at_least_two_views(self):
        with self.assertRaises(ValueError):
            StereoTriangulationEngine.triangulate_point_dlt(
                [(100.0, 100.0)], [self.cam1]
            )

    def test_triangulate_requires_matching_point_camera_counts(self):
        with self.assertRaises(ValueError):
            StereoTriangulationEngine.triangulate_point_dlt(
                [(100.0, 100.0), (200.0, 200.0)], [self.cam1]
            )


class CheiralitySelectionTests(unittest.TestCase):
    def test_select_cheirality_valid_solution_returns_tuple(self):
        E = np.eye(3)
        K1 = np.eye(3)
        K2 = np.eye(3)
        result = StereoTriangulationEngine.select_cheirality_valid_solution(
            E, K1, K2, (100.0, 100.0), (200.0, 200.0)
        )
        self.assertEqual(len(result), 3)
        R, t, X = result
        self.assertEqual(R.shape, (3, 3))
        self.assertEqual(t.shape, (3, 1))
        self.assertEqual(X.shape, (3,))


if __name__ == "__main__":
    unittest.main()
