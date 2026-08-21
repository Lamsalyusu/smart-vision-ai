import unittest
import numpy as np

from core.domain.spatial.triangulation import (
    CameraProjectionMatrix,
    StereoTriangulationEngine,
)


class StereoTriangulationTests(unittest.TestCase):
    def test_triangulation_dlt_synthetic(self):
        # Camera 1 at origin looking along +Z
        K1 = np.array([
            [1000.0, 0.0, 500.0],
            [0.0, 1000.0, 500.0],
            [0.0, 0.0, 1.0],
        ])
        R1 = np.eye(3)
        t1 = np.zeros((3, 1))
        P1 = np.dot(K1, np.hstack([R1, t1]))

        # Camera 2 translated along X axis by 1.0 meter
        K2 = np.copy(K1)
        R2 = np.eye(3)
        t2 = np.array([[-1.0], [0.0], [0.0]])
        P2 = np.dot(K2, np.hstack([R2, t2]))

        # Ground truth 3D point at X=0, Y=0, Z=5.0 meters
        X_true = np.array([0.0, 0.0, 5.0, 1.0])

        # Project into both cameras
        p1_proj = np.dot(P1, X_true)
        pt1 = (p1_proj[0] / p1_proj[2], p1_proj[1] / p1_proj[2])

        p2_proj = np.dot(P2, X_true)
        pt2 = (p2_proj[0] / p2_proj[2], p2_proj[1] / p2_proj[2])

        # Reconstruct via SVD DLT
        cam1 = CameraProjectionMatrix("c1", P1)
        cam2 = CameraProjectionMatrix("c2", P2)
        X_reconstructed = StereoTriangulationEngine.triangulate_point_dlt([pt1, pt2], [cam1, cam2])

        np.testing.assert_allclose(X_reconstructed, X_true[:3], atol=1e-4)

    def test_essential_matrix_decomposition(self):
        R_true = np.eye(3)
        t_true = np.array([[0.0], [1.0], [0.0]])
        # Tx skew matrix
        Tx = np.array([
            [0, -t_true[2, 0], t_true[1, 0]],
            [t_true[2, 0], 0, -t_true[0, 0]],
            [-t_true[1, 0], t_true[0, 0], 0],
        ])
        E = np.dot(Tx, R_true)

        solutions = StereoTriangulationEngine.decompose_essential_matrix(E)
        self.assertEqual(len(solutions), 4)
        for R, t in solutions:
            # Determinant of rotation must be +1
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=4)
            # Norm of translation must be unit norm
            self.assertAlmostEqual(float(np.linalg.norm(t)), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
