import unittest

import numpy as np

from core.domain.vision.camera_models import FisheyeKannalaBrandtModel, PinholeCameraModel


class PinholeCameraModelTests(unittest.TestCase):
    def test_intrinsic_matrix(self):
        cam = PinholeCameraModel(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        K = cam.intrinsic_matrix
        self.assertEqual(K.shape, (3, 3))
        self.assertEqual(K[0, 0], 500.0)
        self.assertEqual(K[1, 1], 500.0)
        self.assertEqual(K[0, 2], 320.0)
        self.assertEqual(K[1, 2], 240.0)
        self.assertEqual(K[2, 2], 1.0)

    def test_project_3d_to_pixel(self):
        cam = PinholeCameraModel(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        pt = np.array([1.0, 1.0, 10.0])
        uv = cam.project_3d_to_pixel(pt)
        self.assertEqual(uv.shape, (2,))
        self.assertAlmostEqual(uv[0], 370.0)  # 500*0.1 + 320
        self.assertAlmostEqual(uv[1], 290.0)  # 500*0.1 + 240

    def test_project_3d_to_pixel_z_near_zero(self):
        cam = PinholeCameraModel(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        pt = np.array([1.0, 1.0, 0.0])  # z=0, should be clamped
        uv = cam.project_3d_to_pixel(pt)
        self.assertEqual(uv.shape, (2,))

    def test_unproject_pixel_to_ray(self):
        cam = PinholeCameraModel(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        ray = cam.unproject_pixel_to_ray(320.0, 240.0)
        self.assertEqual(ray.shape, (3,))
        self.assertAlmostEqual(np.linalg.norm(ray), 1.0)

    def test_unproject_pixel_to_ray_converges(self):
        cam = PinholeCameraModel(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        ray = cam.unproject_pixel_to_ray(500.0, 400.0, max_iter=20)
        self.assertEqual(ray.shape, (3,))
        self.assertAlmostEqual(np.linalg.norm(ray), 1.0)


class FisheyeKannalaBrandtModelTests(unittest.TestCase):
    def test_project_center(self):
        cam = FisheyeKannalaBrandtModel(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        pt = np.array([0.0, 0.0, 1.0])
        uv = cam.project(pt)
        self.assertEqual(uv.shape, (2,))
        self.assertAlmostEqual(uv[0], 320.0)
        self.assertAlmostEqual(uv[1], 240.0)

    def test_project_off_center(self):
        cam = FisheyeKannalaBrandtModel(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        pt = np.array([1.0, 0.0, 1.0])
        uv = cam.project(pt)
        self.assertEqual(uv.shape, (2,))
        self.assertTrue(np.all(np.isfinite(uv)))


if __name__ == "__main__":
    unittest.main()
