import unittest

from core.domain.spatial.frustum import (
    CameraIntrinsics,
    OrientedBox3D,
    backproject_frustum_box,
    estimate_surface_normal,
)
from core.domain.spatial.transforms import Quaternion, Vector3


class CameraIntrinsicsTests(unittest.TestCase):
    def test_valid_intrinsics(self):
        k = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        self.assertEqual(k.fx, 500.0)
        self.assertEqual(k.fy, 500.0)

    def test_rejects_non_positive_focal_length(self):
        with self.assertRaises(ValueError):
            CameraIntrinsics(fx=0.0, fy=500.0, cx=320.0, cy=240.0)

    def test_rejects_non_finite_principal_point(self):
        with self.assertRaises(ValueError):
            CameraIntrinsics(fx=500.0, fy=500.0, cx=float("nan"), cy=240.0)

    def test_unproject(self):
        k = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        point = k.unproject(320.0, 240.0, 10.0)
        self.assertAlmostEqual(point.x, 0.0)
        self.assertAlmostEqual(point.y, 0.0)
        self.assertAlmostEqual(point.z, 10.0)

    def test_unproject_rejects_invalid_depth(self):
        k = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        with self.assertRaises(ValueError):
            k.unproject(320.0, 240.0, 0.0)

    def test_project(self):
        k = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        u, v = k.project(Vector3(0.2, 0.1, 1.0))
        self.assertAlmostEqual(u, 420.0)
        self.assertAlmostEqual(v, 290.0)

    def test_project_rejects_behind_camera(self):
        k = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        with self.assertRaises(ValueError):
            k.project(Vector3(1.0, 1.0, 0.0))


class OrientedBox3DTests(unittest.TestCase):
    def test_volume(self):
        box = OrientedBox3D(
            center=Vector3(0.0, 0.0, 0.0),
            size=Vector3(2.0, 3.0, 4.0),
            orientation=Quaternion.identity(),
        )
        self.assertAlmostEqual(box.volume, 24.0)

    def test_contains_point_inside(self):
        box = OrientedBox3D(
            center=Vector3(0.0, 0.0, 0.0),
            size=Vector3(2.0, 2.0, 2.0),
            orientation=Quaternion.identity(),
        )
        self.assertTrue(box.contains_point(Vector3(0.5, 0.5, 0.5)))

    def test_contains_point_outside(self):
        box = OrientedBox3D(
            center=Vector3(0.0, 0.0, 0.0),
            size=Vector3(2.0, 2.0, 2.0),
            orientation=Quaternion.identity(),
        )
        self.assertFalse(box.contains_point(Vector3(2.0, 0.0, 0.0)))

    def test_rejects_non_positive_size(self):
        with self.assertRaises(ValueError):
            OrientedBox3D(
                center=Vector3(0.0, 0.0, 0.0),
                size=Vector3(0.0, 2.0, 2.0),
                orientation=Quaternion.identity(),
            )


class SurfaceNormalTests(unittest.TestCase):
    def test_estimate_surface_normal(self):
        points = [
            Vector3(0.0, 0.0, 0.0),
            Vector3(1.0, 0.0, 0.0),
            Vector3(0.0, 1.0, 0.0),
        ]
        normal = estimate_surface_normal(points)
        # Normal should be (0,0,-1) oriented toward camera (z<0)
        self.assertAlmostEqual(normal.x, 0.0)
        self.assertAlmostEqual(normal.y, 0.0)
        self.assertAlmostEqual(normal.z, -1.0)

    def test_estimate_surface_normal_collinear_returns_vertical(self):
        points = [
            Vector3(0.0, 0.0, 0.0),
            Vector3(1.0, 0.0, 0.0),
            Vector3(2.0, 0.0, 0.0),
        ]
        normal = estimate_surface_normal(points)
        self.assertEqual(normal, Vector3(0.0, 1.0, 0.0))

    def test_estimate_surface_normal_requires_three_points(self):
        with self.assertRaises(ValueError):
            estimate_surface_normal([Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0)])


class FrustumBoxTests(unittest.TestCase):
    def test_backproject_frustum_box(self):
        k = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        box = backproject_frustum_box(300.0, 220.0, 340.0, 260.0, 10.0, 2.0, k)
        self.assertIsInstance(box, OrientedBox3D)
        self.assertGreater(box.size.x, 0)
        self.assertGreater(box.size.y, 0)
        self.assertGreater(box.size.z, 0)

    def test_backproject_rejects_invalid_box(self):
        k = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        with self.assertRaises(ValueError):
            backproject_frustum_box(340.0, 260.0, 300.0, 220.0, 10.0, 2.0, k)


if __name__ == "__main__":
    unittest.main()
