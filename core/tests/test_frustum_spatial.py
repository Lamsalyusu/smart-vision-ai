import unittest

from core.domain.spatial.frustum import (
    CameraIntrinsics,
    OrientedBox3D,
    backproject_frustum_box,
    estimate_surface_normal,
)
from core.domain.spatial.transforms import Quaternion, Vector3


class FrustumSpatialTests(unittest.TestCase):
    def test_intrinsics_unproject_and_project(self) -> None:
        intrinsics = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0)

        # Center pixel (640, 360) at 5 meters should unproject to (0, 0, 5)
        p3d = intrinsics.unproject(640.0, 360.0, depth_m=5.0)
        self.assertAlmostEqual(p3d.x, 0.0)
        self.assertAlmostEqual(p3d.y, 0.0)
        self.assertAlmostEqual(p3d.z, 5.0)

        # Reproject back
        u, v = intrinsics.project(p3d)
        self.assertAlmostEqual(u, 640.0)
        self.assertAlmostEqual(v, 360.0)

    def test_surface_normal_estimation(self) -> None:
        # Ground plane at y = 0
        points = [
            Vector3(0.0, 0.0, 2.0),
            Vector3(1.0, 0.0, 2.0),
            Vector3(0.0, 0.0, 3.0),
        ]
        normal = estimate_surface_normal(points)
        # Normal should be perpendicular to x and z (aligned along y)
        self.assertAlmostEqual(abs(normal.y), 1.0, places=4)
        self.assertAlmostEqual(normal.x, 0.0, places=4)

    def test_oriented_box_containment(self) -> None:
        box = OrientedBox3D(
            center=Vector3(0.0, 0.0, 3.0),
            size=Vector3(2.0, 2.0, 2.0),
            orientation=Quaternion.identity(),
        )
        self.assertAlmostEqual(box.volume, 8.0)

        # Point inside
        self.assertTrue(box.contains_point(Vector3(0.5, -0.5, 3.2)))
        # Point outside
        self.assertFalse(box.contains_point(Vector3(2.5, 0.0, 3.0)))

    def test_backproject_frustum_box(self) -> None:
        intrinsics = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=500.0)
        obb = backproject_frustum_box(
            x1=400.0,
            y1=400.0,
            x2=600.0,
            y2=600.0,
            depth_m=4.0,
            depth_span_m=0.8,
            intrinsics=intrinsics,
        )
        self.assertAlmostEqual(obb.center.x, 0.0)
        self.assertAlmostEqual(obb.center.y, 0.0)
        self.assertAlmostEqual(obb.center.z, 4.0)
        self.assertAlmostEqual(obb.size.z, 0.8)
        self.assertGreater(obb.volume, 0.0)


if __name__ == "__main__":
    unittest.main()
