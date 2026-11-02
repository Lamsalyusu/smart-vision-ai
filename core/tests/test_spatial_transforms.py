import math
import unittest

from core.domain.spatial.transforms import (
    CoordinateFrame,
    Quaternion,
    Transform3D,
    Vector3,
)


class Vector3Tests(unittest.TestCase):
    def test_addition(self):
        v1 = Vector3(1.0, 2.0, 3.0)
        v2 = Vector3(4.0, 5.0, 6.0)
        result = v1 + v2
        self.assertEqual(result, Vector3(5.0, 7.0, 9.0))

    def test_subtraction(self):
        v1 = Vector3(5.0, 7.0, 9.0)
        v2 = Vector3(1.0, 2.0, 3.0)
        result = v1 - v2
        self.assertEqual(result, Vector3(4.0, 5.0, 6.0))

    def test_scaled(self):
        v = Vector3(1.0, 2.0, 3.0)
        result = v.scaled(2.5)
        self.assertEqual(result, Vector3(2.5, 5.0, 7.5))

    def test_scaled_rejects_non_finite(self):
        v = Vector3(1.0, 2.0, 3.0)
        with self.assertRaises(ValueError):
            v.scaled(float("nan"))

    def test_cross(self):
        v1 = Vector3(1.0, 0.0, 0.0)
        v2 = Vector3(0.0, 1.0, 0.0)
        result = v1.cross(v2)
        self.assertEqual(result, Vector3(0.0, 0.0, 1.0))

    def test_rejects_non_finite_coordinates(self):
        with self.assertRaises(ValueError):
            Vector3(float("nan"), 0.0, 0.0)


class QuaternionTests(unittest.TestCase):
    def test_identity(self):
        q = Quaternion.identity()
        self.assertEqual(q, Quaternion(1.0, 0.0, 0.0, 0.0))

    def test_from_components_normalizes(self):
        q = Quaternion.from_components(2.0, 0.0, 0.0, 0.0)
        self.assertEqual(q, Quaternion.identity())

    def test_from_components_rejects_zero(self):
        with self.assertRaises(ValueError):
            Quaternion.from_components(0.0, 0.0, 0.0, 0.0)

    def test_rejects_non_unit_length(self):
        with self.assertRaises(ValueError):
            Quaternion(2.0, 0.0, 0.0, 0.0)

    def test_conjugate(self):
        q = Quaternion(0.0, 1.0, 0.0, 0.0)
        self.assertEqual(q.conjugate(), Quaternion(0.0, -1.0, 0.0, 0.0))

    def test_multiplied(self):
        q1 = Quaternion(0.0, 1.0, 0.0, 0.0)
        q2 = Quaternion(0.0, 0.0, 1.0, 0.0)
        # i * j = k
        result = q1.multiplied(q2)
        self.assertAlmostEqual(result.w, 0.0)
        self.assertAlmostEqual(result.x, 0.0)
        self.assertAlmostEqual(result.y, 0.0)
        self.assertAlmostEqual(result.z, 1.0)

    def test_rotate_90_degrees_about_z(self):
        half_angle = math.pi / 4
        rotation = Quaternion(math.cos(half_angle), 0.0, 0.0, math.sin(half_angle))
        rotated = rotation.rotate(Vector3(1.0, 0.0, 0.0))
        self.assertAlmostEqual(rotated.x, 0.0, places=6)
        self.assertAlmostEqual(rotated.y, 1.0, places=6)
        self.assertAlmostEqual(rotated.z, 0.0, places=6)


class CoordinateFrameTests(unittest.TestCase):
    def test_valid_frame_id(self):
        frame = CoordinateFrame("camera")
        self.assertEqual(frame.frame_id, "camera")

    def test_rejects_empty_frame_id(self):
        with self.assertRaises(ValueError):
            CoordinateFrame("")

    def test_rejects_whitespace_frame_id(self):
        with self.assertRaises(ValueError):
            CoordinateFrame(" camera ")


class Transform3DTests(unittest.TestCase):
    def test_transform_point(self):
        camera = CoordinateFrame("camera")
        world = CoordinateFrame("world")
        transform = Transform3D(
            camera, world, Vector3(1.0, 2.0, 3.0), Quaternion.identity()
        )
        point = Vector3(4.0, 5.0, 6.0)
        result = transform.transform_point(point)
        self.assertEqual(result, Vector3(5.0, 7.0, 9.0))

    def test_inverse_round_trip(self):
        camera = CoordinateFrame("camera")
        world = CoordinateFrame("world")
        transform = Transform3D(
            camera,
            world,
            Vector3(3.0, -2.0, 1.0),
            Quaternion.from_components(0.8, 0.0, 0.0, 0.6),
        )
        point = Vector3(4.0, 5.0, 6.0)
        recovered = transform.inverse().transform_point(transform.transform_point(point))
        self.assertAlmostEqual(recovered.x, point.x)
        self.assertAlmostEqual(recovered.y, point.y)
        self.assertAlmostEqual(recovered.z, point.z)

    def test_then_composes(self):
        camera = CoordinateFrame("camera")
        world = CoordinateFrame("world")
        map_frame = CoordinateFrame("map")
        first = Transform3D(
            camera, world, Vector3(1.0, 0.0, 0.0), Quaternion.identity()
        )
        second = Transform3D(
            world, map_frame, Vector3(0.0, 2.0, 0.0), Quaternion.identity()
        )
        composed = first.then(second)
        self.assertEqual(composed.from_frame, camera)
        self.assertEqual(composed.to_frame, map_frame)
        self.assertEqual(composed.translation, Vector3(1.0, 2.0, 0.0))

    def test_then_rejects_mismatched_frames(self):
        camera = CoordinateFrame("camera")
        world = CoordinateFrame("world")
        map_frame = CoordinateFrame("map")
        first = Transform3D(camera, world, Vector3(0.0, 0.0, 0.0), Quaternion.identity())
        incompatible = Transform3D(
            map_frame, world, Vector3(0.0, 0.0, 0.0), Quaternion.identity()
        )
        with self.assertRaises(ValueError):
            first.then(incompatible)


if __name__ == "__main__":
    unittest.main()
