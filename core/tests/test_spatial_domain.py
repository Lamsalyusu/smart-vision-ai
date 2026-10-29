import math
import unittest

from core.domain.spatial import CoordinateFrame, Quaternion, Transform3D, Vector3


class SpatialTransformTests(unittest.TestCase):
    def test_quaternion_rotates_vector_about_z_axis(self) -> None:
        half_angle = math.pi / 4
        rotation = Quaternion(math.cos(half_angle), 0, 0, math.sin(half_angle))

        rotated = rotation.rotate(Vector3(1, 0, 0))

        self.assertAlmostEqual(rotated.x, 0)
        self.assertAlmostEqual(rotated.y, 1)
        self.assertAlmostEqual(rotated.z, 0)

    def test_transform_inverse_round_trip(self) -> None:
        camera = CoordinateFrame("camera")
        world = CoordinateFrame("world")
        transform = Transform3D(
            camera,
            world,
            Vector3(3, -2, 1),
            Quaternion.from_components(0.8, 0, 0, 0.6),
        )
        point = Vector3(4, 5, 6)

        recovered = transform.inverse().transform_point(transform.transform_point(point))

        self.assertAlmostEqual(recovered.x, point.x)
        self.assertAlmostEqual(recovered.y, point.y)
        self.assertAlmostEqual(recovered.z, point.z)

    def test_composition_requires_matching_intermediate_frame(self) -> None:
        camera = CoordinateFrame("camera")
        world = CoordinateFrame("world")
        map_frame = CoordinateFrame("map")
        first = Transform3D(camera, world, Vector3(0, 0, 0), Quaternion.identity())
        incompatible = Transform3D(map_frame, world, Vector3(0, 0, 0), Quaternion.identity())

        with self.assertRaises(ValueError):
            first.then(incompatible)

    def test_quaternion_rejects_non_unit_length(self) -> None:
        with self.assertRaises(ValueError):
            Quaternion(2, 0, 0, 0)

    def test_quaternion_factory_normalizes_components(self) -> None:
        quaternion = Quaternion.from_components(2, 0, 0, 0)

        self.assertEqual(quaternion, Quaternion.identity())


if __name__ == "__main__":
    unittest.main()