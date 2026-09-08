import unittest
import numpy as np

from core.domain.physics import (
    RigidBody3D,
    NumericalPhysicsIntegrator,
    GJKCollisionDetector,
    ContactImpulseSolver,
)


class TestPhysicsSubsystem(unittest.TestCase):
    def test_rigid_body_integration(self):
        body = RigidBody3D(mass=2.0)
        body.apply_force(np.array([10.0, 0.0, 0.0]))
        NumericalPhysicsIntegrator.semi_implicit_euler(body, dt=0.1)

        self.assertAlmostEqual(body.linear_velocity[0], 0.5, delta=1e-4)
        self.assertAlmostEqual(body.position[0], 0.05, delta=1e-4)

    def test_gjk_collision_detection(self):
        cube1 = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=float)

        cube2_overlapping = cube1 + np.array([0.5, 0.0, 0.0])
        cube3_separated = cube1 + np.array([5.0, 0.0, 0.0])

        self.assertTrue(GJKCollisionDetector.test_intersection(cube1, cube2_overlapping))
        self.assertFalse(GJKCollisionDetector.test_intersection(cube1, cube3_separated))


if __name__ == "__main__":
    unittest.main()
