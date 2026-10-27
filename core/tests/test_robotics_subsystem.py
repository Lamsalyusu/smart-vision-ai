import unittest

import numpy as np

from core.domain.robotics import (
    AStarPlanner,
    DHParameter,
    GridMap,
    ManipulatorJacobianAnalyzer,
    PIDController,
    PolynomialTrajectoryGenerator,
    SerialManipulatorKinematics,
)


class TestRoboticsSubsystem(unittest.TestCase):
    def test_serial_kinematics(self):
        dh = [
            DHParameter(a=0.0, alpha=np.pi/2, d=0.2, theta=0.0),
            DHParameter(a=0.5, alpha=0.0, d=0.0, theta=0.0),
            DHParameter(a=0.4, alpha=0.0, d=0.0, theta=0.0),
        ]
        arm = SerialManipulatorKinematics(dh)
        t_ee, _ = arm.forward_kinematics(np.array([0.0, 0.0, 0.0]))
        self.assertEqual(t_ee.shape, (4, 4))

        j = arm.compute_jacobian(np.array([0.0, 0.0, 0.0]))
        self.assertEqual(j.shape, (6, 3))
        w = ManipulatorJacobianAnalyzer.yoshikawa_manipulability(j)
        self.assertGreaterEqual(w, 0.0)

    def test_quintic_trajectory(self):
        coeffs = PolynomialTrajectoryGenerator.quintic_spline(
            q0=0.0, q1=1.0, v0=0.0, v1=0.0, acc0=0.0, acc1=0.0, duration=2.0
        )
        pos, vel, _acc = PolynomialTrajectoryGenerator.evaluate_polynomial(coeffs, t=2.0)
        self.assertAlmostEqual(pos, 1.0, delta=1e-4)
        self.assertAlmostEqual(vel, 0.0, delta=1e-4)

    def test_astar_planner(self):
        grid = np.zeros((10, 10), dtype=np.uint8)
        grid[2:8, 5] = 1  # Vertical wall obstacle
        grid[5, 5] = 0     # Passageway opening
        grid_map = GridMap(width=10, height=10, resolution=1.0)
        grid_map.grid = grid
        planner = AStarPlanner(grid_map)
        path = planner.plan(start=(1, 5), goal=(8, 5))
        self.assertIsNotNone(path)
        self.assertEqual(path[0], (1, 5))
        self.assertEqual(path[-1], (8, 5))

    def test_pid_controller(self):
        pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
        u = pid.update(setpoint=10.0, measured=0.0, dt=0.1)
        self.assertGreater(u, 0.0)


if __name__ == "__main__":
    unittest.main()
