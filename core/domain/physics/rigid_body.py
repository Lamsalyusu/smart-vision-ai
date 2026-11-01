from __future__ import annotations

import numpy as np


class RigidBody3D:
    """6-Degrees-of-Freedom physical rigid body state with spatial dynamics."""

    def __init__(
        self,
        mass: float = 1.0,
        inertia_body: np.ndarray | None = None,
        initial_position: np.ndarray | None = None,
    ) -> None:
        self.mass = max(1e-4, mass)
        self.inv_mass = 1.0 / self.mass

        if inertia_body is not None:
            self.inertia_body = np.array(inertia_body, dtype=np.float64)
        else:
            # Default unit cube moment of inertia: I = 1/6 * m * s^2
            self.inertia_body = np.eye(3, dtype=np.float64) * (self.mass / 6.0)
        self.inv_inertia_body = np.linalg.inv(self.inertia_body)

        self.position = np.array(initial_position if initial_position is not None else [0.0, 0.0, 0.0], dtype=np.float64)
        self.linear_velocity = np.zeros(3, dtype=np.float64)
        self.linear_momentum = np.zeros(3, dtype=np.float64)

        # Orientation quaternion: [w, x, y, z]
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.angular_velocity = np.zeros(3, dtype=np.float64)
        self.angular_momentum = np.zeros(3, dtype=np.float64)

        # External forces and torques
        self.force_accumulator = np.zeros(3, dtype=np.float64)
        self.torque_accumulator = np.zeros(3, dtype=np.float64)

    def rotation_matrix(self) -> np.ndarray:
        """Converts unit quaternion [w, x, y, z] to 3x3 rotation matrix R."""
        w, x, y, z = self.orientation
        return np.array([
            [1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - z*w),       2.0 * (x*z + y*w)],
            [2.0 * (x*y + z*w),       1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - x*w)],
            [2.0 * (x*z - y*w),       2.0 * (y*z + x*w),       1.0 - 2.0 * (x*x + y*y)]
        ], dtype=np.float64)

    def inv_inertia_world(self) -> np.ndarray:
        """Inertia tensor in world coordinates: I_world^-1 = R * I_body^-1 * R^T."""
        r = self.rotation_matrix()
        return r @ self.inv_inertia_body @ r.T

    def apply_force(self, force: np.ndarray, point_world: np.ndarray | None = None) -> None:
        """Apply force at a given world position, generating torque: tau = r x F."""
        self.force_accumulator += force
        if point_world is not None:
            r = point_world - self.position
            self.torque_accumulator += np.cross(r, force)

    def clear_accumulators(self) -> None:
        self.force_accumulator.fill(0.0)
        self.torque_accumulator.fill(0.0)
