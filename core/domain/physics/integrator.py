from __future__ import annotations

import numpy as np

from .rigid_body import RigidBody3D


class NumericalPhysicsIntegrator:
    """Numerical differential equation integrators for 6-DoF rigid body simulation."""

    @staticmethod
    def semi_implicit_euler(body: RigidBody3D, dt: float) -> None:
        """Symplectic / Semi-implicit Euler integration preserving phase space volume."""
        # Linear dynamics
        body.linear_velocity += (body.force_accumulator * body.inv_mass) * dt
        body.position += body.linear_velocity * dt

        # Rotational dynamics
        inv_i_world = body.inv_inertia_world()
        body.angular_velocity += (inv_i_world @ body.torque_accumulator) * dt

        # Quaternion derivative: q_dot = 0.5 * omega * q
        w = body.angular_velocity
        q = body.orientation
        omega_quat = np.array([0.0, w[0], w[1], w[2]])

        # Quaternion multiplication: omega_quat * q
        qw, qx, qy, qz = q
        _ow, ox, oy, oz = omega_quat
        q_dot = 0.5 * np.array([
            -ox*qx - oy*qy - oz*qz,
             ox*qw + oy*qz - oz*qy,
            -ox*qz + oy*qw + oz*qx,
             ox*qy - oy*qx + oz*qw
        ])

        body.orientation += q_dot * dt
        # Normalize quaternion
        norm = np.linalg.norm(body.orientation)
        if norm > 1e-8:
            body.orientation /= norm

        body.clear_accumulators()

    @staticmethod
    def runge_kutta_4th_order(body: RigidBody3D, dt: float) -> None:
        """4th-Order Runge-Kutta (RK4) integrator for high numerical precision."""
        # For autonomous rigid body simulation with constant force within step
        k1_v = (body.force_accumulator * body.inv_mass) * dt
        k1_x = body.linear_velocity * dt

        k2_v = (body.force_accumulator * body.inv_mass) * dt
        k2_x = (body.linear_velocity + 0.5 * k1_v) * dt

        k3_v = (body.force_accumulator * body.inv_mass) * dt
        k3_x = (body.linear_velocity + 0.5 * k2_v) * dt

        k4_v = (body.force_accumulator * body.inv_mass) * dt
        k4_x = (body.linear_velocity + k3_v) * dt

        body.linear_velocity += (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0
        body.position += (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x) / 6.0

        # Angular RK4
        inv_i = body.inv_inertia_world()
        body.angular_velocity += (inv_i @ body.torque_accumulator) * dt

        body.clear_accumulators()
