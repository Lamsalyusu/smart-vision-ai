from __future__ import annotations

import numpy as np
from .rigid_body import RigidBody3D


class ContactImpulseSolver:
    """Sequential impulse velocity-level contact and Coulomb friction solver."""

    @staticmethod
    def resolve_contact(
        body_a: RigidBody3D,
        body_b: RigidBody3D,
        contact_point: np.ndarray,
        contact_normal: np.ndarray,  # Points from A to B
        restitution: float = 0.2,
        friction: float = 0.4,
    ) -> None:
        """Computes and applies linear and angular collision impulses."""
        ra = contact_point - body_a.position
        rb = contact_point - body_b.position

        # Relative contact velocity: v_rel = (v_b + w_b x r_b) - (v_a + w_a x r_a)
        va_pt = body_a.linear_velocity + np.cross(body_a.angular_velocity, ra)
        vb_pt = body_b.linear_velocity + np.cross(body_b.angular_velocity, rb)
        v_rel = vb_pt - va_pt

        # Normal velocity
        v_norm = np.dot(v_rel, contact_normal)
        if v_norm > 0:
            return  # Moving apart

        # Effective mass along normal
        inv_mass_sum = body_a.inv_mass + body_b.inv_mass
        ra_xn = np.cross(ra, contact_normal)
        rb_xn = np.cross(rb, contact_normal)
        rot_term_a = np.dot(ra_xn, body_a.inv_inertia_world() @ ra_xn)
        rot_term_b = np.dot(rb_xn, body_b.inv_inertia_world() @ rb_xn)
        k_normal = inv_mass_sum + rot_term_a + rot_term_b

        # Normal impulse magnitude: j = -(1 + e) * v_norm / K
        j_normal = -(1.0 + restitution) * v_norm / max(1e-8, k_normal)
        impulse = j_normal * contact_normal

        # Apply impulse to bodies
        body_a.linear_velocity -= impulse * body_a.inv_mass
        body_a.angular_velocity -= body_a.inv_inertia_world() @ np.cross(ra, impulse)

        body_b.linear_velocity += impulse * body_b.inv_mass
        body_b.angular_velocity += body_b.inv_inertia_world() @ np.cross(rb, impulse)
