from __future__ import annotations

import numpy as np


class GJKCollisionDetector:
    """Gilbert-Johnson-Keerthi (GJK) algorithm for convex polyhedron collision detection."""

    @staticmethod
    def support_polyhedron(vertices: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Support mapping: argmax_{v in V} (v . d)."""
        dots = np.dot(vertices, direction)
        return vertices[np.argmax(dots)]

    @classmethod
    def minkowski_support(cls, shape_a: np.ndarray, shape_b: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Support function for Minkowski difference: S_A-B(d) = S_A(d) - S_B(-d)."""
        sa = cls.support_polyhedron(shape_a, d)
        sb = cls.support_polyhedron(shape_b, -d)
        return sa - sb

    @classmethod
    def test_intersection(cls, shape_a: np.ndarray, shape_b: np.ndarray) -> bool:
        """Tests if convex polyhedra A and B collide."""
        # Initial search direction
        d = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        simplex = [cls.minkowski_support(shape_a, shape_b, d)]

        # Reverse direction towards origin
        d = -simplex[0]

        for _ in range(30):
            p = cls.minkowski_support(shape_a, shape_b, d)
            if np.dot(p, d) < 0:
                # No intersection possible
                return False

            simplex.append(p)
            # Update simplex and search direction
            if len(simplex) == 2:
                # Line segment case
                a, b = simplex[1], simplex[0]
                ab = b - a
                ao = -a
                d = np.cross(np.cross(ab, ao), ab)
                if np.linalg.norm(d) < 1e-8:
                    return True
            elif len(simplex) == 3:
                # Triangle case
                a, b, c = simplex[2], simplex[1], simplex[0]
                ab = b - a
                ac = c - a
                abc = np.cross(ab, ac)
                ao = -a
                d = abc if np.dot(abc, ao) > 0 else -abc
            elif len(simplex) == 4:
                # Tetrahedron contains origin
                return True

        return False
