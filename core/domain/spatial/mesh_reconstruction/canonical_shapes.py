from __future__ import annotations

import numpy as np


class CanonicalShapeGenerator:
    """Generates dense 3D point cloud surfaces for vision recognition classes and geometric primitives."""

    @staticmethod
    def sphere(radius: float = 0.5, resolution: int = 24) -> np.ndarray:
        """Sample points uniformly on a 3D sphere surface."""
        u = np.linspace(0, 2 * np.pi, resolution * 2)
        v = np.linspace(0, np.pi, resolution)
        x = (radius * np.outer(np.cos(u), np.sin(v))).flatten()
        y = (radius * np.outer(np.sin(u), np.sin(v))).flatten()
        z = (radius * np.outer(np.ones(np.size(u)), np.cos(v))).flatten()
        return np.stack([x, y, z], axis=-1)

    @staticmethod
    def cylinder(radius: float = 0.3, height: float = 0.8, resolution: int = 20) -> np.ndarray:
        """Sample points on a 3D cylinder surface including top/bottom caps."""
        theta = np.linspace(0, 2 * np.pi, resolution * 2)
        z_vals = np.linspace(-height / 2, height / 2, resolution)
        t_grid, z_grid = np.meshgrid(theta, z_vals)
        x = (radius * np.cos(t_grid)).flatten()
        y = (radius * np.sin(t_grid)).flatten()
        z = z_grid.flatten()

        # End caps
        r_caps = np.linspace(0, radius, resolution // 2)
        t_caps = np.linspace(0, 2 * np.pi, resolution)
        r_g, t_g = np.meshgrid(r_caps, t_caps)
        cap_x = (r_g * np.cos(t_g)).flatten()
        cap_y = (r_g * np.sin(t_g)).flatten()
        cap_z_top = np.full_like(cap_x, height / 2)
        cap_z_bot = np.full_like(cap_x, -height / 2)

        pts = np.vstack([
            np.stack([x, y, z], axis=-1),
            np.stack([cap_x, cap_y, cap_z_top], axis=-1),
            np.stack([cap_x, cap_y, cap_z_bot], axis=-1),
        ])
        return pts

    @staticmethod
    def box(width: float = 0.6, depth: float = 0.6, height: float = 0.8, resolution: int = 12) -> np.ndarray:
        """Sample points on a 3D rectangular box surface."""
        hw, hd, hh = width / 2, depth / 2, height / 2
        pts = []

        # Front / Back (Z normal)
        u, v = np.meshgrid(np.linspace(-hw, hw, resolution), np.linspace(-hh, hh, resolution))
        pts.append(np.stack([u.flatten(), np.full_like(u.flatten(), -hd), v.flatten()], axis=-1))
        pts.append(np.stack([u.flatten(), np.full_like(u.flatten(), hd), v.flatten()], axis=-1))

        # Left / Right (X normal)
        u, v = np.meshgrid(np.linspace(-hd, hd, resolution), np.linspace(-hh, hh, resolution))
        pts.append(np.stack([np.full_like(u.flatten(), -hw), u.flatten(), v.flatten()], axis=-1))
        pts.append(np.stack([np.full_like(u.flatten(), hw), u.flatten(), v.flatten()], axis=-1))

        # Top / Bottom (Y normal)
        u, v = np.meshgrid(np.linspace(-hw, hw, resolution), np.linspace(-hd, hd, resolution))
        pts.append(np.stack([u.flatten(), v.flatten(), np.full_like(u.flatten(), -hh)], axis=-1))
        pts.append(np.stack([u.flatten(), v.flatten(), np.full_like(u.flatten(), hh)], axis=-1))

        return np.concatenate(pts, axis=0)

    @classmethod
    def generate_object(cls, class_name: str) -> np.ndarray:
        """Generates synthetic 3D point cloud corresponding to common computer vision classes."""
        c = class_name.lower().strip()

        if c in ("water_bottle", "bottle", "can"):
            # Body cylinder + neck
            body = cls.cylinder(radius=0.18, height=0.6, resolution=16)
            neck = cls.cylinder(radius=0.08, height=0.18, resolution=12)
            neck[:, 2] += 0.38
            return np.vstack([body, neck])

        if c in ("chair", "furniture"):
            # Seat
            seat = cls.box(width=0.5, depth=0.5, height=0.06, resolution=10)
            seat[:, 2] += 0.0
            # Backrest
            back = cls.box(width=0.5, depth=0.06, height=0.5, resolution=10)
            back[:, 1] += 0.22
            back[:, 2] += 0.28
            # 4 Legs
            legs = []
            for dx in (-0.2, 0.2):
                for dy in (-0.2, 0.2):
                    leg = cls.cylinder(radius=0.03, height=0.45, resolution=8)
                    leg[:, 0] += dx
                    leg[:, 1] += dy
                    leg[:, 2] -= 0.25
                    legs.append(leg)
            return np.vstack([seat, back, *legs])

        if c in ("mobile_phone", "cell phone", "phone"):
            # Sleek thin rectangular slab
            return cls.box(width=0.22, depth=0.03, height=0.45, resolution=12)

        if c in ("helmet", "headwear"):
            # Semi-sphere dome
            sphere_pts = cls.sphere(radius=0.35, resolution=18)
            dome = sphere_pts[sphere_pts[:, 2] >= -0.05]
            rim = cls.cylinder(radius=0.38, height=0.05, resolution=14)
            rim[:, 2] -= 0.06
            return np.vstack([dome, rim])

        if c in ("knife", "blade", "pen"):
            # Elongated handle and thin tip
            handle = cls.cylinder(radius=0.03, height=0.3, resolution=10)
            handle[:, 2] -= 0.15
            tip = cls.cylinder(radius=0.015, height=0.3, resolution=8)
            tip[:, 2] += 0.15
            return np.vstack([handle, tip])

        if c in ("human_face", "person", "face"):
            # Ellipsoid with nose protrusion
            head = cls.sphere(radius=0.35, resolution=18)
            head[:, 1] *= 1.25  # Height stretch
            nose = cls.cylinder(radius=0.05, height=0.1, resolution=8)
            nose[:, 0] += 0.0
            nose[:, 1] += 0.0
            nose[:, 2] += 0.35
            return np.vstack([head, nose])

        # Default fallback: unit sphere
        return cls.sphere(radius=0.45, resolution=18)
