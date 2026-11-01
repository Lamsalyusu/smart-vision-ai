from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DHParameter:
    """Denavit-Hartenberg (DH) parameter representation of a kinematic joint link."""
    a: float      # Link length along x_i
    alpha: float  # Link twist angle about x_i (radians)
    d: float      # Link offset along z_{i-1}
    theta: float  # Joint angle about z_{i-1} (radians)

    def transformation_matrix(self, q_offset: float = 0.0) -> np.ndarray:
        """Compute standard 4x4 homogeneous transformation matrix A_i."""
        t = self.theta + q_offset
        ct, st = math.cos(t), math.sin(t)
        ca, sa = math.cos(self.alpha), math.sin(self.alpha)

        return np.array([
            [ct, -st * ca,  st * sa, self.a * ct],
            [st,  ct * ca, -ct * sa, self.a * st],
            [0.0,      sa,       ca,      self.d],
            [0.0,     0.0,      0.0,         1.0]
        ], dtype=np.float64)


class SerialManipulatorKinematics:
    """Kinematics engine for n-DoF serial articulated robot arms (e.g., UR5, PUMA)."""

    def __init__(self, dh_params: list[DHParameter]) -> None:
        self.dh_params = dh_params
        self.dof = len(dh_params)

    def forward_kinematics(self, joint_angles: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Compute end-effector pose T_0_n and intermediate coordinate frames."""
        t_cum = np.eye(4, dtype=np.float64)
        frames = [t_cum.copy()]

        for i in range(self.dof):
            q_val = float(joint_angles[i])
            t_link = self.dh_params[i].transformation_matrix(q_offset=q_val)
            t_cum = t_cum @ t_link
            frames.append(t_cum.copy())

        return t_cum, frames

    def numerical_inverse_kinematics_dls(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q_init: np.ndarray,
        damping: float = 0.05,
        max_iters: int = 100,
        tol_pos: float = 1e-4,
        tol_rot: float = 1e-3,
    ) -> tuple[np.ndarray, bool]:
        """Solves 6-DoF inverse kinematics using Damped Least Squares (Levenberg-Marquardt)."""
        q = np.array(q_init, dtype=np.float64).copy()

        for iteration in range(max_iters):
            t_ee, _ = self.forward_kinematics(q)
            current_pos = t_ee[:3, 3]
            current_rot = t_ee[:3, :3]

            pos_err = target_pos - current_pos
            # Orientation error vector from rotation matrix difference
            rot_err_mat = target_rot @ current_rot.T
            from ..slam.lie_algebra import SO3
            rot_err = SO3.log(rot_err_mat)

            err = np.concatenate([pos_err, rot_err])
            if np.linalg.norm(pos_err) < tol_pos and np.linalg.norm(rot_err) < tol_rot:
                return q, True

            # Compute Geometric Jacobian J
            j = self.compute_jacobian(q)
            # DLS formulation: delta_q = J^T * (J * J^T + lambda^2 * I)^-1 * err
            j_jt = j @ j.T
            damped_inv = np.linalg.inv(j_jt + (damping ** 2) * np.eye(6))
            delta_q = j.T @ damped_inv @ err

            q += delta_q

        return q, False

    def compute_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute the 6xN Geometric Jacobian matrix."""
        _, frames = self.forward_kinematics(joint_angles)
        p_ee = frames[-1][:3, 3]
        j = np.zeros((6, self.dof), dtype=np.float64)

        for i in range(self.dof):
            # Joint axis z_{i-1} in base frame
            z_axis = frames[i][:3, 2]
            p_joint = frames[i][:3, 3]

            # Revolute joint: J_v = z_{i-1} x (p_ee - p_{i-1}), J_omega = z_{i-1}
            j[:3, i] = np.cross(z_axis, p_ee - p_joint)
            j[3:, i] = z_axis

        return j
