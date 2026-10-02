from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math


@dataclass(frozen=True, slots=True)
class JointState:
    position: float
    velocity: float
    acceleration: float
    torque: float


@dataclass(frozen=True, slots=True)
class RigidBodyState:
    position: np.ndarray
    orientation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray
    angular_acceleration: np.ndarray


class InertiaTensor:
    def __init__(self, Ixx: float, Iyy: float, Izz: float,
                 Ixy: float = 0.0, Ixz: float = 0.0, Iyz: float = 0.0):
        self.tensor = np.array([
            [Ixx, -Ixy, -Ixz],
            [-Ixy, Iyy, -Iyz],
            [-Ixz, -Iyz, Izz]
        ], dtype=np.float64)

    @classmethod
    def from_sphere(cls, mass: float, radius: float) -> "InertiaTensor":
        I = 2.0 / 5.0 * mass * radius**2
        return cls(I, I, I)

    @classmethod
    def from_box(cls, mass: float, dx: float, dy: float, dz: float) -> "InertiaTensor":
        Ixx = mass / 12.0 * (dy**2 + dz**2)
        Iyy = mass / 12.0 * (dx**2 + dz**2)
        Izz = mass / 12.0 * (dx**2 + dy**2)
        return cls(Ixx, Iyy, Izz)

    @classmethod
    def from_cylinder(cls, mass: float, radius: float, height: float) -> "InertiaTensor":
        Ixx = mass / 12.0 * (3 * radius**2 + height**2)
        Iyy = Ixx
        Izz = mass / 2.0 * radius**2
        return cls(Ixx, Iyy, Izz)

    def to_world(self, rotation: np.ndarray) -> np.ndarray:
        return rotation @ self.tensor @ rotation.T


@dataclass(frozen=True, slots=True)
class SpatialInertia:
    mass: float
    com: np.ndarray
    inertia: InertiaTensor

    def spatial_matrix(self) -> np.ndarray:
        m = self.mass
        c = self.com
        I = self.inertia.tensor

        M = np.zeros((6, 6))
        M[:3, :3] = np.eye(3) * m
        M[:3, 3:] = -m * np.array([[0, -c[2], c[1]],
                                    [c[2], 0, -c[0]],
                                    [-c[1], c[0], 0]])
        M[3:, :3] = m * np.array([[0, -c[2], c[1]],
                                   [c[2], 0, -c[0]],
                                   [-c[1], c[0], 0]])
        M[3:, 3:] = I + m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))
        return M


class RigidBody:
    def __init__(
        self,
        body_id: str,
        mass: float,
        inertia: InertiaTensor,
        com: np.ndarray = None,
        position: np.ndarray = None,
        orientation: np.ndarray = None,
        linear_velocity: np.ndarray = None,
        angular_velocity: np.ndarray = None,
    ):
        self.body_id = body_id
        self.mass = mass
        self.inertia = inertia
        self.com = com if com is not None else np.zeros(3)
        self.position = position if position is not None else np.zeros(3)
        self.orientation = orientation if orientation is not None else np.eye(3)
        self.linear_velocity = linear_velocity if linear_velocity is not None else np.zeros(3)
        self.angular_velocity = angular_velocity if angular_velocity is not None else np.zeros(3)
        self.force = np.zeros(3)
        self.torque = np.zeros(3)

    @property
    def spatial_inertia(self) -> SpatialInertia:
        return SpatialInertia(self.mass, self.com, self.inertia)

    @property
    def world_inertia(self) -> np.ndarray:
        return self.orientation @ self.inertia.tensor @ self.orientation.T

    @property
    def world_inertia_inv(self) -> np.ndarray:
        return np.linalg.inv(self.world_inertia)

    def apply_force(self, force: np.ndarray, point: np.ndarray = None):
        self.force += force
        if point is not None:
            r = point - self.position
            self.torque += np.cross(r, force)

    def apply_torque(self, torque: np.ndarray):
        self.torque += torque

    def integrate(self, dt: float):
        self.linear_velocity += self.force / self.mass * dt
        self.position += self.linear_velocity * dt

        angular_acc = self.world_inertia_inv @ (self.torque - np.cross(self.angular_velocity, self.world_inertia @ self.angular_velocity))
        self.angular_velocity += angular_acc * dt

        omega_skew = np.array([[0, -self.angular_velocity[2], self.angular_velocity[1]],
                               [self.angular_velocity[2], 0, -self.angular_velocity[0]],
                               [-self.angular_velocity[1], self.angular_velocity[0], 0]])
        R_dot = omega_skew @ self.orientation
        self.orientation += R_dot * dt
        U, _, Vt = np.linalg.svd(self.orientation)
        self.orientation = U @ Vt

        self.force = np.zeros(3)
        self.torque = np.zeros(3)

    def get_kinetic_energy(self) -> float:
        trans = 0.5 * self.mass * np.dot(self.linear_velocity, self.linear_velocity)
        rot = 0.5 * np.dot(self.angular_velocity, self.world_inertia @ self.angular_velocity)
        return trans + rot

    def get_momentum(self) -> Tuple[np.ndarray, np.ndarray]:
        linear = self.mass * self.linear_velocity
        angular = self.world_inertia @ self.angular_velocity
        return linear, angular


class Joint:
    def __init__(
        self,
        joint_id: str,
        parent: RigidBody,
        child: RigidBody,
        joint_type: str = "revolute",
        axis: np.ndarray = None,
        limits: Tuple[float, float] = None,
        damping: float = 0.1,
        stiffness: float = 0.0,
    ):
        self.joint_id = joint_id
        self.parent = parent
        self.child = child
        self.joint_type = joint_type
        self.axis = axis if axis is not None else np.array([0, 0, 1.0])
        self.limits = limits
        self.damping = damping
        self.stiffness = stiffness
        self.position = 0.0
        self.velocity = 0.0
        self.torque = 0.0

    def get_relative_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.joint_type == "revolute":
            angle = self.position
            axis = self.axis
            c, s = math.cos(angle), math.sin(angle)
            R = np.array([
                [c + axis[0]**2 * (1-c), axis[0]*axis[1]*(1-c) - axis[2]*s, axis[0]*axis[2]*(1-c) + axis[1]*s],
                [axis[1]*axis[0]*(1-c) + axis[2]*s, c + axis[1]**2*(1-c), axis[1]*axis[2]*(1-c) - axis[0]*s],
                [axis[2]*axis[0]*(1-c) - axis[1]*s, axis[2]*axis[1]*(1-c) + axis[0]*s, c + axis[2]**2*(1-c)]
            ])
            return R, np.zeros(3)
        elif self.joint_type == "prismatic":
            return np.eye(3), self.axis * self.position
        return np.eye(3), np.zeros(3)

    def compute_jacobian(self) -> np.ndarray:
        if self.joint_type == "revolute":
            r = self.child.position - self.parent.position
            J_angular = self.axis
            J_linear = np.cross(self.axis, r)
            return np.concatenate([J_linear, J_angular])
        elif self.joint_type == "prismatic":
            return np.concatenate([self.axis, np.zeros(3)])
        return np.zeros(6)

    def apply_limits(self):
        if self.limits:
            self.position = max(self.limits[0], min(self.limits[1], self.position))


class MultibodySystem:
    def __init__(self):
        self.bodies: Dict[str, RigidBody] = {}
        self.joints: Dict[str, Joint] = {}
        self.gravity = np.array([0, -9.81, 0])
        self.dt = 1/60

    def add_body(self, body: RigidBody):
        self.bodies[body.body_id] = body

    def add_joint(self, joint: Joint):
        self.joints[joint.joint_id] = joint

    def step(self):
        self._compute_forces()
        self._integrate()

    def _compute_forces(self):
        for body in self.bodies.values():
            body.force = body.mass * self.gravity

        for joint in self.joints.values():
            if joint.stiffness > 0:
                spring_torque = -joint.stiffness * joint.position
                joint.torque += spring_torque
            if joint.damping > 0:
                damper_torque = -joint.damping * joint.velocity
                joint.torque += damper_torque

            tau = joint.torque
            r = joint.child.position - joint.parent.position
            joint.parent.force += np.cross(joint.axis, tau)
            joint.parent.torque += tau
            joint.child.force -= np.cross(joint.axis, tau)
            joint.child.torque -= tau

    def _integrate(self):
        for body in self.bodies.values():
            body.integrate(self.dt)

        for joint in self.joints.values():
            if joint.joint_type == "revolute":
                joint.velocity = np.dot(joint.axis, joint.child.angular_velocity - joint.parent.angular_velocity)
                joint.position += joint.velocity * self.dt
                joint.apply_limits()
            elif joint.joint_type == "prismatic":
                joint.velocity = np.dot(joint.axis, joint.child.linear_velocity - joint.parent.linear_velocity)
                joint.position += joint.velocity * self.dt
                joint.apply_limits()

    def get_total_energy(self) -> float:
        total = 0.0
        for body in self.bodies.values():
            total += body.get_kinetic_energy()
            total += body.mass * 9.81 * body.position[1]
        return total


class RobotDynamics:
    def __init__(
        self,
        num_joints: int,
        link_masses: List[float],
        link_coms: List[np.ndarray],
        link_inertias: List[InertiaTensor],
        joint_axes: List[np.ndarray],
        joint_types: List[str],
    ):
        self.num_joints = num_joints
        self.link_masses = link_masses
        self.link_coms = link_coms
        self.link_inertias = link_inertias
        self.joint_axes = joint_axes
        self.joint_types = joint_types
        self.gravity = np.array([0, -9.81, 0])

    def forward_kinematics(self, q: np.ndarray) -> List[np.ndarray]:
        transforms = [np.eye(4)]
        T = np.eye(4)
        for i in range(self.num_joints):
            if self.joint_types[i] == "revolute":
                angle = q[i]
                axis = self.joint_axes[i]
                c, s = math.cos(angle), math.sin(angle)
                R = np.array([
                    [c + axis[0]**2*(1-c), axis[0]*axis[1]*(1-c) - axis[2]*s, axis[0]*axis[2]*(1-c) + axis[1]*s],
                    [axis[1]*axis[0]*(1-c) + axis[2]*s, c + axis[1]**2*(1-c), axis[1]*axis[2]*(1-c) - axis[0]*s],
                    [axis[2]*axis[0]*(1-c) - axis[1]*s, axis[2]*axis[1]*(1-c) + axis[0]*s, c + axis[2]**2*(1-c)]
                ])
            else:
                R = np.eye(3)
            T_joint = np.eye(4)
            T_joint[:3, :3] = R
            T = T @ T_joint
            transforms.append(T)
        return transforms

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        transforms = self.forward_kinematics(q)
        J = np.zeros((6, self.num_joints))
        for i in range(self.num_joints):
            T_i = transforms[i]
            z_i = T_i[:3, 2]
            p_i = T_i[:3, 3]
            p_ee = transforms[-1][:3, 3]
            J[:3, i] = np.cross(z_i, p_ee - p_i)
            J[3:, i] = z_i
        return J

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        n = self.num_joints
        M = np.zeros((n, n))
        transforms = self.forward_kinematics(q)

        for i in range(n):
            for j in range(n):
                J_i = self._joint_jacobian(i, q, transforms)
                J_j = self._joint_jacobian(j, q, transforms)
                M[i, j] = self._compute_inertia_term(J_i, J_j, i, j)

        return M

    def _joint_jacobian(self, joint_idx: int, q: np.ndarray, transforms: List[np.ndarray]) -> np.ndarray:
        J = np.zeros(6)
        T_j = transforms[joint_idx]
        z = T_j[:3, 2]
        p = T_j[:3, 3]
        p_ee = transforms[-1][:3, 3]
        J[:3] = np.cross(z, p_ee - p)
        J[3:] = z
        return J

    def _compute_inertia_term(self, J_i: np.ndarray, J_j: np.ndarray, i: int, j: int) -> float:
        if i == j:
            return self.link_masses[i] * np.dot(J_i[:3], J_i[:3]) + np.dot(J_i[3:], self.link_inertias[i].tensor @ J_i[3:])
        return 0.0

    def coriolis_matrix(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        n = self.num_joints
        C = np.zeros((n, n))
        M = self.mass_matrix(q)
        eps = 1e-6

        for i in range(n):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[i] += eps
            q_minus[i] -= eps

            M_plus = self.mass_matrix(q_plus)
            M_minus = self.mass_matrix(q_minus)
            dM_dqi = (M_plus - M_minus) / (2 * eps)

            for j in range(n):
                for k in range(n):
                    C[j, k] += 0.5 * (dM_dqi[j, k] + dM_dqi[k, j] - dM_dqi[k, j]) * qd[i]

        return C

    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        n = self.num_joints
        G = np.zeros(n)
        transforms = self.forward_kinematics(q)

        for i in range(n):
            T = transforms[i + 1]
            J = self._joint_jacobian(i, q, transforms)
            m = self.link_masses[i]
            G += m * J[:3] @ self.gravity

        return G

    def inverse_dynamics(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, qd)
        G = self.gravity_vector(q)
        return M @ qdd + C @ qd + G

    def forward_dynamics(self, q: np.ndarray, qd: np.ndarray, tau: np.ndarray) -> np.ndarray:
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, qd)
        G = self.gravity_vector(q)
        return np.linalg.solve(M, tau - C @ qd - G)


class TrajectoryOptimizer:
    def __init__(
        self,
        robot: RobotDynamics,
        dt: float = 0.01,
        horizon: int = 100,
    ):
        self.robot = robot
        self.dt = dt
        self.horizon = horizon

    def optimize(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        qd_start: np.ndarray = None,
        qd_goal: np.ndarray = None,
        obstacles: List[Callable] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if qd_start is None:
            qd_start = np.zeros_like(q_start)
        if qd_goal is None:
            qd_goal = np.zeros_like(q_goal)

        n = self.robot.num_joints
        N = self.horizon

        Q = np.eye(n) * 10.0
        R = np.eye(n) * 0.1
        Qf = np.eye(n) * 100.0

        X = np.zeros((N + 1, n))
        U = np.zeros((N, n))

        for i in range(N + 1):
            alpha = i / N
            X[i] = (1 - alpha) * q_start + alpha * q_goal

        for _ in range(50):
            for i in range(N):
                q = X[i]
                q_next = X[i + 1]
                U[i] = (q_next - q) / self.dt

            for i in range(N):
                q = X[i]
                u = U[i]
                tau = self.robot.inverse_dynamics(q, U[i], (U[i+1] - U[i]) / self.dt if i < N-1 else np.zeros_like(U[i]))
                U[i] = tau

            for i in range(N):
                qdd = (U[i+1] - U[i]) / self.dt if i < N-1 else np.zeros_like(U[i])
                X[i+1] = X[i] + self.dt * U[i] + 0.5 * self.dt**2 * qdd

        return X.tolist(), U.tolist()

    def check_collision(self, q: np.ndarray, obstacles: List[Callable]) -> bool:
        if obstacles is None:
            return False
        for obs in obstacles:
            if obs(q):
                return True
        return False


class ImpedanceController:
    def __init__(
        self,
        stiffness: np.ndarray,
        damping: np.ndarray,
        inertia: np.ndarray = None,
    ):
        self.K = stiffness
        self.D = damping
        self.M = inertia if inertia is not None else np.eye(len(stiffness))

    def compute_torque(self, q: np.ndarray, qd: np.ndarray, q_des: np.ndarray, qd_des: np.ndarray, qdd_des: np.ndarray) -> np.ndarray:
        e = q_des - q
        ed = qd_des - qd
        return self.K @ e + self.D @ ed + self.M @ qdd_des


class AdmittanceController:
    def __init__(
        self,
        M: np.ndarray,
        D: np.ndarray,
        K: np.ndarray,
    ):
        self.M = M
        self.D = D
        self.K = K
        self.x = np.zeros_like(M[0])
        self.xd = np.zeros_like(M[0])

    def update(self, F_ext: np.ndarray, dt: float) -> np.ndarray:
        xdd = np.linalg.inv(self.M) @ (F_ext - self.D @ self.xd - self.K @ self.x)
        self.xd += xdd * dt
        self.x += self.xd * dt
        return self.x


class ForceController:
    def __init__(
        self,
        K_f: float,
        D_f: float,
        target_force: float,
    ):
        self.K_f = K_f
        self.D_f = D_f
        self.target_force = target_force
        self.integral = 0.0

    def compute_position(self, measured_force: float, dt: float) -> float:
        error = self.target_force - measured_force
        self.integral += error * dt
        return self.K_f * error + self.D_f * error / dt + self.integral


class HybridForcePositionController:
    def __init__(
        self,
        position_controller: BaseController,
        force_controller: ForceController,
        selection_matrix: np.ndarray,
    ):
        self.position_controller = position_controller
        self.force_controller = force_controller
        self.S = selection_matrix

    def compute(self, x: np.ndarray, x_des: np.ndarray, F_measured: np.ndarray, F_des: np.ndarray, dt: float) -> np.ndarray:
        pos_out = self.position_controller.update(x_des, x, dt)
        force_error = F_des - F_measured
        force_out = self.force_controller.compute_position(force_error[0], dt)

        return (np.eye(6) - self.S) @ pos_out + self.S @ np.array([force_out] * 6)


class BaseController(ABC):
    @abstractmethod
    def update(self, setpoint: float, measured: float, dt: float) -> float:
        pass


class PIDController(BaseController):
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, setpoint: float, measured: float, dt: float) -> float:
        error = setpoint - measured
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class SlidingModeController:
    def __init__(self, lambda_: float, gain: float, phi: float = 0.01):
        self.lambda_ = lambda_
        self.gain = gain
        self.phi = phi

    def compute(self, error: float, error_dot: float) -> float:
        s = error_dot + self.lambda_ * error
        if abs(s) > self.phi:
            return -self.gain * np.sign(s)
        return -self.gain * s / self.phi