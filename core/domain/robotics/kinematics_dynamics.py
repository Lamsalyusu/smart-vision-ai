from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math
import time
import heapq


@dataclass(frozen=True, slots=True)
class JointState:
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    torques: np.ndarray
    names: List[str] = field(default_factory=list)

    def __post_init__(self):
        n = len(self.positions)
        for arr in [self.velocities, self.accelerations, self.torques]:
            if len(arr) != n:
                raise ValueError("All arrays must have same length")
        if len(self.names) != n:
            raise ValueError("Names must match joint count")


@dataclass(frozen=True, slots=True)
class EndEffectorState:
    position: np.ndarray
    orientation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    force: np.ndarray
    torque: np.ndarray


class ForwardKinematics(ABC):
    @abstractmethod
    def compute(self, joint_positions: np.ndarray) -> EndEffectorState:
        pass

    @abstractmethod
    def compute_jacobian(self, joint_positions: np.ndarray) -> np.ndarray:
        pass


class SerialChainFK(ForwardKinematics):
    def __init__(
        self,
        dh_params: List[Tuple[float, float, float, float]],
        base_transform: Optional[np.ndarray] = None,
    ):
        self.dh_params = dh_params
        self.base_transform = base_transform if base_transform is not None else np.eye(4)
        self.num_joints = len(dh_params)

    def _dh_transform(self, theta: float, d: float, a: float, alpha: float) -> np.ndarray:
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,     d],
            [0,   0,      0,      1],
        ], dtype=np.float64)

    def compute(self, joint_positions: np.ndarray) -> EndEffectorState:
        T = self.base_transform.copy()

        for i, (theta, d, a, alpha) in enumerate(self.dh_params):
            q = joint_positions[i] if i < len(joint_positions) else 0
            T = T @ self._dh_transform(q + theta, d, a, alpha)

        position = T[:3, 3]
        orientation = self._rotation_matrix_to_quaternion(T[:3, :3])
        linear_vel = np.zeros(3)
        angular_vel = np.zeros(3)
        force = np.zeros(3)
        torque = np.zeros(3)

        return EndEffectorState(position, orientation, linear_vel, angular_vel, force, torque)

    def compute_jacobian(self, joint_positions: np.ndarray) -> np.ndarray:
        J = np.zeros((6, self.num_joints))
        T = self.base_transform.copy()
        positions = [T[:3, 3]]
        z_axes = [T[:3, 2]]

        for i, (theta, d, a, alpha) in enumerate(self.dh_params):
            q = joint_positions[i] if i < len(joint_positions) else 0
            T = T @ self._dh_transform(q + theta, d, a, alpha)
            positions.append(T[:3, 3])
            z_axes.append(T[:3, 2])

        end_pos = positions[-1]

        for i in range(self.num_joints):
            z = z_axes[i]
            p = positions[i]
            J[:3, i] = np.cross(z, end_pos - p)
            J[3:, i] = z

        return J

    def compute_all_frames(self, joint_positions: np.ndarray) -> List[np.ndarray]:
        frames = []
        T = self.base_transform.copy()
        frames.append(T)

        for i, (theta, d, a, alpha) in enumerate(self.dh_params):
            q = joint_positions[i] if i < len(joint_positions) else 0
            T = T @ self._dh_transform(q + theta, d, a, alpha)
            frames.append(T)

        return frames

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        trace = np.trace(R)
        if trace > 0:
            s = 2.0 * math.sqrt(trace + 1.0)
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[1, 0] + R[0, 1]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        return np.array([qx, qy, qz, qw])


class InverseKinematics(ABC):
    @abstractmethod
    def solve(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        initial_guess: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        pass


class NumericalIK(InverseKinematics):
    def __init__(
        self,
        fk: ForwardKinematics,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        damping: float = 1e-3,
        joint_limits: Optional[List[Tuple[float, float]]] = None,
    ):
        self.fk = fk
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping = damping
        self.joint_limits = joint_limits

    def solve(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        initial_guess: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        n = self.fk.num_joints
        q = initial_guess if initial_guess is not None else np.zeros(n)

        for _ in range(self.max_iterations):
            state = self.fk.compute(q)
            J = self.fk.compute_jacobian(q)

            pos_error = target_position - state.position
            error = np.concatenate([pos_error, np.zeros(3)])

            if target_orientation is not None:
                orn_error = self._orientation_error(state.orientation, target_orientation)
                error[3:] = orn_error

            if np.linalg.norm(error) < self.tolerance:
                return q

            J_damped = J.T @ np.linalg.inv(J @ J.T + self.damping**2 * np.eye(6))
            q = q + J_damped @ error

            if self.joint_limits:
                for i, (low, high) in enumerate(self.joint_limits):
                    q[i] = np.clip(q[i], low, high)

        state = self.fk.compute(q)
        pos_error = target_position - state.position
        if np.linalg.norm(pos_error) < self.tolerance * 10:
            return q
        return None

    def _orientation_error(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        qc = current / (np.linalg.norm(current) + 1e-8)
        qt = target / (np.linalg.norm(target) + 1e-8)
        q_err = self._quaternion_multiply(qt, self._quaternion_conjugate(qc))
        return 2 * q_err[:3]

    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
        w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ])

    def _quaternion_conjugate(self, q: np.ndarray) -> np.ndarray:
        return np.array([-q[0], -q[1], -q[2], q[3]])


class AnalyticalIK(InverseKinematics):
    def __init__(self, robot_type: str = "manipulator"):
        self.robot_type = robot_type

    def solve(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        initial_guess: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        if self.robot_type == "manipulator":
            return self._solve_6dof(target_position, target_orientation)
        return None

    def _solve_6dof(self, pos: np.ndarray, orn: Optional[np.ndarray]) -> Optional[np.ndarray]:
        x, y, z = pos
        d1, a2, a3, d4 = 0.1, 0.5, 0.5, 0.1

        r = math.sqrt(x**2 + y**2)
        if r < 1e-6:
            return None

        q1 = math.atan2(y, x)
        q234 = 0

        if orn is not None:
            R = self._quaternion_to_matrix(orn)
            q234 = math.atan2(R[1, 0]*math.cos(q1) + R[1, 1]*math.sin(q1), R[0, 0]*math.cos(q1) + R[0, 1]*math.sin(q1))

        r_wrist = r - d4 * math.sin(q234)
        z_wrist = z - d1 - d4 * math.cos(q234)

        D = (r_wrist**2 + z_wrist**2 - a2**2 - a3**2) / (2 * a2 * a3)
        if abs(D) > 1:
            return None

        q3 = math.atan2(-math.sqrt(1 - D**2), D)

        q2 = math.atan2(z_wrist, r_wrist) - math.atan2(a3 * math.sin(q3), a2 + a3 * math.cos(q3))

        return np.array([q1, q2, q3, q234 - q2 - q3, 0, 0])

    def _quaternion_to_matrix(self, q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        return np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y],
        ])


class DynamicsModel(ABC):
    @abstractmethod
    def forward_dynamics(
        self,
        state: JointState,
        torques: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_dynamics(
        self,
        state: JointState,
        desired_acc: np.ndarray,
    ) -> np.ndarray:
        pass


class RigidBodyDynamics(DynamicsModel):
    def __init__(
        self,
        masses: List[float],
        inertias: List[np.ndarray],
        com_positions: List[np.ndarray],
        fk: ForwardKinematics,
        gravity: np.ndarray = None,
    ):
        self.masses = masses
        self.inertias = inertias
        self.com_positions = com_positions
        self.fk = fk
        self.gravity = gravity if gravity is not None else np.array([0, 0, -9.81])
        self.num_bodies = len(masses)

    def forward_dynamics(self, state: JointState, torques: np.ndarray) -> np.ndarray:
        M = self._mass_matrix(state.positions)
        C = self._coriolis_matrix(state.positions, state.velocities)
        G = self._gravity_vector(state.positions)

        acc = np.linalg.solve(M, torques - C @ state.velocities - G)
        return acc

    def inverse_dynamics(self, state: JointState, desired_acc: np.ndarray) -> np.ndarray:
        M = self._mass_matrix(state.positions)
        C = self._coriolis_matrix(state.positions, state.velocities)
        G = self._gravity_vector(state.positions)

        torques = M @ desired_acc + C @ state.velocities + G
        return torques

    def _mass_matrix(self, q: np.ndarray) -> np.ndarray:
        n = len(q)
        M = np.zeros((n, n))
        frames = self.fk.compute_all_frames(q)

        for i in range(self.num_bodies):
            if i + 1 >= len(frames):
                break
            T = frames[i + 1]
            J = self._compute_jacobian_com(i, q, frames)

            m = self.masses[i]
            I = self.inertias[i]

            M += m * J[:3, :].T @ J[:3, :] + J[3:, :].T @ I @ J[3:, :]

        return M

    def _coriolis_matrix(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        n = len(q)
        C = np.zeros((n, n))

        M_q = self._mass_matrix
        eps = 1e-6

        for i in range(n):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[i] += eps
            q_minus[i] -= eps

            M_plus = M_q(q_plus)
            M_minus = M_q(q_minus)
            dM_dqi = (M_plus - M_minus) / (2 * eps)

            for j in range(n):
                for k in range(n):
                    C[j, k] += 0.5 * (dM_dqi[j, k] + dM_dqi[k, j] - dM_dqi[k, j]) * qd[i]

        return C

    def _gravity_vector(self, q: np.ndarray) -> np.ndarray:
        n = len(q)
        G = np.zeros(n)
        frames = self.fk.compute_all_frames(q)

        for i in range(self.num_bodies):
            if i + 1 >= len(frames):
                break
            T = frames[i + 1]
            J = self._compute_jacobian_com(i, q, frames)

            m = self.masses[i]
            G += m * J[:3, :].T @ self.gravity

        return G

    def _compute_jacobian_com(self, link_idx: int, q: np.ndarray, frames: List) -> np.ndarray:
        n = len(q)
        J = np.zeros((6, n))

        T_com = frames[link_idx + 1]
        p_com = T_com[:3, 3]
        R_com = T_com[:3, :3]
        com_local = self.com_positions[link_idx]
        p_com = T_com[:3, :3] @ com_local + T_com[:3, 3]

        for i in range(link_idx + 1):
            if i >= n:
                break
            z_i = frames[i][:3, 2]
            p_i = frames[i][:3, 3]
            J[:3, i] = np.cross(z_i, p_com - p_i)
            J[3:, i] = z_i

        return J


class TrajectoryPlanner:
    def __init__(
        self,
        max_velocity: np.ndarray,
        max_acceleration: np.ndarray,
        max_jerk: Optional[np.ndarray] = None,
    ):
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk if max_jerk is not None else max_acceleration * 10

    def plan_joint_trajectory(
        self,
        start: JointState,
        goal: JointState,
        duration: float,
        dt: float = 0.01,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        n = len(start.positions)
        num_steps = int(duration / dt) + 1

        positions = []
        velocities = []
        accelerations = []

        for i in range(n):
            pos_traj, vel_traj, acc_traj = self._plan_1d(
                start.positions[i], start.velocities[i], start.accelerations[i],
                goal.positions[i], goal.velocities[i], goal.accelerations[i],
                duration, dt,
                self.max_velocity[i], self.max_acceleration[i], self.max_jerk[i],
            )
            if i == 0:
                positions = [[] for _ in range(num_steps)]
                velocities = [[] for _ in range(num_steps)]
                accelerations = [[] for _ in range(num_steps)]

            for t in range(num_steps):
                positions[t].append(pos_traj[t])
                velocities[t].append(vel_traj[t])
                accelerations[t].append(acc_traj[t])

        return (np.array(positions), np.array(velocities), np.array(accelerations))

    def _plan_1d(
        self,
        p0, v0, a0, pf, vf, af,
        duration, dt,
        vmax, amax, jmax,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_steps = int(duration / dt) + 1
        t = np.linspace(0, duration, num_steps)

        coeffs = self._quintic_coeffs(p0, v0, a0, pf, vf, af, duration)

        pos = coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 + coeffs[3]*t**3 + coeffs[4]*t**4 + coeffs[5]*t**5
        vel = coeffs[1] + 2*coeffs[2]*t + 3*coeffs[3]*t**2 + 4*coeffs[4]*t**3 + 5*coeffs[5]*t**4
        acc = 2*coeffs[2] + 6*coeffs[3]*t + 12*coeffs[4]*t**2 + 20*coeffs[5]*t**3

        pos = np.clip(pos, p0 - abs(pf-p0)*2, pf + abs(pf-p0)*2)
        vel = np.clip(vel, -vmax, vmax)
        acc = np.clip(acc, -amax, amax)

        return pos, vel, acc

    def _quintic_coeffs(self, p0, v0, a0, pf, vf, af, T):
        T2, T3, T4, T5 = T**2, T**3, T**4, T**5
        a00 = pf - p0 - v0*T - 0.5*a0*T2
        a10 = vf - v0 - a0*T
        a20 = af - a0

        c3 = (10*a00 - 4*a10*T + 0.5*a20*T2) / T3
        c4 = (-15*a00 + 7*a10*T - a20*T2) / T4
        c5 = (6*a00 - 3*a10*T + 0.5*a20*T2) / T5

        return np.array([p0, v0, 0.5*a0, c3, c4, c5])

    def plan_cartesian_trajectory(
        self,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        goal_pos: np.ndarray,
        goal_orn: np.ndarray,
        duration: float,
        dt: float = 0.01,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        num_steps = int(duration / dt) + 1
        t = np.linspace(0, duration, num_steps)

        positions = []
        orientations = []

        for ti in t:
            s = self._smooth_step(ti / duration)
            pos = start_pos + s * (goal_pos - start_pos)
            orn = self._slerp(start_orn, goal_orn, s)
            positions.append(pos)
            orientations.append(orn)

        return positions, orientations

    def _smooth_step(self, t: float) -> float:
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _slerp(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        q1 = q1 / (np.linalg.norm(q1) + 1e-8)
        q2 = q2 / (np.linalg.norm(q2) + 1e-8)

        dot = np.dot(q1, q2)
        if dot < 0:
            q2 = -q2
            dot = -dot

        if dot > 0.9995:
            return q1 + t * (q2 - q1)

        theta = math.acos(np.clip(dot, -1, 1))
        sin_theta = math.sin(theta)

        w1 = math.sin((1 - t) * theta) / sin_theta
        w2 = math.sin(t * theta) / sin_theta

        return w1 * q1 + w2 * q2


class MotionPlanner:
    def __init__(
        self,
        robot_type: str = "manipulator",
        collision_checker: Optional[Callable] = None,
    ):
        self.robot_type = robot_type
        self.collision_checker = collision_checker

    def plan_rrt(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        bounds: List[Tuple[float, float]],
        max_iterations: int = 5000,
        step_size: float = 0.1,
        goal_bias: float = 0.1,
    ) -> Optional[List[np.ndarray]]:
        if self.collision_checker and self.collision_checker(start):
            return None
        if self.collision_checker and self.collision_checker(goal):
            return None

        tree = {tuple(start): None}
        nodes = [start.copy()]

        for _ in range(max_iterations):
            if np.random.random() < 0.1:
                sample = goal.copy()
            else:
                sample = np.array([np.random.uniform(low, high) for low, high in bounds])

            nearest_idx = self._nearest_node(nodes, sample)
            nearest = nodes[nearest_idx]

            direction = sample - nearest
            dist = np.linalg.norm(direction)
            if dist > step_size:
                new_node = nearest + direction / dist * step_size
            else:
                new_node = sample

            if self.collision_checker and self.collision_checker(new_node):
                continue

            new_tuple = tuple(new_node)
            tree[new_tuple] = tuple(nearest)
            nodes.append(new_node)

            if np.linalg.norm(new_node - goal) < step_size:
                tree[tuple(goal)] = new_tuple
                return self._extract_path(tree, start, goal)

        return None

    def _nearest_node(self, nodes: List[np.ndarray], target: np.ndarray) -> int:
        dists = [np.linalg.norm(n - target) for n in nodes]
        return int(np.argmin(dists))

    def _extract_path(
        self,
        tree: Dict,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> List[np.ndarray]:
        path = [goal]
        current = tuple(goal)
        while current != tuple(start):
            current = tree[current]
            if current is None:
                break
            path.append(np.array(current))
        path.reverse()
        return path


class Controller(ABC):
    @abstractmethod
    def compute(
        self,
        current_state: JointState,
        desired_state: JointState,
        dt: float,
    ) -> np.ndarray:
        pass


class PIDController(Controller):
    def __init__(
        self,
        kp: np.ndarray,
        ki: np.ndarray,
        kd: np.ndarray,
        integral_limit: Optional[np.ndarray] = None,
        output_limit: Optional[np.ndarray] = None,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.integral = np.zeros_like(kp)
        self.prev_error = np.zeros_like(kp)

    def compute(
        self,
        current_state: JointState,
        desired_state: JointState,
        dt: float,
    ) -> np.ndarray:
        error = desired_state.positions - current_state.positions
        self.integral += error * dt

        if self.integral_limit is not None:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        if self.output_limit is not None:
            output = np.clip(output, -self.output_limit, self.output_limit)

        return output

    def reset(self):
        self.integral = np.zeros_like(self.kp)
        self.prev_error = np.zeros_like(self.kp)


class ComputedTorqueController(Controller):
    def __init__(
        self,
        dynamics: DynamicsModel,
        kp: np.ndarray,
        kv: np.ndarray,
    ):
        self.dynamics = dynamics
        self.kp = kp
        self.kv = kv

    def compute(
        self,
        current_state: JointState,
        desired_state: JointState,
        dt: float,
    ) -> np.ndarray:
        pos_error = desired_state.positions - current_state.positions
        vel_error = desired_state.velocities - current_state.velocities

        desired_acc = desired_state.accelerations + self.kv * vel_error + self.kp * pos_error

        torques = self.dynamics.inverse_dynamics(current_state, desired_acc)
        return torques


class ImpedanceController(Controller):
    def __init__(
        self,
        stiffness: np.ndarray,
        damping: np.ndarray,
        inertia: Optional[np.ndarray] = None,
    ):
        self.stiffness = stiffness
        self.damping = damping
        self.inertia = inertia if inertia is not None else np.ones_like(stiffness)

    def compute(
        self,
        current_state: JointState,
        desired_state: JointState,
        dt: float,
    ) -> np.ndarray:
        pos_error = desired_state.positions - current_state.positions
        vel_error = desired_state.velocities - current_state.velocities

        tau = self.stiffness * pos_error + self.damping * vel_error

        if hasattr(current_state, 'force') and current_state.force is not None:
            tau += current_state.force

        return tau


class AdmittanceController(Controller):
    def __init__(
        self,
        mass: np.ndarray,
        damping: np.ndarray,
        stiffness: np.ndarray,
    ):
        self.mass = mass
        self.damping = damping
        self.stiffness = stiffness
        self.position = np.zeros_like(mass)
        self.velocity = np.zeros_like(mass)

    def compute(
        self,
        current_state: JointState,
        desired_state: JointState,
        dt: float,
    ) -> np.ndarray:
        force_error = desired_state.force - current_state.force if hasattr(current_state, 'force') else np.zeros_like(self.mass)

        acc = (force_error - self.damping * self.velocity - self.stiffness * (self.position - desired_state.positions)) / self.mass
        self.velocity += acc * dt
        self.position += self.velocity * dt

        return self.position
