from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Union
from enum import Enum
from abc import ABC, abstractmethod
import math
from collections import defaultdict, deque
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize, least_squares
from scipy.linalg import pinv, svd, qr, cholesky

class JointType(Enum):
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    FIXED = "fixed"
    CONTINUOUS = "continuous"
    PLANAR = "planar"
    SPHERICAL = "spherical"

class ControlMode(Enum):
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    IMPEDANCE = "impedance"
    ADMITTANCE = "admittance"
    HYBRID = "hybrid"

class GraspType(Enum):
    PINCH = "pinch"
    POWER = "power"
    PRECISION = "precision"
    LATERAL = "lateral"
    TRIPOD = "tripod"
    QUADPOD = "quadpod"
    WRAP = "wrap"
    HOOK = "hook"

@dataclass(frozen=True, slots=True)
class JointState:
    name: str
    position: float
    velocity: float
    effort: float
    joint_type: JointType = JointType.REVOLUTE
    limits: Tuple[float, float] = (-math.pi, math.pi)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class LinkState:
    name: str
    pose: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

@dataclass(frozen=True, slots=True)
class EndEffectorState:
    pose: np.ndarray
    twist: np.ndarray
    wrench: np.ndarray
    contact_points: List[np.ndarray] = field(default_factory=list)
    contact_forces: List[np.ndarray] = field(default_factory=list)

@dataclass(frozen=True, slots=True)
class Grasp:
    grasp_type: GraspType
    pregrasp_pose: np.ndarray
    grasp_pose: np.ndarray
    finger_positions: List[np.ndarray]
    contact_points: List[np.ndarray]
    contact_normals: List[np.ndarray]
    force_closure: bool
    quality: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    time: float
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    effort: Optional[np.ndarray] = None

@dataclass(frozen=True, slots=True)
class ManipulationTask:
    task_id: str
    target_pose: np.ndarray
    object_pose: Optional[np.ndarray] = None
    grasp: Optional[Grasp] = None
    constraints: List[Callable] = field(default_factory=list)
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class RobotModel:
    def __init__(self, name: str, urdf_path: Optional[str] = None):
        self.name = name
        self.joints: List[JointState] = []
        self.links: Dict[str, LinkState] = {}
        self.joint_limits: Dict[str, Tuple[float, float]] = {}
        self.joint_types: Dict[str, JointType] = {}
        self.parent_map: Dict[str, str] = {}
        self.child_map: Dict[str, List[str]] = defaultdict(list)
        self.link_frames: Dict[str, np.ndarray] = {}
        self.joint_axes: Dict[str, np.ndarray] = {}
        self.joint_origins: Dict[str, np.ndarray] = {}
        self.mass: Dict[str, float] = {}
        self.inertia: Dict[str, np.ndarray] = {}
        self.com: Dict[str, np.ndarray] = {}
        self.num_joints = 0
        self.num_links = 0
        self.base_link = ""
        self.end_effector_link = ""
        self.dof = 0

    def add_joint(self, name: str, joint_type: JointType, parent: str, child: str,
                  origin: np.ndarray, axis: np.ndarray, limits: Tuple[float, float] = (-math.pi, math.pi)):
        self.joints.append(JointState(name=name, position=0.0, velocity=0.0, effort=0.0,
                                      joint_type=joint_type, limits=limits))
        self.joint_limits[name] = limits
        self.joint_types[name] = joint_type
        self.parent_map[child] = parent
        self.child_map[parent].append(child)
        self.joint_origins[name] = origin
        self.joint_axes[name] = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 0 else np.array([0, 0, 1])
        self.num_joints += 1
        if joint_type != JointType.FIXED:
            self.dof += 1

    def add_link(self, name: str, mass: float = 1.0, inertia: Optional[np.ndarray] = None,
                 com: Optional[np.ndarray] = None):
        self.links[name] = LinkState(name=name, pose=np.eye(4),
                                     velocity=np.zeros(6), acceleration=np.zeros(6))
        self.mass[name] = mass
        self.inertia[name] = inertia if inertia is not None else np.eye(3) * 0.01
        self.com[name] = com if com is not None else np.zeros(3)
        self.num_links += 1
        if not self.base_link:
            self.base_link = name

    def set_base_link(self, name: str):
        self.base_link = name

    def set_end_effector(self, name: str):
        self.end_effector_link = name

    def get_joint_names(self) -> List[str]:
        return [j.name for j in self.joints if j.joint_type != JointType.FIXED]

    def get_joint_indices(self, names: List[str]) -> List[int]:
        joint_names = self.get_joint_names()
        return [joint_names.index(n) for n in names if n in joint_names]

class ForwardKinematics:
    def __init__(self, robot: RobotModel):
        self.robot = robot

    def compute(self, joint_positions: np.ndarray, link_name: Optional[str] = None) -> np.ndarray:
        link = link_name or self.robot.end_effector_link
        pose = np.eye(4)
        current = link

        joint_pos_dict = dict(zip(self.robot.get_joint_names(), joint_positions))

        while current != self.robot.base_link and current in self.robot.parent_map:
            parent = self.robot.parent_map[current]
            joint_name = self._find_joint(parent, current)

            if joint_name and joint_name in joint_pos_dict:
                q = joint_pos_dict[joint_name]
                joint_type = self.robot.joint_types.get(joint_name, JointType.REVOLUTE)
                origin = self.robot.joint_origins.get(joint_name, np.eye(4))
                axis = self.robot.joint_axes.get(joint_name, np.array([0, 0, 1]))

                transform = self._joint_transform(joint_type, q, origin, axis)
                pose = transform @ pose

            current = parent

        return pose

    def _find_joint(self, parent: str, child: str) -> Optional[str]:
        for joint in self.robot.joints:
            if joint.name in self.robot.parent_map:
                if self.robot.parent_map.get(joint.name) == parent and \
                   any(c == child for c in self.robot.child_map.get(parent, [])):
                    return joint.name
        return None

    def _joint_transform(self, joint_type: JointType, q: float,
                         origin: np.ndarray, axis: np.ndarray) -> np.ndarray:
        T = origin.copy()

        if joint_type == JointType.REVOLUTE:
            rot = R.from_rotvec(axis * q).as_matrix()
            T[:3, :3] = rot @ T[:3, :3]
        elif joint_type == JointType.PRISMATIC:
            T[:3, 3] += axis * q
        elif joint_type == JointType.CONTINUOUS:
            rot = R.from_rotvec(axis * q).as_matrix()
            T[:3, :3] = rot @ T[:3, :3]

        return T

    def compute_all_links(self, joint_positions: np.ndarray) -> Dict[str, np.ndarray]:
        poses = {}
        joint_pos_dict = dict(zip(self.robot.get_joint_names(), joint_positions))

        def traverse(link: str, parent_pose: np.ndarray):
            pose = parent_pose.copy()
            if link != self.robot.base_link and link in self.robot.parent_map:
                parent = self.robot.parent_map[link]
                joint_name = self._find_joint(parent, link)
                if joint_name and joint_name in joint_pos_dict:
                    q = joint_pos_dict[joint_name]
                    joint_type = self.robot.joint_types.get(joint_name, JointType.REVOLUTE)
                    origin = self.robot.joint_origins.get(joint_name, np.eye(4))
                    axis = self.robot.joint_axes.get(joint_name, np.array([0, 0, 1]))
                    transform = self._joint_transform(joint_type, q, origin, axis)
                    pose = transform @ pose

            poses[link] = pose
            for child in self.robot.child_map.get(link, []):
                traverse(child, pose)

        traverse(self.robot.base_link, np.eye(4))
        return poses

class InverseKinematics:
    def __init__(self, robot: RobotModel, fk: ForwardKinematics):
        self.robot = robot
        self.fk = fk
        self.max_iterations = 100
        self.tolerance = 1e-6
        self.damping = 0.01
        self.joint_limits_weight = 0.1
        self.nullspace_weight = 0.01

    def solve(self, target_pose: np.ndarray, initial_guess: Optional[np.ndarray] = None,
              method: str = "damped_least_squares") -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        n = self.robot.dof

        if initial_guess is None:
            initial_guess = np.zeros(n)

        q = initial_guess.copy()

        if method == "damped_least_squares":
            return self._damped_least_squares(target_pose, q)
        elif method == "levenberg_marquardt":
            return self._levenberg_marquardt(target_pose, q)
        elif method == "newton_raphson":
            return self._newton_raphson(target_pose, q)
        elif method == "gradient_descent":
            return self._gradient_descent(target_pose, q)
        else:
            return self._damped_least_squares(target_pose, q)

    def _damped_least_squares(self, target: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, bool, Dict]:
        info = {"iterations": 0, "error": float('inf'), "converged": False}

        for i in range(self.max_iterations):
            current_pose = self.fk.compute(q)
            error = self._pose_error(current_pose, target)
            info["error"] = np.linalg.norm(error)

            if info["error"] < self.tolerance:
                info["converged"] = True
                info["iterations"] = i
                return q, True, info

            J = self._compute_jacobian(q)
            J_damped = J.T @ pinv(J @ J.T + self.damping**2 * np.eye(6))
            dq = J_damped @ error

            q += dq
            q = self._enforce_limits(q)
            info["iterations"] = i + 1

        return q, False, info

    def _levenberg_marquardt(self, target: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, bool, Dict]:
        info = {"iterations": 0, "error": float('inf'), "converged": False}
        lambda_lm = 0.01

        for i in range(self.max_iterations):
            current_pose = self.fk.compute(q)
            error = self._pose_error(current_pose, target)
            info["error"] = np.linalg.norm(error)

            if info["error"] < self.tolerance:
                info["converged"] = True
                info["iterations"] = i
                return q, True, info

            J = self._compute_jacobian(q)
            H = J.T @ J + lambda_lm * np.eye(len(q))
            g = J.T @ error

            try:
                dq = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                dq = -pinv(H) @ g

            q_new = q + dq
            q_new = self._enforce_limits(q_new)

            new_pose = self.fk.compute(q_new)
            new_error = np.linalg.norm(self._pose_error(new_pose, target))

            if new_error < info["error"]:
                q = q_new
                lambda_lm *= 0.8
            else:
                lambda_lm *= 1.5

            info["iterations"] = i + 1

        return q, False, info

    def _newton_raphson(self, target: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, bool, Dict]:
        info = {"iterations": 0, "error": float('inf'), "converged": False}

        for i in range(self.max_iterations):
            current_pose = self.fk.compute(q)
            error = self._pose_error(current_pose, target)
            info["error"] = np.linalg.norm(error)

            if info["error"] < self.tolerance:
                info["converged"] = True
                info["iterations"] = i
                return q, True, info

            J = self._compute_jacobian(q)
            try:
                dq = np.linalg.solve(J.T @ J, J.T @ error)
            except np.linalg.LinAlgError:
                dq = pinv(J) @ error

            q += dq
            q = self._enforce_limits(q)
            info["iterations"] = i + 1

        return q, False, info

    def _gradient_descent(self, target: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, bool, Dict]:
        info = {"iterations": 0, "error": float('inf'), "converged": False}
        alpha = 0.1

        for i in range(self.max_iterations):
            current_pose = self.fk.compute(q)
            error = self._pose_error(current_pose, target)
            info["error"] = np.linalg.norm(error)

            if info["error"] < self.tolerance:
                info["converged"] = True
                info["iterations"] = i
                return q, True, info

            J = self._compute_jacobian(q)
            grad = J.T @ error
            q -= alpha * grad
            q = self._enforce_limits(q)
            info["iterations"] = i + 1

        return q, False, info

    def _compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        n = len(q)
        J = np.zeros((6, n))
        eps = 1e-6

        for i in range(n):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[i] += eps
            q_minus[i] -= eps

            pose_plus = self.fk.compute(q_plus)
            pose_minus = self.fk.compute(q_minus)

            J[:3, i] = (pose_plus[:3, 3] - pose_minus[:3, 3]) / (2 * eps)
            rot_plus = R.from_matrix(pose_plus[:3, :3]).as_rotvec()
            rot_minus = R.from_matrix(pose_minus[:3, :3]).as_rotvec()
            J[3:, i] = (rot_plus - rot_minus) / (2 * eps)

        return J

    def _pose_error(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        pos_error = target[:3, 3] - current[:3, 3]
        rot_current = R.from_matrix(current[:3, :3])
        rot_target = R.from_matrix(target[:3, :3])
        rot_error = (rot_target * rot_current.inv()).as_rotvec()
        return np.concatenate([pos_error, rot_error])

    def _enforce_limits(self, q: np.ndarray) -> np.ndarray:
        joint_names = self.robot.get_joint_names()
        for i, name in enumerate(joint_names):
            if name in self.robot.joint_limits:
                lower, upper = self.robot.joint_limits[name]
                q[i] = np.clip(q[i], lower, upper)
        return q

class TrajectoryPlanner:
    def __init__(self, robot: RobotModel):
        self.robot = robot
        self.fk = ForwardKinematics(robot)
        self.ik = InverseKinematics(robot, self.fk)

    def plan_joint_space(self, start: np.ndarray, goal: np.ndarray,
                         duration: float, dt: float = 0.01,
                         method: str = "cubic") -> List[TrajectoryPoint]:
        n = len(start)
        num_steps = int(duration / dt) + 1
        times = np.linspace(0, duration, num_steps)
        trajectory = []

        if method == "cubic":
            coeffs = self._cubic_coeffs(start, goal, duration)
        elif method == "quintic":
            coeffs = self._quintic_coeffs(start, goal, duration)
        elif method == "trapezoidal":
            coeffs = self._trapezoidal_coeffs(start, goal, duration)
        else:
            coeffs = self._cubic_coeffs(start, goal, duration)

        for t in times:
            if method in ["cubic", "quintic"]:
                pos = self._eval_poly(coeffs[0], t)
                vel = self._eval_poly(coeffs[1], t)
                acc = self._eval_poly(coeffs[2], t)
            else:
                pos, vel, acc = self._eval_trapezoidal(coeffs, t)

            trajectory.append(TrajectoryPoint(
                time=t, positions=pos, velocities=vel, accelerations=acc
            ))

        return trajectory

    def plan_cartesian_space(self, start_pose: np.ndarray, goal_pose: np.ndarray,
                             duration: float, dt: float = 0.01,
                             method: str = "linear") -> List[TrajectoryPoint]:
        num_steps = int(duration / dt) + 1
        times = np.linspace(0, duration, num_steps)
        trajectory = []

        start_pos = start_pose[:3, 3]
        goal_pos = goal_pose[:3, 3]
        start_rot = R.from_matrix(start_pose[:3, :3])
        goal_rot = R.from_matrix(goal_pose[:3, :3])

        for i, t in enumerate(times):
            alpha = t / duration

            if method == "linear":
                pos = start_pos + alpha * (goal_pos - start_pos)
                rot = start_rot.slerp(goal_rot, alpha)
            elif method == "slerp":
                pos = start_pos + alpha * (goal_pos - start_pos)
                rot = start_rot.slerp(goal_rot, alpha)
            elif method == "screw":
                pos, rot = self._screw_interpolation(start_pose, goal_pose, alpha)
            else:
                pos = start_pos + alpha * (goal_pos - start_pos)
                rot = start_rot.slerp(goal_rot, alpha)

            pose = np.eye(4)
            pose[:3, :3] = rot.as_matrix()
            pose[:3, 3] = pos

            q, success, _ = self.ik.solve(pose)
            if not success:
                q = np.zeros(self.robot.dof)

            if i > 0:
                vel = (q - trajectory[-1].positions) / dt
                acc = (vel - trajectory[-1].velocities) / dt
            else:
                vel = np.zeros(self.robot.dof)
                acc = np.zeros(self.robot.dof)

            trajectory.append(TrajectoryPoint(
                time=t, positions=q, velocities=vel, accelerations=acc
            ))

        return trajectory

    def plan_through_waypoints(self, waypoints: List[np.ndarray],
                               durations: List[float], dt: float = 0.01) -> List[TrajectoryPoint]:
        full_trajectory = []
        time_offset = 0.0

        for i in range(len(waypoints) - 1):
            traj = self.plan_cartesian_space(waypoints[i], waypoints[i + 1],
                                             durations[i], dt)
            for pt in traj:
                pt.time += time_offset
                full_trajectory.append(pt)
            time_offset += durations[i]

        return full_trajectory

    def _cubic_coeffs(self, start: np.ndarray, goal: np.ndarray,
                      duration: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        a0 = start
        a1 = np.zeros_like(start)
        a2 = 3 * (goal - start) / duration**2
        a3 = -2 * (goal - start) / duration**3
        return np.array([a0, a1, a2, a3]), np.array([a1, 2*a2, 3*a3]), np.array([2*a2, 6*a3])

    def _quintic_coeffs(self, start: np.ndarray, goal: np.ndarray,
                        duration: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        a0 = start
        a1 = np.zeros_like(start)
        a2 = np.zeros_like(start)
        a3 = 10 * (goal - start) / duration**3
        a4 = -15 * (goal - start) / duration**4
        a5 = 6 * (goal - start) / duration**5
        return np.array([a0, a1, a2, a3, a4, a5]), \
               np.array([a1, 2*a2, 3*a3, 4*a4, 5*a5]), \
               np.array([2*a2, 6*a3, 12*a4, 20*a5])

    def _trapezoidal_coeffs(self, start: np.ndarray, goal: np.ndarray,
                            duration: float) -> Dict[str, Any]:
        return {"start": start, "goal": goal, "duration": duration,
                "accel_time": duration / 3, "max_vel": 1.5 * (goal - start) / duration}

    def _eval_poly(self, coeffs: np.ndarray, t: float) -> np.ndarray:
        result = np.zeros_like(coeffs[0])
        for i, c in enumerate(coeffs):
            result += c * (t ** i)
        return result

    def _eval_trapezoidal(self, coeffs: Dict, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        start = coeffs["start"]
        goal = coeffs["goal"]
        duration = coeffs["duration"]
        ta = coeffs["accel_time"]
        vmax = coeffs["max_vel"]

        if t < ta:
            pos = start + 0.5 * vmax / ta * t**2
            vel = vmax / ta * t
            acc = vmax / ta
        elif t < duration - ta:
            pos = start + vmax * (t - 0.5 * ta)
            vel = vmax
            acc = np.zeros_like(vmax)
        else:
            td = duration - t
            pos = goal - 0.5 * vmax / ta * td**2
            vel = vmax / ta * td
            acc = -vmax / ta

        return pos, vel, acc

    def _screw_interpolation(self, start: np.ndarray, goal: np.ndarray,
                             alpha: float) -> Tuple[np.ndarray, R]:
        rel = np.linalg.inv(start) @ goal
        screw = self._pose_to_screw(rel)
        interp = screw * alpha
        interp_pose = self._screw_to_pose(interp)
        result = start @ interp_pose
        return result[:3, 3], R.from_matrix(result[:3, :3])

    def _pose_to_screw(self, pose: np.ndarray) -> np.ndarray:
        rot = R.from_matrix(pose[:3, :3])
        rotvec = rot.as_rotvec()
        theta = np.linalg.norm(rotvec)

        if theta < 1e-6:
            return np.concatenate([np.zeros(3), pose[:3, 3]])

        axis = rotvec / theta
        moment = np.cross(pose[:3, 3], axis) + axis * np.dot(pose[:3, 3], axis) * (1 - np.cos(theta)) / theta
        return np.concatenate([moment, axis]) * theta

    def _screw_to_pose(self, screw: np.ndarray) -> np.ndarray:
        v = screw[:3]
        omega = screw[3:]
        theta = np.linalg.norm(omega)

        if theta < 1e-6:
            pose = np.eye(4)
            pose[:3, 3] = v
            return pose

        axis = omega / theta
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R_mat = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K
        G = np.eye(3) * theta + (1 - math.cos(theta)) * K + (theta - math.sin(theta)) * K @ K
        p = G @ v

        pose = np.eye(4)
        pose[:3, :3] = R_mat
        pose[:3, 3] = p
        return pose

class Dynamics:
    def __init__(self, robot: RobotModel):
        self.robot = robot
        self.gravity = np.array([0, 0, -9.81])

    def compute_mass_matrix(self, q: np.ndarray) -> np.ndarray:
        n = self.robot.dof
        M = np.zeros((n, n))

        link_poses = self._compute_all_link_poses(q)
        link_velocities = self._compute_all_link_velocities(q, np.zeros(n))

        for i, joint_name in enumerate(self.robot.get_joint_names()):
            for j, joint_name2 in enumerate(self.robot.get_joint_names()):
                if i > j:
                    continue
                val = 0.0
                for link_name, pose in link_poses.items():
                    if link_name not in self.robot.mass:
                        continue
                    m = self.robot.mass[link_name]
                    I = self.robot.inertia[link_name]
                    com = self.robot.com[link_name]

                    Jv_i = self._compute_linear_jacobian(q, link_name, i)
                    Jv_j = self._compute_linear_jacobian(q, link_name, j)
                    Jw_i = self._compute_angular_jacobian(q, link_name, i)
                    Jw_j = self._compute_angular_jacobian(q, link_name, j)

                    val += m * Jv_i @ Jv_j + Jw_i @ I @ Jw_j.T

                M[i, j] = val
                M[j, i] = val

        return M + 1e-6 * np.eye(n)

    def compute_coriolis(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        n = self.robot.dof
        C = np.zeros(n)
        M = self.compute_mass_matrix(q)

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    dM_ij_dqk = self._mass_matrix_derivative(q, i, j, k)
                    dM_ik_dqj = self._mass_matrix_derivative(q, i, k, j)
                    dM_jk_dqi = self._mass_matrix_derivative(q, j, k, i)
                    christoffel = 0.5 * (dM_ij_dqk + dM_ik_dqj - dM_jk_dqi)
                    C[i] += christoffel * qdot[j] * qdot[k]

        return C

    def compute_gravity(self, q: np.ndarray) -> np.ndarray:
        n = self.robot.dof
        G = np.zeros(n)
        link_poses = self._compute_all_link_poses(q)

        for i, joint_name in enumerate(self.robot.get_joint_names()):
            for link_name, pose in link_poses.items():
                if link_name not in self.robot.mass:
                    continue
                m = self.robot.mass[link_name]
                com_world = pose @ np.append(self.robot.com[link_name], 1)
                Jv = self._compute_linear_jacobian(q, link_name, i)
                G[i] += m * Jv @ self.gravity

        return G

    def forward_dynamics(self, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray) -> np.ndarray:
        M = self.compute_mass_matrix(q)
        C = self.compute_coriolis(q, qdot)
        G = self.compute_gravity(q)
        qddot = pinv(M) @ (tau - C - G)
        return qddot

    def inverse_dynamics(self, q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray) -> np.ndarray:
        M = self.compute_mass_matrix(q)
        C = self.compute_coriolis(q, qdot)
        G = self.compute_gravity(q)
        tau = M @ qddot + C + G
        return tau

    def _compute_all_link_poses(self, q: np.ndarray) -> Dict[str, np.ndarray]:
        return self.fk.compute_all_links(q)

    def _compute_all_link_velocities(self, q: np.ndarray, qdot: np.ndarray) -> Dict[str, np.ndarray]:
        J = self._compute_full_jacobian(q)
        v = J @ qdot
        velocities = {}
        idx = 0
        for link_name in self.robot.links:
            velocities[link_name] = v[idx*6:(idx+1)*6]
            idx += 1
        return velocities

    def _compute_full_jacobian(self, q: np.ndarray) -> np.ndarray:
        n = self.robot.dof
        num_links = len(self.robot.links)
        J = np.zeros((6 * num_links, n))

        for i, link_name in enumerate(self.robot.links):
            for j, joint_name in enumerate(self.robot.get_joint_names()):
                J[i*6:i*6+3, j] = self._compute_linear_jacobian(q, link_name, j)
                J[i*6+3:i*6+6, j] = self._compute_angular_jacobian(q, link_name, j)

        return J

    def _compute_linear_jacobian(self, q: np.ndarray, link_name: str, joint_idx: int) -> np.ndarray:
        joint_names = self.robot.get_joint_names()
        if joint_idx >= len(joint_names):
            return np.zeros(3)

        joint_name = joint_names[joint_idx]
        joint_type = self.robot.joint_types.get(joint_name, JointType.REVOLUTE)

        link_poses = self._compute_all_link_poses(q)
        if link_name not in link_poses:
            return np.zeros(3)

        joint_pose = self._compute_joint_pose(q, joint_name)
        link_pose = link_poses[link_name]

        axis = joint_pose[:3, :3] @ self.robot.joint_axes.get(joint_name, np.array([0, 0, 1]))
        r = link_pose[:3, 3] - joint_pose[:3, 3]

        if joint_type == JointType.REVOLUTE or joint_type == JointType.CONTINUOUS:
            return np.cross(axis, r)
        elif joint_type == JointType.PRISMATIC:
            return axis
        return np.zeros(3)

    def _compute_angular_jacobian(self, q: np.ndarray, link_name: str, joint_idx: int) -> np.ndarray:
        joint_names = self.robot.get_joint_names()
        if joint_idx >= len(joint_names):
            return np.zeros(3)

        joint_name = joint_names[joint_idx]
        joint_type = self.robot.joint_types.get(joint_name, JointType.REVOLUTE)

        joint_pose = self._compute_joint_pose(q, joint_name)
        axis = joint_pose[:3, :3] @ self.robot.joint_axes.get(joint_name, np.array([0, 0, 1]))

        if joint_type == JointType.REVOLUTE or joint_type == JointType.CONTINUOUS:
            return axis
        return np.zeros(3)

    def _compute_joint_pose(self, q: np.ndarray, joint_name: str) -> np.ndarray:
        pose = np.eye(4)
        current = joint_name

        joint_pos_dict = dict(zip(self.robot.get_joint_names(), q))

        while current != self.robot.base_link and current in self.robot.parent_map:
            parent = self.robot.parent_map[current]
            j_name = self._find_joint(parent, current)

            if j_name and j_name in joint_pos_dict:
                q_val = joint_pos_dict[j_name]
                j_type = self.robot.joint_types.get(j_name, JointType.REVOLUTE)
                origin = self.robot.joint_origins.get(j_name, np.eye(4))
                axis = self.robot.joint_axes.get(j_name, np.array([0, 0, 1]))
                transform = self._joint_transform(j_type, q_val, origin, axis)
                pose = transform @ pose

            current = parent

        return pose

    def _find_joint(self, parent: str, child: str) -> Optional[str]:
        for joint in self.robot.joints:
            if joint.name in self.robot.parent_map:
                if self.robot.parent_map.get(joint.name) == parent:
                    return joint.name
        return None

    def _joint_transform(self, joint_type: JointType, q: float,
                         origin: np.ndarray, axis: np.ndarray) -> np.ndarray:
        T = origin.copy()
        if joint_type == JointType.REVOLUTE:
            rot = R.from_rotvec(axis * q).as_matrix()
            T[:3, :3] = rot @ T[:3, :3]
        elif joint_type == JointType.PRISMATIC:
            T[:3, 3] += axis * q
        elif joint_type == JointType.CONTINUOUS:
            rot = R.from_rotvec(axis * q).as_matrix()
            T[:3, :3] = rot @ T[:3, :3]
        return T

    def _mass_matrix_derivative(self, q: np.ndarray, i: int, j: int, k: int) -> float:
        eps = 1e-6
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[k] += eps
        q_minus[k] -= eps

        M_plus = self.compute_mass_matrix(q_plus)
        M_minus = self.compute_mass_matrix(q_minus)

        return (M_plus[i, j] - M_minus[i, j]) / (2 * eps)

class ImpedanceController:
    def __init__(self, robot: RobotModel, Kp: np.ndarray = None, Kd: np.ndarray = None):
        self.robot = robot
        self.dynamics = Dynamics(robot)
        self.Kp = Kp if Kp is not None else np.eye(6) * 100
        self.Kd = Kd if Kd is not None else np.eye(6) * 20
        self.desired_pose = np.eye(4)
        self.desired_twist = np.zeros(6)

    def compute_torque(self, q: np.ndarray, qdot: np.ndarray,
                       desired_pose: np.ndarray = None,
                       desired_twist: np.ndarray = None,
                       desired_wrench: np.ndarray = None) -> np.ndarray:
        if desired_pose is not None:
            self.desired_pose = desired_pose
        if desired_twist is not None:
            self.desired_twist = desired_twist

        fk = ForwardKinematics(self.robot)
        current_pose = fk.compute(q)
        J = self.dynamics._compute_full_jacobian(q)[:6, :]

        pose_error = self._pose_error(current_pose, self.desired_pose)
        twist = J @ qdot
        twist_error = self.desired_twist - twist

        F = self.Kp @ pose_error + self.Kd @ twist_error

        if desired_wrench is not None:
            F += desired_wrench

        tau = J.T @ F
        tau += self.dynamics.compute_gravity(q)

        return tau

    def _pose_error(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        pos_error = target[:3, 3] - current[:3, 3]
        rot_current = R.from_matrix(current[:3, :3])
        rot_target = R.from_matrix(target[:3, :3])
        rot_error = (rot_target * rot_current.inv()).as_rotvec()
        return np.concatenate([pos_error, rot_error])

class AdmittanceController:
    def __init__(self, robot: RobotModel, M: np.ndarray = None, D: np.ndarray = None, K: np.ndarray = None):
        self.robot = robot
        self.M = M if M is not None else np.eye(6) * 1.0
        self.D = D if D is not None else np.eye(6) * 10.0
        self.K = K if K is not None else np.eye(6) * 100.0
        self.pose = np.eye(4)
        self.twist = np.zeros(6)
        self.dt = 0.001

    def update(self, measured_wrench: np.ndarray, dt: float = None) -> np.ndarray:
        if dt is not None:
            self.dt = dt

        acc = pinv(self.M) @ (measured_wrench - self.D @ self.twist - self.K @ self._pose_error(self.pose, np.eye(4)))
        self.twist += acc * self.dt

        pos = self.pose[:3, 3] + self.twist[:3] * self.dt
        rot_vec = self.twist[3:] * self.dt
        rot = R.from_rotvec(rot_vec).as_matrix()
        self.pose[:3, :3] = rot @ self.pose[:3, :3]
        self.pose[:3, 3] = pos

        fk = ForwardKinematics(self.robot)
        ik = InverseKinematics(self.robot, fk)
        q, success, _ = ik.solve(self.pose)

        return q if success else np.zeros(self.robot.dof)

    def _pose_error(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        pos_error = target[:3, 3] - current[:3, 3]
        rot_current = R.from_matrix(current[:3, :3])
        rot_target = R.from_matrix(target[:3, :3])
        rot_error = (rot_target * rot_current.inv()).as_rotvec()
        return np.concatenate([pos_error, rot_error])

    def reset(self, pose: np.ndarray = None):
        self.pose = pose if pose is not None else np.eye(4)
        self.twist = np.zeros(6)

class GraspPlanner:
    def __init__(self, robot: RobotModel):
        self.robot = robot
        self.fk = ForwardKinematics(robot)
        self.ik = InverseKinematics(robot, self.fk)

    def plan_grasp(self, object_pose: np.ndarray, object_geometry: Dict[str, Any],
                   approach_direction: np.ndarray = None,
                   grasp_type: GraspType = GraspType.PINCH) -> List[Grasp]:
        grasps = []

        if approach_direction is None:
            approach_direction = -object_pose[:3, 2]

        if grasp_type == GraspType.PINCH:
            grasps.extend(self._plan_pinch_grasp(object_pose, object_geometry, approach_direction))
        elif grasp_type == GraspType.POWER:
            grasps.extend(self._plan_power_grasp(object_pose, object_geometry, approach_direction))
        elif grasp_type == GraspType.PRECISION:
            grasps.extend(self._plan_precision_grasp(object_pose, object_geometry, approach_direction))
        elif grasp_type == GraspType.TRIPOD:
            grasps.extend(self._plan_tripod_grasp(object_pose, object_geometry, approach_direction))
        elif grasp_type == GraspType.WRAP:
            grasps.extend(self._plan_wrap_grasp(object_pose, object_geometry, approach_direction))

        return grasps

    def _plan_pinch_grasp(self, object_pose: np.ndarray, geometry: Dict,
                          approach: np.ndarray) -> List[Grasp]:
        grasps = []
        width = geometry.get('width', 0.05)
        depth = geometry.get('depth', 0.05)

        for angle in [0, math.pi/4, math.pi/2, 3*math.pi/4]:
            rot = R.from_rotvec(object_pose[:3, 2] * angle).as_matrix()
            grasp_pose = object_pose.copy()
            grasp_pose[:3, :3] = rot @ grasp_pose[:3, :3]
            grasp_pose[:3, 3] += rot @ approach * (depth + 0.02)

            finger1 = grasp_pose[:3, 3] + rot @ np.array([0, width/2, 0])
            finger2 = grasp_pose[:3, 3] - rot @ np.array([0, width/2, 0])

            pregrasp = grasp_pose.copy()
            pregrasp[:3, 3] += rot @ approach * 0.1

            q_pregrasp, success1, _ = self.ik.solve(pregrasp)
            q_grasp, success2, _ = self.ik.solve(grasp_pose)

            if success1 and success2:
                quality = self._compute_grasp_quality([finger1, finger2],
                                                     [rot @ np.array([0, -1, 0]),
                                                      rot @ np.array([0, 1, 0])])
                grasps.append(Grasp(
                    grasp_type=GraspType.PINCH,
                    pregrasp_pose=pregrasp,
                    grasp_pose=grasp_pose,
                    finger_positions=[finger1, finger2],
                    contact_points=[finger1, finger2],
                    contact_normals=[rot @ np.array([0, -1, 0]), rot @ np.array([0, 1, 0])],
                    force_closure=True,
                    quality=quality
                ))

        return grasps

    def _plan_power_grasp(self, object_pose: np.ndarray, geometry: Dict,
                          approach: np.ndarray) -> List[Grasp]:
        grasps = []
        radius = geometry.get('radius', 0.03)
        height = geometry.get('height', 0.1)

        for angle in np.linspace(0, 2*math.pi, 8):
            rot = R.from_rotvec(object_pose[:3, 2] * angle).as_matrix()
            grasp_pose = object_pose.copy()
            grasp_pose[:3, :3] = rot @ grasp_pose[:3, :3]
            grasp_pose[:3, 3] += rot @ approach * 0.05

            fingers = []
            normals = []
            for i in range(4):
                finger_angle = angle + i * math.pi/2
                finger_pos = grasp_pose[:3, 3] + rot @ np.array([
                    radius * math.cos(finger_angle),
                    radius * math.sin(finger_angle),
                    -height/2
                ])
                fingers.append(finger_pos)
                normals.append(-rot @ np.array([
                    math.cos(finger_angle),
                    math.sin(finger_angle),
                    0
                ]))

            pregrasp = grasp_pose.copy()
            pregrasp[:3, 3] += rot @ approach * 0.1

            q_pregrasp, success1, _ = self.ik.solve(pregrasp)
            q_grasp, success2, _ = self.ik.solve(grasp_pose)

            if success1 and success2:
                quality = self._compute_grasp_quality(fingers, normals)
                grasps.append(Grasp(
                    grasp_type=GraspType.POWER,
                    pregrasp_pose=pregrasp,
                    grasp_pose=grasp_pose,
                    finger_positions=fingers,
                    contact_points=fingers,
                    contact_normals=normals,
                    force_closure=True,
                    quality=quality
                ))

        return grasps

    def _plan_precision_grasp(self, object_pose: np.ndarray, geometry: Dict,
                              approach: np.ndarray) -> List[Grasp]:
        return self._plan_pinch_grasp(object_pose, geometry, approach)

    def _plan_tripod_grasp(self, object_pose: np.ndarray, geometry: Dict,
                           approach: np.ndarray) -> List[Grasp]:
        grasps = []
        radius = geometry.get('radius', 0.03)

        for angle in np.linspace(0, 2*math.pi, 6):
            rot = R.from_rotvec(object_pose[:3, 2] * angle).as_matrix()
            grasp_pose = object_pose.copy()
            grasp_pose[:3, :3] = rot @ grasp_pose[:3, :3]

            fingers = []
            normals = []
            for i in range(3):
                finger_angle = angle + i * 2*math.pi/3
                finger_pos = grasp_pose[:3, 3] + rot @ np.array([
                    radius * math.cos(finger_angle),
                    radius * math.sin(finger_angle),
                    0
                ])
                fingers.append(finger_pos)
                normals.append(-rot @ np.array([
                    math.cos(finger_angle),
                    math.sin(finger_angle),
                    0
                ]))

            pregrasp = grasp_pose.copy()
            pregrasp[:3, 3] += rot @ approach * 0.1

            q_pregrasp, success1, _ = self.ik.solve(pregrasp)
            q_grasp, success2, _ = self.ik.solve(grasp_pose)

            if success1 and success2:
                quality = self._compute_grasp_quality(fingers, normals)
                grasps.append(Grasp(
                    grasp_type=GraspType.TRIPOD,
                    pregrasp_pose=pregrasp,
                    grasp_pose=grasp_pose,
                    finger_positions=fingers,
                    contact_points=fingers,
                    contact_normals=normals,
                    force_closure=True,
                    quality=quality
                ))

        return grasps

    def _plan_wrap_grasp(self, object_pose: np.ndarray, geometry: Dict,
                         approach: np.ndarray) -> List[Grasp]:
        return self._plan_power_grasp(object_pose, geometry, approach)

    def _compute_grasp_quality(self, contact_points: List[np.ndarray],
                               contact_normals: List[np.ndarray]) -> float:
        if len(contact_points) < 2:
            return 0.0

        G = []
        for p, n in zip(contact_points, contact_normals):
            G.append(np.concatenate([n, np.cross(p, n)]))
        G = np.array(G).T

        if G.shape[1] < 6:
            return 0.0

        try:
            svd_vals = svd(G, compute_uv=False)
            return float(svd_vals[-1] / svd_vals[0]) if svd_vals[0] > 0 else 0.0
        except:
            return 0.0

    def evaluate_grasp(self, grasp: Grasp, external_wrench: np.ndarray = None) -> Dict[str, float]:
        G = []
        for p, n in zip(grasp.contact_points, grasp.contact_normals):
            G.append(np.concatenate([n, np.cross(p, n)]))
        G = np.array(G).T

        metrics = {}

        try:
            svd_vals = svd(G, compute_uv=False)
            metrics['min_singular_value'] = float(svd_vals[-1])
            metrics['condition_number'] = float(svd_vals[0] / svd_vals[-1]) if svd_vals[-1] > 0 else float('inf')
            metrics['grasp_isotropy'] = float(svd_vals[-1] / svd_vals[0]) if svd_vals[0] > 0 else 0.0
        except:
            metrics['min_singular_value'] = 0.0
            metrics['condition_number'] = float('inf')
            metrics['grasp_isotropy'] = 0.0

        if external_wrench is not None:
            try:
                forces = pinv(G) @ external_wrench
                metrics['max_contact_force'] = float(np.max(np.abs(forces)))
                metrics['total_force'] = float(np.sum(np.abs(forces)))
            except:
                metrics['max_contact_force'] = float('inf')
                metrics['total_force'] = float('inf')

        metrics['force_closure'] = grasp.force_closure
        metrics['quality'] = grasp.quality

        return metrics

class ManipulationPlanner:
    def __init__(self, robot: RobotModel):
        self.robot = robot
        self.fk = ForwardKinematics(robot)
        self.ik = InverseKinematics(robot, self.fk)
        self.traj_planner = TrajectoryPlanner(robot)
        self.grasp_planner = GraspPlanner(robot)

    def plan_pick_and_place(self, object_pose: np.ndarray, place_pose: np.ndarray,
                            object_geometry: Dict[str, Any],
                            grasp_type: GraspType = GraspType.PINCH) -> Dict[str, Any]:
        grasps = self.grasp_planner.plan_grasp(object_pose, object_geometry, grasp_type=grasp_type)

        if not grasps:
            return {"success": False, "reason": "No valid grasps found"}

        best_grasp = max(grasps, key=lambda g: g.quality)

        pregrasp_traj = self.traj_planner.plan_cartesian_space(
            self.fk.compute(np.zeros(self.robot.dof)), best_grasp.pregrasp_pose, 2.0)
        grasp_traj = self.traj_planner.plan_cartesian_space(
            best_grasp.pregrasp_pose, best_grasp.grasp_pose, 1.0)
        lift_traj = self.traj_planner.plan_cartesian_space(
            best_grasp.grasp_pose, self._lift_pose(best_grasp.grasp_pose), 1.0)
        transport_traj = self.traj_planner.plan_cartesian_space(
            self._lift_pose(best_grasp.grasp_pose), self._lift_pose(place_pose), 3.0)
        lower_traj = self.traj_planner.plan_cartesian_space(
            self._lift_pose(place_pose), place_pose, 1.0)
        retreat_traj = self.traj_planner.plan_cartesian_space(
            place_pose, self._retreat_pose(place_pose), 1.0)

        return {
            "success": True,
            "grasp": best_grasp,
            "trajectories": {
                "pregrasp": pregrasp_traj,
                "grasp": grasp_traj,
                "lift": lift_traj,
                "transport": transport_traj,
                "lower": lower_traj,
                "retreat": retreat_traj
            }
        }

    def _lift_pose(self, pose: np.ndarray) -> np.ndarray:
        lifted = pose.copy()
        lifted[:3, 3] += np.array([0, 0, 0.1])
        return lifted

    def _retreat_pose(self, pose: np.ndarray) -> np.ndarray:
        retreated = pose.copy()
        retreated[:3, 3] += pose[:3, 2] * 0.1
        return retreated

    def plan_dual_arm_task(self, left_task: ManipulationTask, right_task: ManipulationTask) -> Dict[str, Any]:
        left_traj = self.plan_single_arm_task(left_task)
        right_traj = self.plan_single_arm_task(right_task)

        return {
            "left": left_traj,
            "right": right_traj,
            "synchronized": self._synchronize_trajectories(left_traj, right_traj)
        }

    def plan_single_arm_task(self, task: ManipulationTask) -> List[TrajectoryPoint]:
        current_q = np.zeros(self.robot.dof)
        waypoints = []

        if task.grasp:
            waypoints.append(task.grasp.pregrasp_pose)
            waypoints.append(task.grasp.grasp_pose)
        if task.target_pose is not None:
            waypoints.append(task.target_pose)

        if not waypoints:
            return []

        durations = [2.0] * len(waypoints)
        return self.traj_planner.plan_through_waypoints(waypoints, durations)

    def _synchronize_trajectories(self, left: List[TrajectoryPoint],
                                  right: List[TrajectoryPoint]) -> List[Tuple[TrajectoryPoint, TrajectoryPoint]]:
        min_len = min(len(left), len(right))
        return list(zip(left[:min_len], right[:min_len]))

class CollisionChecker:
    def __init__(self, robot: RobotModel):
        self.robot = robot
        self.fk = ForwardKinematics(robot)
        self.collision_objects: List[Dict] = []
        self.self_collision_pairs: List[Tuple[str, str]] = []

    def add_collision_object(self, name: str, geometry: Dict[str, Any], pose: np.ndarray):
        self.collision_objects.append({
            "name": name,
            "geometry": geometry,
            "pose": pose
        })

    def add_self_collision_pair(self, link1: str, link2: str):
        self.self_collision_pairs.append((link1, link2))

    def check_collision(self, q: np.ndarray) -> Tuple[bool, List[str]]:
        link_poses = self.fk.compute_all_links(q)
        collisions = []

        for obj in self.collision_objects:
            for link_name, link_pose in link_poses.items():
                if self._check_link_object_collision(link_name, link_pose, obj):
                    collisions.append(f"{link_name} - {obj['name']}")

        for link1, link2 in self.self_collision_pairs:
            if link1 in link_poses and link2 in link_poses:
                if self._check_link_link_collision(link1, link_poses[link1], link2, link_poses[link2]):
                    collisions.append(f"{link1} - {link2}")

        return len(collisions) > 0, collisions

    def check_trajectory(self, trajectory: List[TrajectoryPoint]) -> Tuple[bool, List[str]]:
        for pt in trajectory:
            collision, collisions = self.check_collision(pt.positions)
            if collision:
                return True, collisions
        return False, []

    def _check_link_object_collision(self, link_name: str, link_pose: np.ndarray,
                                     obj: Dict) -> bool:
        geom = obj["geometry"]
        obj_pose = obj["pose"]

        link_radius = 0.05
        obj_type = geom.get("type", "sphere")

        if obj_type == "sphere":
            obj_radius = geom.get("radius", 0.1)
            dist = np.linalg.norm(link_pose[:3, 3] - obj_pose[:3, 3])
            return dist < (link_radius + obj_radius)
        elif obj_type == "box":
            size = geom.get("size", [0.2, 0.2, 0.2])
            rel_pos = np.linalg.inv(obj_pose) @ np.append(link_pose[:3, 3], 1)
            return all(abs(rel_pos[i]) < size[i]/2 + link_radius for i in range(3))

        return False

    def _check_link_link_collision(self, link1: str, pose1: np.ndarray,
                                   link2: str, pose2: np.ndarray) -> bool:
        dist = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])
        return dist < 0.02

class ForceController:
    def __init__(self, robot: RobotModel):
        self.robot = robot
        self.dynamics = Dynamics(robot)
        self.Kp = np.eye(6) * 10.0
        self.Ki = np.eye(6) * 1.0
        self.integral = np.zeros(6)

    def compute_torque(self, q: np.ndarray, qdot: np.ndarray,
                       desired_wrench: np.ndarray,
                       measured_wrench: np.ndarray) -> np.ndarray:
        error = desired_wrench - measured_wrench
        self.integral += error * 0.001

        F = self.Kp @ error + self.Ki @ self.integral

        fk = ForwardKinematics(self.robot)
        J = self.dynamics._compute_full_jacobian(q)[:6, :]

        tau = J.T @ F
        tau += self.dynamics.compute_gravity(q)

        return tau

    def reset(self):
        self.integral = np.zeros(6)

class HybridController:
    def __init__(self, robot: RobotModel):
        self.robot = robot
        self.impedance = ImpedanceController(robot)
        self.force = ForceController(robot)
        self.selection_matrix = np.eye(6)

    def set_selection_matrix(self, S: np.ndarray):
        self.selection_matrix = S

    def compute_torque(self, q: np.ndarray, qdot: np.ndarray,
                       desired_pose: np.ndarray,
                       desired_twist: np.ndarray,
                       desired_wrench: np.ndarray,
                       measured_wrench: np.ndarray) -> np.ndarray:
        S = self.selection_matrix

        tau_impedance = self.impedance.compute_torque(q, qdot, desired_pose, desired_twist)
        tau_force = self.force.compute_torque(q, qdot, desired_wrench, measured_wrench)

        J = self.impedance.dynamics._compute_full_jacobian(q)[:6, :]
        tau = pinv(J.T) @ (S @ J @ tau_impedance + (np.eye(6) - S) @ J @ tau_force)
        tau += self.impedance.dynamics.compute_gravity(q)

        return tau

class MotionPlanner:
    def __init__(self, robot: RobotModel):
        self.robot = robot
        self.collision_checker = CollisionChecker(robot)
        self.traj_planner = TrajectoryPlanner(robot)

    def plan_rrt(self, start: np.ndarray, goal: np.ndarray,
                 max_iterations: int = 1000, step_size: float = 0.1) -> Optional[List[np.ndarray]]:
        tree = {tuple(start): None}
        goal_threshold = 0.1

        for _ in range(max_iterations):
            if np.random.random() < 0.1:
                sample = goal
            else:
                sample = self._random_config()

            nearest = self._nearest_neighbor(tree, sample)
            new_config = self._steer(nearest, sample, step_size)

            if not self.collision_checker.check_collision(new_config)[0]:
                tree[tuple(new_config)] = tuple(nearest)

                if np.linalg.norm(new_config - goal) < goal_threshold:
                    return self._reconstruct_path(tree, tuple(new_config))

        return None

    def plan_prm(self, start: np.ndarray, goal: np.ndarray,
                 num_samples: int = 500, k_neighbors: int = 10) -> Optional[List[np.ndarray]]:
        nodes = [start]
        edges = defaultdict(list)

        for _ in range(num_samples):
            sample = self._random_config()
            if not self.collision_checker.check_collision(sample)[0]:
                nodes.append(sample)

        nodes.append(goal)

        for i, node in enumerate(nodes):
            distances = [(j, np.linalg.norm(node - n)) for j, n in enumerate(nodes) if j != i]
            distances.sort(key=lambda x: x[1])

            for j, _ in distances[:k_neighbors]:
                if self._check_edge(nodes[i], nodes[j]):
                    edges[i].append(j)
                    edges[j].append(i)

        return self._astar(edges, 0, len(nodes) - 1, nodes)

    def _random_config(self) -> np.ndarray:
        q = np.zeros(self.robot.dof)
        for i, name in enumerate(self.robot.get_joint_names()):
            lower, upper = self.robot.joint_limits.get(name, (-math.pi, math.pi))
            q[i] = np.random.uniform(lower, upper)
        return q

    def _nearest_neighbor(self, tree: Dict, target: np.ndarray) -> np.ndarray:
        min_dist = float('inf')
        nearest = None
        for config in tree.keys():
            dist = np.linalg.norm(np.array(config) - target)
            if dist < min_dist:
                min_dist = dist
                nearest = config
        return np.array(nearest) if nearest is not None else start

    def _steer(self, from_config: np.ndarray, to_config: np.ndarray, step_size: float) -> np.ndarray:
        direction = to_config - from_config
        dist = np.linalg.norm(direction)
        if dist < step_size:
            return to_config
        return from_config + direction / dist * step_size

    def _check_edge(self, q1: np.ndarray, q2: np.ndarray) -> bool:
        steps = 10
        for i in range(steps + 1):
            alpha = i / steps
            q = q1 + alpha * (q2 - q1)
            if self.collision_checker.check_collision(q)[0]:
                return False
        return True

    def _reconstruct_path(self, tree: Dict, goal: Tuple) -> List[np.ndarray]:
        path = []
        current = goal
        while current is not None:
            path.append(np.array(current))
            current = tree[current]
        path.reverse()
        return path

    def _astar(self, edges: Dict, start_idx: int, goal_idx: int, nodes: List) -> Optional[List[np.ndarray]]:
        import heapq

        open_set = [(0, start_idx)]
        came_from = {}
        g_score = {i: float('inf') for i in range(len(nodes))}
        g_score[start_idx] = 0
        f_score = {i: float('inf') for i in range(len(nodes))}
        f_score[start_idx] = np.linalg.norm(nodes[start_idx] - nodes[goal_idx])

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_idx:
                path = []
                while current in came_from:
                    path.append(nodes[current])
                    current = came_from[current]
                path.append(nodes[start_idx])
                path.reverse()
                return path

            for neighbor in edges[current]:
                tentative_g = g_score[current] + np.linalg.norm(nodes[current] - nodes[neighbor])
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + np.linalg.norm(nodes[neighbor] - nodes[goal_idx])
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None