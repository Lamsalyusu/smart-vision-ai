from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math
import time


class ControlType(Enum):
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    IMPEDANCE = "impedance"
    ADMITTANCE = "admittance"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class JointCommand:
    position: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    torque: Optional[np.ndarray] = None
    stiffness: Optional[np.ndarray] = None
    damping: Optional[np.ndarray] = None
    control_type: ControlType = ControlType.POSITION


@dataclass(frozen=True, slots=True)
class JointState:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    torque: np.ndarray
    timestamp: float = 0.0


@dataclass(frozen=True, slots=True)
class RobotState:
    joint_states: List[JointState]
    base_pose: Optional[np.ndarray] = None
    base_velocity: Optional[np.ndarray] = None
    contact_states: List[bool] = field(default_factory=list)


class Controller(ABC):
    @abstractmethod
    def compute(self, command: JointCommand, state: RobotState, dt: float) -> np.ndarray:
        pass

    @abstractmethod
    def reset(self):
        pass


class PIDController:
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
        error = desired_state.position - current_state.position
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


class ComputedTorqueController:
    def __init__(
        self,
        robot_model: 'RobotModel',
        kp: np.ndarray,
        kv: np.ndarray,
    ):
        self.robot_model = robot_model
        self.kp = kp
        self.kv = kv

    def compute(
        self,
        current_state: JointState,
        desired_state: JointState,
        dt: float,
    ) -> np.ndarray:
        pos_error = desired_state.position - current_state.position
        vel_error = desired_state.velocity - current_state.velocity

        desired_acc = desired_state.acceleration + self.kv @ vel_error + self.kp @ pos_error

        tau = self.robot_model.inverse_dynamics(current_state, desired_acc)
        return tau


class ImpedanceController:
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
        pos_error = desired_state.position - current_state.position
        vel_error = desired_state.velocity - current_state.velocity

        tau = self.stiffness @ pos_error + self.damping @ vel_error

        if hasattr(current_state, 'external_force') and current_state.external_force is not None:
            tau += current_state.external_force

        return tau


class AdmittanceController:
    def __ __init__(
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
        force_error = desired_state.torque - current_state.torque if hasattr(desired_state, 'torque') else np.zeros_like(self.mass)

        acc = (force_error - self.damping @ self.velocity - self.stiffness @ (self.position - desired_state.position)) / self.mass
        self.velocity += acc * dt
        self.position += self.velocity * dt

        return self.position


class OperationalSpaceController:
    def __init__(
        self,
        robot_model: 'RobotModel',
        kp: np.ndarray,
        kd: np.ndarray,
        nullspace_stiffness: float = 0.0,
    ):
        self.robot_model = robot_model
        self.kp = kp
        self.kd = kd
        self.nullspace_stiffness = nullspace_stiffness

    def compute(
        self,
        current_state: JointState,
        desired_pose: np.ndarray,
        desired_vel: np.ndarray,
        desired_acc: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        current_pose = self.robot_model.forward_kinematics(current_state.position)[-1][:3, 3]
        current_vel = self.robot_model.jacobian(current_state.position) @ current_state.velocity

        pos_error = desired_pose - current_pose
        vel_error = desired_vel - current_vel

        desired_acc_op = desired_acc + self.kd @ vel_error + self.kp @ pos_error

        J = self.robot_model.jacobian(current_state.position)
        J_pinv = np.linalg.pinv(J)

        lambda_x = np.linalg.inv(J @ np.linalg.inv(self.robot_model.mass_matrix(current_state.position)) @ J.T)
        F = lambda_x @ (desired_acc_op + self.kd @ vel_error + self.kp @ pos_error)

        tau = J.T @ F

        if self.nullspace_stiffness > 0:
            N = np.eye(len(self.robot_model.joint_limits)) - J_pinv @ J
            q0 = self.robot_model.rest_position
            tau += self.nullspace_stiffness * N @ (self.robot_model.rest_position - current_state.position)

        return tau


class ForceController:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        target_force: float,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_force = target_force
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(
        self,
        current_state: JointState,
        desired_force: float,
        dt: float,
    ) -> float:
        error = desired_force - current_state.torque[0] if len(current_state.torque) > 0 else 0
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        return self.kp * error + self.ki * self.integral + self.kd * derivative


class HybridForcePositionController:
    def __init__(
        self,
        position_controller: Controller,
        force_controller: ForceController,
        selection_matrix: np.ndarray,
    ):
        self.position_controller = position_controller
        self.force_controller = force_controller
        self.S = selection_matrix

    def compute(
        self,
        current_state: JointState,
        desired_state: JointState,
        dt: float,
    ) -> np.ndarray:
        pos_out = self.position_controller.compute(current_state, desired_state, dt)
        force_out = self.force_controller.compute(current_state, 0, 0)

        return (np.eye(6) - self.S) @ pos_out + self.S @ np.array([force_out] * 6)


class ControllerManager:
    def __ __init__(self):
        self.controllers: Dict[str, Controller] = {}
        self.active_controller: Optional[str] = None

    def add_controller(self, name: str, controller: Controller):
        self.controllers[name] = controller

    def set_active(self, name: str):
        if name in self.controllers:
            self.active_controller = name

    def compute(self, command: JointCommand, state: RobotState, dt: float) -> np.ndarray:
        if self.active_controller and self.active_controller in self.controllers:
            return self.controllers[self.active_controller].compute(command, state, dt)
        return np.zeros(6)

    def switch_controller(self, name: str, state: RobotState):
        if name in self.controllers:
            if self.active_controller and self.active_controller in self.controllers:
                self.controllers[self.active_controller].reset()
            self.active_controller = name
            self.controllers[name].reset()


class RobotModel:
    def __init__(
        self,
        num_joints: int,
        link_masses: List[float],
        link_coms: List[np.ndarray],
        link_inertias: List[np.ndarray],
        joint_axes: List[np.ndarray],
        joint_types: List[str],
    ):
        self.num_joints = num_joints
        self.link_masses = link_masses
        self.link_coms = link_coms
        self.link_inertias = link_inertias
        self.joint_axes = joint_axes
        self.joint_types = joint_types

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

    def inverse_dynamics(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        return np.zeros(self.num_joints)

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        return np.eye(self.num_joints)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        return np.eye(self.num_joints)

    @property
    def joint_limits(self):
        return [(-np.pi, np.pi)] * self.num_joints

    @property
    def rest_position(self):
        return np.zeros(self.num_joints)


class ControllerSwitcher:
    def __init__(self, controllers: Dict[str, Controller]):
        self.controllers = controllers
        self.current = None

    def switch(self, name: str):
        if name in self.controllers:
            if self.current:
                self.controllers[self.current].reset()
            self.current = name

    def compute(self, command, state, dt):
        if self.current and self.current in self.controllers:
            return self.controllers[self.current].compute(command, state, dt)
        return np.zeros(6)


class GainScheduler:
    def __init__(self, base_gains: Dict[str, Dict[str, float]]):
        self.base_gains = base_gains
        self.current_schedule = "default"

    def schedule(self, condition: str, gains: Dict[str, float]):
        self.base_gains[condition] = gains

    def get_gains(self, condition: str) -> Dict[str, float]:
        return self.base_gains.get(condition, self.base_gains.get("default", {}))


class AdaptiveController:
    def __init__(
        self,
        nominal_controller: Controller,
        adaptation_rate: float = 0.01,
    ):
        self.nominal_controller = nominal_controller
        self.adaptation_rate = adaptation_rate
        self.parameter_estimates = {}

    def compute(self, command, state, dt):
        output = self.nominal_controller.compute(command, state, dt)
        self._adapt_parameters(command, state)
        return output

    def _adapt_parameters(self, command, state):
        pass


class SlidingModeController:
    def __init__(self, lambda_: float, gain: float, phi: float = 0.01):
        self.lambda_ = lambda_
        self.gain = gain
        self.phi = phi

    def compute(self, error: np.ndarray, error_dot: np.ndarray) -> np.ndarray:
        s = error_dot + self.lambda_ @ error
        if np.any(np.abs(s) > self.phi):
            return -self.gain * np.sign(s)
        return -self.gain * s / self.phi


class BacksteppingController:
    def __init__(self, gains: List[float]):
        self.gains = gains

    def compute(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        z1 = x[0] - r[0]
        alpha1 = -self.gains[0] * z1

        z2 = x[1] - alpha1
        u = -self.gains[1] * z2 + self.gains[0] * z1

        return np.array([u])


class ModelReferenceAdaptiveController:
    def __init__(
        self,
        Am: np.ndarray,
        Bm: np.ndarray,
        adaptation_gain: np.ndarray,
        gamma: float = 1.0,
    ):
        self.Am = Am
        self.Bm = Bm
        self.Gamma = adaptation_gain
        self.gamma = gamma
        self.theta = np.zeros((Am.shape[0], Am.shape[1]))

    def compute(self, state: np.ndarray, reference: np.ndarray, dt: float) -> float:
        e = state - reference
        e_dot = self.Am @ e + self.Bm @ reference - self.Am @ reference - self.Bm @ reference

        self.theta += self.Gamma @ np.outer(e, state) * dt

        u = self.theta @ state + self.gamma * reference
        return float(u)


class GainSchedulingController:
    def __init__(self):
        self.gain_schedules: Dict[str, Dict[str, np.ndarray]] = {}

    def add_schedule(self, condition: str, gains: Dict[str, np.ndarray]):
        self.gain_schedules[condition] = gains

    def get_gains(self, condition: str) -> Dict[str, np.ndarray]:
        return self.gain_schedules.get(condition, {})


class PIDGainTuner:
    def __init__(self):
        self.tuning_rules = {
            'ziegler_nichols': self._ziegler_nichols,
            'cohen_coon': self._cohen_coon,
            'lambda': self._lambda_tuning,
            'imc': self._imc_tuning,
        }

    def tune(self, method: str, K: float, L: float, T: float) -> Dict[str, float]:
        if method in self.tuning_rules:
            return self.tuning_rules[method](K, L, T)
        return {'kp': 1.0, 'ki': 0.1, 'kd': 0.01}

    def _ziegler_nichols(self, K: float, L: float, T: float) -> Dict[str, float]:
        kp = 1.2 * T / (K * L)
        ki = kp / (2 * L)
        kd = kp * 0.5 * L
        return {'kp': kp, 'ki': ki, 'kd': kd}

    def _cohen_coon(self, K: float, L: float, T: float) -> Dict[str, float]:
        kp = (1.35 / K) * (T / L) + 0.25
        ki = kp / (L * (2.5 + 0.5 * T / L))
        kd = kp * 0.37 * L / (1 + 0.2 * T / L)
        return {'kp': kp, 'ki': ki, 'kd': kd}

    def _lambda_tuning(self, K: float, L: float, T: float, lambda_factor: float = 1.0) -> Dict[str, float]:
        tau_c = lambda_factor * T
        kp = tau / (K * (tau_c + L))
        ki = 1 / (K * (tau_c + L))
        kd = 0.0
        return {'kp': kp, 'ki': ki, 'kd': kd}

    def _imc_tuning(self, K: float, L: float, T: float, tau_c: float = None) -> Dict[str, float]:
        if tau_c is None:
            tau_c = max(0.1 * T, 0.8 * L)
        kp = (tau + L / 2) / (K * (tau_c + L / 2))
        ki = 1 / (K * (tau_c + L / 2))
        kd = tau * L / (2 * tau + L)
        return {'kp': kp, 'ki': ki, 'kd': kd}


class ControllerDesigner:
    @staticmethod
    def design_pid_for_foptd(K: float, L: float, T: float, method: str = "ziegler_nichols") -> Dict[str, float]:
        tuner = PIDGainTuner()
        return tuner.tune(method, K, L, T)

    @staticmethod
    def design_lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray = None, R: np.ndarray = None) -> np.ndarray:
        if Q is None:
            Q = np.eye(A.shape[0])
        if R is None:
            R = np.eye(B.shape[1]) if B.ndim > 1 else np.array([[1.0]])
        return np.linalg.solve(R, B.T @ np.linalg.solve(A.T @ Q + Q @ A, Q @ B))

    @staticmethod
    def design_mpc(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, horizon: int = 10) -> Dict:
        return {'A': A, 'B': B, 'Q': Q, 'R': R, 'horizon': horizon}

    @staticmethod
    def design_sliding_mode(lambda_: np.ndarray, k: float, phi: float = 0.01) -> SlidingModeController:
        return SlidingModeController(lambda_, k, phi)

    @staticmethod
    def design_backstepping(gains: List[float]) -> BacksteppingController:
        return BacksteppingController(gains)

    @staticmethod
    def design_model_reference(Am: np.ndarray, Bm: np.ndarray, Gamma: np.ndarray) -> ModelReferenceAdaptiveController:
        return ModelReferenceAdaptiveController(Am, Bm, Gamma)


class ControlLoop:
    def __init__(
        self,
        controller: Controller,
        process: Callable[[float], float],
        dt: float = 0.01,
    ):
        self.controller = controller
        self.process = process
        self.dt = dt
        self.running = False
        self.history = []

    def run(self, setpoint: float, duration: float) -> List[Tuple[float, float, float]]:
        t = 0.0
        y = 0.0
        results = []

        while t < duration:
            u = self.controller.update(setpoint, y, self.dt)
            y = self.process(u)
            self.history.append((t, setpoint, y))
            results.append((t, setpoint, y))
            t += self.dt

        return results


class MultiLoopController:
    def __init__(self):
        self.loops: Dict[str, ControlLoop] = {}
        self.coupling: Dict[str, Dict[str, float]] = {}

    def add_loop(self, name: str, loop: ControlLoop):
        self.loops[name] = loop

    def add_coupling(self, from_loop: str, to_loop: str, gain: float):
        if from_loop not in self.coupling:
            self.coupling[from_loop] = {}
        self.coupling[from_loop][to_loop] = gain

    def run(self, setpoints: Dict[str, float], duration: float) -> Dict[str, List[Tuple[float, float, float]]]:
        results = {}
        for name, loop in self.loops.items():
            results[name] = loop.run(setpoints.get(name, 0.0), duration)
        return results


class CascadeController:
    def __init__(
        self,
        outer_controller: Controller,
        inner_controller: Controller,
    ):
        self.outer = outer_controller
        self.inner = inner_controller

    def update(self, outer_setpoint: float, outer_measured: float, inner_measured: float, dt: float) -> float:
        inner_setpoint = self.outer.update(outer_setpoint, outer_measured, dt)
        inner_output = self.inner.update(inner_setpoint, inner_measured, dt)
        return inner_output


class RatioController:
    def __init__(self, ratio: float, wild_stream: Callable[[], float]):
        self.ratio = ratio
        self.wild_stream = wild_stream

    def update(self, controlled_stream: float, dt: float) -> float:
        wild = self.wild_stream()
        setpoint = self.ratio * wild
        error = setpoint - controlled_stream
        return error


class OverrideController:
    def __init__(self, primary: Controller, secondary: Controller, condition: Callable[[float], bool]):
        self.primary = primary
        self.secondary = secondary
        self.condition = condition

    def update(self, setpoint: float, measured: float, dt: float) -> float:
        if self.condition(measured):
            return self.secondary.update(setpoint, measured, dt)
        return self.primary.update(setpoint, measured, dt)


class SplitRangeController:
    def __init__(
        self,
        controller: Controller,
        split_point: float,
        valve1_range: Tuple[float, float],
        valve2_range: Tuple[float, float],
    ):
        self.controller = controller
        self.split_point = split_point
        self.valve1_range = valve1_range
        self.valve2_range = valve2_range

    def update(self, setpoint: float, measured: float, dt: float) -> Tuple[float, float]:
        output = self.controller.update(setpoint, measured, dt)

        if output <= self.split_point:
            valve1 = self._scale(output, 0, self.split_point, self.valve1_range[0], self.valve1_range[1])
            valve2 = self.valve2_range[0]
        else:
            valve1 = self.valve1_range[1]
            valve2 = self._scale(output, self.split_point, 100, self.valve2_range[0], self.valve2_range[1])

        return valve1, valve2

    def _scale(self, x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
        return out_min + (x - in_min) * (out_max - out_min) / (in_max - in_min)