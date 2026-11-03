from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict


class ControlMode(Enum):
    POSITION = "position"
    VELOCITY = "velocity"
    FORCE = "force"
    IMPEDANCE = "impedance"
    ADMITTANCE = "admittance"
    HYBRID = "hybrid"
    COMPLIANCE = "compliance"


class ForceControlAlgorithm(Enum):
    EXPLICIT_FORCE = "explicit_force"
    IMPLICIT_FORCE = "implicit_force"
    HYBRID_POSITION_FORCE = "hybrid_position_force"
    IMPEDANCE_CONTROL = "impedance_control"
    ADMITTANCE_CONTROL = "admittance_control"
    PARALLEL_FORCE_POSITION = "parallel_force_position"
    STIFFNESS_CONTROL = "stiffness_control"
    DAMPING_CONTROL = "damping_control"


@dataclass(frozen=True, slots=True)
class ForceVector:
    force: np.ndarray
    torque: np.ndarray

    @property
    def wrench(self) -> np.ndarray:
        return np.concatenate([self.force, self.torque])

    @staticmethod
    def from_wrench(wrench: np.ndarray) -> 'ForceVector':
        return ForceVector(wrench[:3], wrench[3:])


@dataclass(frozen=True, slots=True)
class ImpedanceParameters:
    stiffness: np.ndarray
    damping: np.ndarray
    inertia: np.ndarray

    @property
    def K(self) -> np.ndarray:
        return np.diag(self.stiffness) if self.stiffness.ndim == 1 else self.stiffness

    @property
    def D(self) -> np.ndarray:
        return np.diag(self.damping) if self.damping.ndim == 1 else self.damping

    @property
    def M(self) -> np.ndarray:
        return np.diag(self.inertia) if self.inertia.ndim == 1 else self.inertia


@dataclass(frozen=True, slots=True)
class ForceControlTarget:
    desired_force: np.ndarray
    desired_torque: np.ndarray
    stiffness: Optional[np.ndarray] = None
    damping: Optional[np.ndarray] = None

    @property
    def wrench(self) -> np.ndarray:
        return np.concatenate([self.desired_force, self.desired_torque])


class BaseForceController(ABC):
    @abstractmethod
    def compute_control(
        self,
        current_pose: np.ndarray,
        current_wrench: ForceVector,
        target: ForceControlTarget,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass


class ExplicitForceController(BaseForceController):
    def __init__(
        self,
        kp: float = 100.0,
        ki: float = 10.0,
        kd: float = 1.0,
        max_force: float = 100.0
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_force = max_force
        self.integral_error = np.zeros(6)
        self.prev_error = np.zeros(6)

    def compute_control(
        self,
        current_pose: np.ndarray,
        current_wrench: ForceVector,
        target: ForceControlTarget,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        error = target.wrench - current_wrench.wrench

        self.integral_error += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else np.zeros(6)
        self.prev_error = error

        control = self.kp * error + self.ki * self.integral_error + self.kd * derivative

        force_norm = np.linalg.norm(control[:3])
        if force_norm > self.max_force:
            control[:3] = control[:3] * self.max_force / force_norm

        torque_norm = np.linalg.norm(control[3:])
        if torque_norm > self.max_force:
            control[3:] = control[3:] * self.max_force / torque_norm

        return control[:3], control[3:]


class ImpedanceController(BaseForceController):
    def __init__(
        self,
        impedance_params: ImpedanceParameters,
        target_pose: np.ndarray = None
    ):
        self.impedance = impedance_params
        self.target_pose = target_pose if target_pose is not None else np.eye(4)

    def compute_control(
        self,
        current_pose: np.ndarray,
        current_wrench: ForceVector,
        target: ForceControlTarget,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        pos_error = self._compute_pose_error(current_pose, self.target_pose)

        K = self.impedance.K
        D = self.impedance.D
        M = self.impedance.M

        pos_error_vec = np.concatenate([
            self.target_pose[:3, 3] - current_pose[:3, 3],
            self._rotation_error(current_pose[:3, :3], self.target_pose[:3, :3])
        ])

        current_vel = np.concatenate([
            np.zeros(3),
            np.zeros(3)
        ])

        desired_wrench = K @ pos_error_vec + D @ (-current_vel)
        desired_wrench = desired_wrench + target.wrench

        return desired_wrench[:3], desired_wrench[3:]

    def _compute_pose_error(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        pos_error = target[:3, 3] - current[:3, 3]
        rot_error = self._rotation_error(current[:3, :3], target[:3, :3])
        return np.concatenate([pos_error, rot_error])

    def _rotation_error(self, R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
        R_err = R_target @ R_current.T
        angle_axis = cv2.Rodrigues(R_err)[0].flatten()
        return angle_axis


class AdmittanceController(BaseForceController):
    def __init__(
        self,
        admittance_params: ImpedanceParameters,
        target_pose: np.ndarray = None
    ):
        self.admittance = admittance_params
        self.target_pose = target_pose if target_pose is not None else np.eye(4)
        self.current_pose = np.eye(4)

    def compute_control(
        self,
        current_pose: np.ndarray,
        current_wrench: ForceVector,
        target: ForceControlTarget,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.current_pose = current_pose

        M_inv = np.linalg.inv(self.admittance.M)
        D_inv = np.linalg.inv(self.admittance.D + 1e-6 * np.eye(6))
        K = self.admittance.K

        wrench_error = target.wrench - np.concatenate([np.zeros(3), np.zeros(3)])
        accel = M_inv @ (wrench_error - self.admittance.D @ np.zeros(6) - self.admittance.K @ np.zeros(6))

        return np.zeros(3), np.zeros(3)

    def update_pose(self, dt: float):
        pass


class HybridForcePositionController(BaseForceController):
    def __init__(
        self,
        selection_matrix: np.ndarray,
        force_controller: BaseForceController,
        position_controller: BaseForceController
    ):
        self.S = selection_matrix
        self.force_controller = force_controller
        self.position_controller = position_controller

    def compute_control(
        self,
        current_pose: np.ndarray,
        current_wrench: ForceVector,
        target: ForceControlTarget,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        S = self.S
        I_minus_S = np.eye(6) - S

        force_target = ForceControlTarget(
            desired_force=S[:3, :3] @ target.desired_force,
            desired_torque=S[3:, 3:] @ target.desired_torque
        )

        force_command = self.force_controller.compute_control(
            np.eye(4), current_wrench, force_target, 0.01
        )

        position_target = ForceControlTarget(
            desired_force=I_minus_S[:3, :3] @ target.desired_force,
            desired_torque=I_minus_S[3:, 3:] @ target.desired_torque
        )

        position_command = self.position_controller.compute_control(
            np.eye(4), current_wrench, position_target, 0.01
        )

        force_cmd = np.concatenate(force_command)
        pos_cmd = np.concatenate(position_command)

        total_command = S @ force_cmd + I_minus_S @ pos_cmd

        return total_command[:3], total_command[3:]


class ParallelForcePositionController(BaseForceController):
    def __init__(
        self,
        force_controller: BaseForceController,
        position_controller: BaseForceController,
        force_weight: float = 0.5
    ):
        self.force_controller = force_controller
        self.position_controller = position_controller
        self.force_weight = force_weight

    def compute_control(
        self,
        current_pose: np.ndarray,
        current_wrench: ForceVector,
        target: ForceControlTarget,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        force_command = self.force_controller.compute_control(
            current_pose, current_wrench, target, dt
        )

        position_target = ForceControlTarget(
            desired_force=np.zeros(3),
            desired_torque=np.zeros(3)
        )
        position_command = self.position_controller.compute_control(
            current_pose, current_wrench, position_target, dt
        )

        force_cmd = np.concatenate(force_command)
        pos_cmd = np.concatenate(position_command)

        total_command = self.force_weight * force_cmd + (1 - self.force_weight) * pos_cmd

        return total_command[:3], total_command[3:]


class StiffnessController(BaseForceController):
    def __init__(self, stiffness: np.ndarray):
        self.K = np.diag(stiffness) if stiffness.ndim == 1 else stiffness

    def compute_control(
        self,
        current_pose: np.ndarray,
        current_wrench: ForceVector,
        target: ForceControlTarget,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        pos_error = target.desired_force
        force = self.K @ pos_error
        return force[:3], force[3:]


class DampingController(BaseForceController):
    def __init__(self, damping: np.ndarray):
        self.D = np.diag(damping) if damping.ndim == 1 else damping

    def compute_control(
        self,
        current_pose: np.ndarray,
        current_wrench: ForceVector,
        target: ForceControlTarget,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        force = -self.D @ np.zeros(6)
        return force[:3], force[3:]


class ForceControlPipeline:
    def __init__(
        self,
        controller: BaseForceController,
        max_force: float = 100.0,
        max_torque: float = 50.0
    ):
        self.controller = controller
        self.max_force = max_force
        self.max_torque = max_torque
        self.dt = 0.001

    def compute_command(
        self,
        current_pose: np.ndarray,
        current_wrench: ForceVector,
        target: ForceControlTarget
    ) -> Tuple[np.ndarray, np.ndarray]:
        force, torque = self.controller.compute_control(
            current_pose, current_wrench, target, self.dt
        )

        force_norm = np.linalg.norm(force)
        if force_norm > self.max_force:
            force = force * self.max_force / force_norm

        torque_norm = np.linalg.norm(torque)
        if torque_norm > self.max_torque:
            torque = torque * self.max_torque / torque_norm

        return force, torque

    def set_dt(self, dt: float):
        self.dt = dt


class ForceControllerFactory:
    @staticmethod
    def create_controller(
        algorithm: ForceControlAlgorithm,
        **kwargs
    ) -> BaseForceController:
        if algorithm == ForceControlAlgorithm.EXPLICIT_FORCE:
            return ExplicitForceController(**kwargs)
        elif algorithm == ForceControlAlgorithm.IMPEDANCE_CONTROL:
            return ImpedanceController(**kwargs)
        elif algorithm == ForceControlAlgorithm.ADMITTANCE_CONTROL:
            return AdmittanceController(**kwargs)
        elif algorithm == ForceControlAlgorithm.HYBRID_POSITION_FORCE:
            return HybridForcePositionController(**kwargs)
        elif algorithm == ForceControlAlgorithm.PARALLEL_FORCE_POSITION:
            return ParallelForcePositionController(**kwargs)
        elif algorithm == ForceControlAlgorithm.STIFFNESS_CONTROL:
            return StiffnessController(**kwargs)
        elif algorithm == ForceControlAlgorithm.DAMPING_CONTROL:
            return DampingController(**kwargs)
        else:
            return ExplicitForceController(**kwargs)


class ForceControlMetrics:
    @staticmethod
    def force_tracking_error(
        desired: np.ndarray,
        actual: np.ndarray
    ) -> Dict[str, float]:
        error = desired - actual
        return {
            'rmse': float(np.sqrt(np.mean(error**2))),
            'max_error': float(np.max(np.abs(desired - actual))),
            'mean_abs_error': float(np.mean(np.abs(desired - actual))),
            'steady_state_error': float(np.mean(np.abs(desired[-10:] - actual[-10:])) if len(desired) >= 10 else np.mean(np.abs(desired - actual)))
        }

    @staticmethod
    def force_overshoot(desired: np.ndarray, actual: np.ndarray) -> float:
        if len(desired) < 2:
            return 0.0

        steady_state = np.mean(desired[-10:]) if len(desired) >= 10 else np.mean(desired)
        peak = np.max(actual)
        if steady_state > 0:
            return float(max(0, (peak - steady_state) / steady_state))
        return 0.0

    @staticmethod
    def force_settling_time(desired: np.ndarray, actual: np.ndarray, tolerance: float = 0.02) -> int:
        if len(desired) != len(actual):
            return -1

        error = np.abs(desired - actual)
        threshold = tolerance * np.abs(desired)

        for i in range(len(error)):
            if np.all(error[i:] < threshold[i:]):
                return i

        return len(desired)

    @staticmethod
    def contact_stiffness_estimation(
        force: np.ndarray,
        displacement: np.ndarray
    ) -> float:
        if len(force) < 2 or len(displacement) < 2:
            return 0.0

        coeffs = np.polyfit(displacement, force, 1)
        return float(coeffs[0])


class ForceControllerTuner:
    @staticmethod
    def tune_pid(
        plant_model: Callable,
        target_force: np.ndarray,
        kp_range: Tuple[float, float] = (0.1, 1000),
        ki_range: Tuple[float, float] = (0.0, 100),
        kd_range: Tuple[float, float] = (0.0, 10)
    ) -> Tuple[float, float, float]:
        best_params = (1.0, 0.0, 0.0)
        best_score = float('inf')

        for kp in np.logspace(np.log10(kp_range[0]), np.log10(kp_range[1]), 10):
            for ki in np.logspace(np.log10(max(ki_range[0], 1e-6)), np.log10(ki_range[1]), 5):
                for kd in np.logspace(np.log10(max(kd_range[0], 1e-6)), np.log10(kd_range[1]), 5):
                    controller = ExplicitForceController(kp=kp, ki=ki, kd=kd)

                    score = ForceControllerTuner._evaluate_controller(controller, target_force=np.array([10.0]))
                    if score < best_score:
                        best_score = score
                        best_params = (kp, ki, kd)

        return best_params

    @staticmethod
    def _evaluate_controller(controller: BaseForceController, target_force: np.ndarray) -> float:
        current_wrench = ForceVector(np.zeros(3), np.zeros(3))
        target = ForceControlTarget(desired_force=target_force, desired_torque=np.zeros(3))

        errors = []
        for i in range(100):
            force, torque = controller.compute_control(np.eye(4), current_wrench, target, 0.01)
            error = np.abs(target_force[0] - force[0])
            errors.append(error)

        return np.mean(errors[-10:]) if len(errors) >= 10 else np.mean(errors)


class ForceControlVisualizer:
    @staticmethod
    def draw_force_profile(
        desired: np.ndarray,
        actual: np.ndarray,
        image_size: Tuple[int, int] = (800, 400)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        n = len(desired)
        if n < 2:
            return vis

        x_scale = (image_size[0] - 100) / (n - 1)
        y_scale = image_size[1] / 2

        max_val = max(np.max(np.abs(desired)), np.max(np.abs(actual)))
        if max_val == 0:
            max_val = 1.0

        y_center = image_size[1] // 2

        prev_d = None
        prev_a = None

        for i in range(n):
            x = 50 + i * (image_size[0] - 100) / (n - 1)
            y_d = int(y_scale - desired[i] / max_val * y_scale)
            y_a = int(y_scale - actual[i] / max_val * y_scale)

            if prev_d is not None:
                cv2.line(vis, (int(prev_d[0]), int(prev_d[1])), (int(x), int(y_d)), (0, 255, 0), 2)
            if prev_a is not None:
                cv2.line(vis, (int(prev_a[0]), int(prev_a[1])), (int(x), int(y_a)), (0, 0, 255), 2)

            prev_d = (x, y_d)
            prev_a = (x, y_a)

        cv2.putText(vis, "Desired", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis, "Actual", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return vis

    @staticmethod
    def draw_wrench(
        wrench: np.ndarray,
        image_size: Tuple[int, int] = (400, 400)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        center = (image_size[0] // 2, image_size[1] // 2)
        scale = min(image_size[0], image_size[1]) / 200.0

        force = wrench[:3]
        torque = wrench[3:]

        for i, (label, vec, color) in enumerate([
            ("Force", force, (0, 255, 0)),
            ("Torque", torque, (0, 0, 255))
        ]):
            if np.linalg.norm(vec) > 1e-6:
                end = (
                    int(center[0] + vec[0] * 10),
                    int(center[1] - vec[1] * 10)
                )
                cv2.arrowedLine(vis, center, end, color, 2)
                cv2.putText(vis, f"{label}: {np.linalg.norm(vec):.2f}", (10, 30 + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return vis


import cv2
from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
