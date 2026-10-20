from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict


class ControlType(Enum):
    PID = "pid"
    LQR = "lqr"
    MPC = "mpc"
    SMC = "sliding_mode"
    ADAPTIVE = "adaptive"
    ROBUST = "robust"
    OPTIMAL = "optimal"
    PREDICTIVE = "predictive"
    FEEDFORWARD = "feedforward"
    FEEDBACK = "feedback"


class ControllerType(Enum):
    POSITION = "position"
    VELOCITY = "velocity"
    FORCE = "force"
    TORQUE = "torque"
    IMPEDANCE = "impedance"
    ADMITTANCE = "admittance"


@dataclass(frozen=True, slots=True)
class PIDGains:
    kp: float
    ki: float
    kd: float
    integral_limit: float = 100.0
    derivative_filter: float = 0.1


@dataclass(frozen=True, slots=True)
class LQRWeights:
    Q: np.ndarray
    R: np.ndarray


@dataclass(frozen=True, slots=True)
class MPCParams:
    horizon: int = 10
    dt: float = 0.01
    Q: np.ndarray = field(default_factory=lambda: np.eye(6))
    R: np.ndarray = field(default_factory=lambda: np.eye(2))
    constraints: Optional[Dict] = None


@dataclass(frozen=True, slots=True)
class ControlGains:
    kp: np.ndarray
    ki: np.ndarray
    kd: np.ndarray
    kf: np.ndarray = field(default_factory=lambda: np.zeros(6))


class BaseController(ABC):
    @abstractmethod
    def compute_control(
        self,
        state: np.ndarray,
        target: np.ndarray,
        dt: float
    ) -> np.ndarray:
        pass

    @abstractmethod
    def reset(self):
        pass


class PIDController(BaseController):
    def __init__(
        self,
        gains: PIDGains,
        output_limits: Tuple[float, float] = (-100.0, 100.0)
    ):
        self.gains = gains
        self.output_limits = output_limits
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = 0.0

    def compute_control(
        self,
        state: np.ndarray,
        target: np.ndarray,
        dt: float
    ) -> np.ndarray:
        error = target - state

        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.gains.integral_limit, self.gains.integral_limit)

        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0

        control = self.gains.kp * error + self.gains.ki * self.integral + self.gains.kd * derivative

        self.prev_error = error

        return np.clip(control, self.output_limits[0], self.output_limits[1])

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


class MultiDimensionalPIDController(BaseController):
    def __init__(
        self,
        gains: ControlGains,
        output_limits: Tuple[np.ndarray, np.ndarray] = None
    ):
        self.gains = gains
        self.integral = np.zeros_like(gains.kp)
        self.prev_error = np.zeros_like(gains.kp)

        if output_limits is None:
            self.output_limits = (-np.ones_like(gains.kp) * 100, np.ones_like(gains.kp) * 100)
        else:
            self.output_limits = output_limits

    def compute_control(
        self,
        state: np.ndarray,
        target: np.ndarray,
        dt: float
    ) -> np.ndarray:
        error = target - state

        self.integral += error * dt
        self.integral = np.clip(self.integral, -100, 100)

        derivative = (error - self.prev_error) / dt if dt > 0 else np.zeros_like(error)

        control = (self.gains.kp * error +
                   self.gains.ki * self.integral +
                   self.gains.kd * derivative +
                   self.gains.kf * target)

        return np.clip(control, self.output_limits[0], self.output_limits[1])

    def reset(self):
        self.integral = np.zeros_like(self.gains.kp)


class LQRController(BaseController):
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray
    ):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.K = self._solve_lqr()

    def _solve_lqr(self) -> np.ndarray:
        try:
            import scipy.linalg
            P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
            K = np.linalg.inv(self.R) @ self.B.T @ P
            return K
        except ImportError:
            return np.zeros((self.B.shape[1], self.A.shape[0]))

    def compute_control(
        self,
        state: np.ndarray,
        target: np.ndarray,
        dt: float
    ) -> np.ndarray:
        error = state - target
        return -self.K @ error

    def reset(self):
        pass


class MPCController(BaseController):
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        params: MPCParams
    ):
        self.A = A
        self.B = B
        self.params = params
        self.horizon = params.horizon
        self.dt = params.dt

    def compute_control(
        self,
        state: np.ndarray,
        target: np.ndarray,
        dt: float
    ) -> np.ndarray:
        n = self.A.shape[0]
        m = self.B.shape[1]
        N = self.horizon

        x = np.zeros((n, N + 1))
        x[:, 0] = state

        u = np.zeros((self.B.shape[1], N))

        Q = self.params.Q
        R = self.params.R

        for _ in range(10):
            for k in range(N - 1, -1, -1):
                if k == N - 1:
                    P = self.params.Q
                else:
                    P = self.params.Q + self.A.T @ P @ self.A - \
                        self.A.T @ P @ self.B @ np.linalg.inv(self.params.R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A

            K = np.linalg.inv(self.params.R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A

            x = state.copy()
            for k in range(N):
                u[:, k] = -K @ x
                x = self.A @ x + self.B @ u[:, k]

        return u[:, 0]

    def reset(self):
        pass


class SlidingModeController(BaseController):
    def __init__(
        self,
        sliding_surface: Callable[[np.ndarray, np.ndarray], np.ndarray],
        switching_gain: float,
        boundary_layer: float = 0.01
    ):
        self.sliding_surface = sliding_surface
        self.switching_gain = switching_gain
        self.boundary_layer = boundary_layer

    def compute_control(
        self,
        state: np.ndarray,
        target: np.ndarray,
        dt: float
    ) -> np.ndarray:
        s = self.sliding_surface(state, target)

        if np.linalg.norm(s) < self.boundary_layer:
            control = -self.switching_gain * s / (self.boundary_layer + 1e-6)
        else:
            control = -self.switching_gain * s / (np.linalg.norm(s) + 1e-6)

        return control

    def reset(self):
        pass


class AdaptiveController(BaseController):
    def __init__(
        self,
        plant_model: Callable[[np.ndarray, np.ndarray], np.ndarray],
        reference_model: Callable[[np.ndarray], np.ndarray],
        adaptation_gain: float = 1.0
    ):
        self.plant_model = plant_model
        self.reference_model = reference_model
        self.adaptation_gain = adaptation_gain
        self.theta = np.zeros(6)

    def compute_control(
        self,
        state: np.ndarray,
        target: np.ndarray,
        dt: float
    ) -> np.ndarray:
        y_m = self.reference_model(target)
        y_p = state

        error = y_m - state
        self.theta += self.adaptation_gain * error * dt

        u = self.theta @ state
        return u

    def reset(self):
        self.theta = np.zeros(6)


class RobustController(BaseController):
    def __init__(
        self,
        nominal_controller: BaseController,
        uncertainty_bound: float,
        robustness_margin: float = 1.5
    ):
        self.nominal_controller = nominal_controller
        self.uncertainty_bound = uncertainty_bound
        self.robustness_margin = robustness_margin

    def compute_control(
        self,
        state: np.ndarray,
        target: np.ndarray,
        dt: float
    ) -> np.ndarray:
        nominal_control = self.nominal_controller.compute_control(state, target, dt)

        uncertainty = self.uncertainty_bound * np.sign(target - state)
        robust_control = nominal_control + self.robustness_margin * uncertainty

        return robust_control

    def reset(self):
        self.nominal_controller.reset()


class PredictiveController(BaseController):
    def __init__(
        self,
        predictor: Callable[[np.ndarray, np.ndarray], np.ndarray],
        optimizer: BaseController,
        horizon: int = 10
    ):
        self.predictor = predictor
        self.optimizer = optimizer
        self.horizon = horizon

    def compute_control(
        self,
        state: np.ndarray,
        target: np.ndarray,
        dt: float
    ) -> np.ndarray:
        predicted_states = [state]

        for i in range(self.horizon):
            pred = self.predictor(predicted_states[-1], np.zeros_like(target))
            predicted_states.append(pred)

        controls = []
        for pred_state in predicted_states:
            controls.append(self.optimizer.compute_control(pred_state, target, dt))

        return np.mean(controls, axis=0)

    def reset(self):
        self.optimizer.reset()


class FeedforwardController(BaseController):
    def __init__(
        self,
        inverse_model: Callable[[np.ndarray, np.ndarray], np.ndarray],
        feedback_controller: BaseController
    ):
        self.inverse_model = inverse_model
        self.feedback_controller = feedback_controller

    def compute_control(
        self,
        state: np.ndarray,
        target: np.ndarray,
        dt: float
    ) -> np.ndarray:
        ff = self.inverse_model(target, target - state)
        fb = self.feedback_controller.compute_control(state, target, dt)

        return ff + fb

    def reset(self):
        self.feedback_controller.reset()


class ControllerTuner:
    @staticmethod
    def tune_pid(
        plant: Callable[[np.ndarray, np.ndarray], np.ndarray],
        target: np.ndarray,
        kp_range: Tuple[float, float] = (0.1, 100),
        ki_range: Tuple[float, float] = (0.0, 10),
        kd_range: Tuple[float, float] = (0.0, 10),
        max_iter: int = 100
    ) -> PIDGains:
        best_params = PIDGains(1.0, 0.0, 0.0)
        best_score = float('inf')

        for _ in range(max_iter):
            kp = random.uniform(*kp_range)
            ki = random.uniform(*ki_range)
            kd = random.uniform(*kd_range)

            controller = PIDController(PIDGains(kp, ki, kd))
            score = ControllerTuner._evaluate(controller, target)

            if score < best_score:
                best_score = score
                best_params = PIDGains(kp, ki, kd)

        return best_params

    @staticmethod
    def _evaluate(controller: PIDController, target: np.ndarray) -> float:
        state = np.zeros_like(target)
        errors = []

        for _ in range(100):
            control = controller.compute_control(state, target, 0.01)
            state += control * 0.01
            errors.append(np.linalg.norm(target - state))

        return np.mean(errors[-10:])

    @staticmethod
    def auto_tune_lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        try:
            import scipy.linalg
            P = scipy.linalg.solve_continuous_are(A, B, Q, R)
            K = np.linalg.inv(R) @ B.T @ P
            return K
        except ImportError:
            return np.zeros((B.shape[1], A.shape[0]))


class ControlSystem:
    def __init__(
        self,
        controller: BaseController,
        plant: Callable[[np.ndarray, np.ndarray], np.ndarray],
        dt: float = 0.01
    ):
        self.controller = controller
        self.plant = plant
        self.dt = dt
        self.state = np.zeros(6)
        self.target = np.zeros(6)
        self.history: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def set_target(self, target: np.ndarray):
        self.target = target

    def step(self) -> np.ndarray:
        control = self.controller.compute_control(self.state, self.target, self.dt)
        self.state = self.plant(self.state, control)
        self.history.append((self.state.copy(), self.target.copy(), control))
        return control

    def simulate(self, duration: float) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        steps = int(self.dt / self.dt)
        for _ in range(steps):
            self.step()
        return self.history


class ControlSystemFactory:
    @staticmethod
    def create_controller(
        controller_type: ControlType,
        **kwargs
    ) -> BaseController:
        if controller_type == ControlType.PID:
            return PIDController(**kwargs)
        elif controller_type == ControlType.LQR:
            return LQRController(**kwargs)
        elif controller_type == ControlType.MPC:
            return MPCController(**kwargs)
        elif controller_type == ControlType.SMC:
            return SlidingModeController(**kwargs)
        elif controller_type == ControlType.ADAPTIVE:
            return AdaptiveController(**kwargs)
        elif controller_type == ControlType.ROBUST:
            return RobustController(**kwargs)
        elif controller_type == ControlType.PREDICTIVE:
            return PredictiveController(**kwargs)
        elif controller_type == ControlType.FEEDFORWARD:
            return FeedforwardController(**kwargs)
        else:
            return PIDController(**kwargs)


class ControllerMetrics:
    @staticmethod
    def tracking_error(desired: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
        error = desired - actual
        return {
            'rmse': float(np.sqrt(np.mean(error**2))),
            'max_error': float(np.max(np.abs(error))),
            'mean_abs_error': float(np.mean(np.abs(error))),
            'steady_state_error': float(np.mean(np.abs(desired[-10:] - actual[-10:])) if len(desired) >= 10 else np.mean(np.abs(desired - actual)))
        }

    @staticmethod
    def overshoot(desired: np.ndarray, actual: np.ndarray) -> float:
        if len(desired) < 2:
            return 0.0

        steady_state = np.mean(desired[-10:]) if len(desired) >= 10 else np.mean(desired)
        peak = np.max(actual)
        if steady_state > 0:
            return max(0, (peak - steady_state) / steady_state)
        return 0.0

    @staticmethod
    def settling_time(desired: np.ndarray, actual: np.ndarray, tolerance: float = 0.02) -> int:
        if len(desired) != len(actual):
            return -1

        error = np.abs(desired - actual)
        threshold = tolerance * np.abs(desired)

        for i in range(len(error)):
            if np.all(error[i:] < threshold[i:]):
                return i

        return len(desired)

    @staticmethod
    def rise_time(desired: np.ndarray, actual: np.ndarray, threshold: float = 0.9) -> int:
        if len(desired) < 2:
            return -1

        target = desired[-1]
        threshold_val = threshold * target

        for i, val in enumerate(actual):
            if val >= threshold_val:
                return i
        return len(actual)


class ControllerVisualizer:
    @staticmethod
    def draw_response(
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


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
