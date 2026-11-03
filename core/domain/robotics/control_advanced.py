from __future__ import annotations

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math


class ControllerType(Enum):
    PID = "pid"
    PI = "pi"
    PD = "pd"
    FEEDFORWARD = "feedforward"
    LQR = "lqr"
    MPC = "mpc"
    ADAPTIVE = "adaptive"
    SLIDING_MODE = "sliding_mode"
    BACKSTEPPING = "backstepping"
    MODEL_REFERENCE = "model_reference"


@dataclass(frozen=True, slots=True)
class PIDGains:
    kp: float
    ki: float
    kd: float
    kf: float = 0.0


@dataclass(frozen=True, slots=True)
class ControllerState:
    setpoint: float
    process_variable: float
    error: float
    integral: float
    derivative: float
    output: float
    timestamp: float


class BaseController(ABC):
    @abstractmethod
    def update(self, setpoint: float, measured: float, dt: float) -> float:
        pass

    @abstractmethod
    def reset(self):
        pass


class PIDController(BaseController):
    def __init__(
        self,
        gains: PIDGains,
        output_limits: Tuple[float, float] = (-float('inf'), float('inf')),
        integral_limits: Tuple[float, float] = (-float('inf'), float('inf')),
        derivative_filter: float = 0.0,
        anti_windup: bool = True,
    ):
        self.gains = gains
        self.output_limits = output_limits
        self.integral_limits = integral_limits
        self.derivative_filter = derivative_filter
        self.anti_windup = anti_windup

        self.integral = 0.0
        self.previous_error = 0.0
        self.filtered_derivative = 0.0
        self.last_output = 0.0
        self.last_time = 0.0

    def update(self, setpoint: float, measured: float, dt: float) -> float:
        error = setpoint - measured

        self.integral += error * dt
        self.integral = max(self.integral_limits[0], min(self.integral_limits[1], self.integral))

        derivative = (error - self.previous_error) / dt if dt > 0 else 0.0
        if self.derivative_filter > 0:
            self.filtered_derivative = (self.derivative_filter * derivative +
                                         (1 - self.derivative_filter) * self.filtered_derivative)
            derivative = self.filtered_derivative

        output = (self.gains.kp * error +
                  self.gains.ki * self.integral +
                  self.gains.kd * derivative +
                  self.gains.kf * setpoint)

        if self.output_limits[0] != -float('inf') or self.output_limits[1] != float('inf'):
            if self.anti_windup and self.last_output != 0:
                if (output > self.output_limits[1] and self.integral > 0) or \
                   (output < self.output_limits[0] and self.integral < 0):
                    pass
                else:
                    output = max(self.output_limits[0], min(self.output_limits[1], output))
            else:
                output = max(self.output_limits[0], min(self.output_limits[1], output))

        self.previous_error = setpoint - measured
        self.last_output = output

        return output

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.filtered_derivative = 0.0
        self.last_output = 0.0


class PIController(PIDController):
    def __init__(self, kp: float, ki: float, **kwargs):
        super().__init__(PIDGains(kp=kp, ki=ki, kd=0.0), **kwargs)


class PDController(PIDController):
    def __init__(self, kp: float, kd: float, **kwargs):
        super().__init__(PIDGains(kp=kp, ki=0.0, kd=kd), **kwargs)


class FeedforwardController:
    def __init__(self, model: Callable[[float], float]):
        self.model = model

    def compute(self, setpoint: float, setpoint_derivative: float = 0.0) -> float:
        return self.model(setpoint)


class PIDFeedforwardController:
    def __init__(self, pid: PIDController, feedforward: FeedforwardController):
        self.pid = pid
        self.feedforward = feedforward

    def update(self, setpoint: float, measured: float, dt: float) -> float:
        pid_output = self.pid.update(setpoint, measured, dt)
        ff_output = self.feedforward.compute(setpoint)
        return pid_output + ff_output


class LQRController:
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
    ):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.K = self._compute_gain()
        self.x = np.zeros(A.shape[0])

    def _compute_gain(self) -> np.ndarray:
        P = self._solve_riccati()
        return np.linalg.inv(self.R) @ self.B.T @ P

    def _solve_riccati(self, max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
        n = self.A.shape[0]
        P = np.eye(n)
        for _ in range(max_iter):
            P_new = self.A.T @ P @ self.A - \
                    self.A.T @ P @ self.B @ np.linalg.inv(self.R + self.B.T @ P @ self.B) @ \
                    self.B.T @ P @ self.A + self.Q
            if np.linalg.norm(P_new - P) < tol:
                break
            P = P_new
        return P

    def update(self, state: np.ndarray, reference: np.ndarray = None) -> float:
        if reference is None:
            reference = np.zeros_like(state)
        error = reference - state
        self.x = state
        u = -self.K @ error
        return float(u[0]) if u.ndim > 0 else float(u)


class MPCController:
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        P: np.ndarray,
        horizon: int = 10,
        constraints: Optional[Dict] = None,
    ):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.P = P
        self.horizon = horizon
        self.constraints = constraints or {}

    def update(self, state: np.ndarray, reference: np.ndarray = None) -> float:
        if reference is None:
            reference = np.zeros_like(state)

        n = state.shape[0]
        m = self.B.shape[1]

        H = self._build_hessian()
        f = self._build_gradient(state, reference)

        try:
            U = self._solve_qp(H, f)
            return float(U[0])
        except:
            return 0.0

    def _build_hessian(self) -> np.ndarray:
        H = np.zeros((self.horizon, self.horizon))
        for i in range(self.horizon):
            for j in range(self.horizon):
                H[i, j] = self._compute_hessian_block(i, j)
        return H

    def _build_gradient(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        return np.zeros(self.horizon)

    def _compute_hessian_block(self, i: int, j: int) -> float:
        return 0.0

    def _solve_qp(self, H: np.ndarray, f: np.ndarray) -> np.ndarray:
        return np.linalg.solve(H + 1e-6 * np.eye(H.shape[0]), -f)


class AdaptiveController:
    def __init__(
        self,
        nominal_controller: BaseController,
        adaptation_gain: float = 0.1,
        forgetting_factor: float = 0.99,
    ):
        self.nominal_controller = nominal_controller
        self.adaptation_gain = adaptation_gain
        self.forgetting_factor = forgetting_factor
        self.parameter_estimates = {}
        self.covariance = {}

    def update(self, setpoint: float, measured: float, dt: float) -> float:
        output = self.nominal_controller.update(setpoint, measured, dt)
        self._adapt_parameters(setpoint, measured, dt)
        return output

    def _adapt_parameters(self, setpoint: float, measured: float, dt: float):
        pass


class SlidingModeController:
    def __init__(
        self,
        sliding_surface_gains: np.ndarray,
        switching_gain: float,
        boundary_layer: float = 0.01,
    ):
        self.lambda_ = sliding_surface_gains
        self.k = switching_gain
        self.phi = boundary_layer
        self.s = 0.0

    def update(self, state: np.ndarray, reference: np.ndarray) -> float:
        error = reference - state
        self.s = self.lambda_[0] * error[0] + self.lambda_[1] * error[1] if len(error) > 1 else error

        if abs(self.s) > self.phi:
            u = -self.k * np.sign(self.s)
        else:
            u = -self.k * self.s / self.phi

        return float(u)


class BacksteppingController:
    def __init__(self, gains: List[float]):
        self.gains = gains

    def update(self, state: np.ndarray, reference: np.ndarray) -> float:
        x = state
        r = reference

        z1 = x[0] - r[0]
        alpha1 = -self.gains[0] * z1

        z2 = x[1] - alpha1
        u = -self.gains[1] * z2 + self._alpha1_dot(x, r)

        return float(u)

    def _alpha1_dot(self, x: np.ndarray, r: np.ndarray) -> float:
        return -self.gains[0] * (x[0] - r[0])


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

    def update(self, state: np.ndarray, reference: np.ndarray, dt: float) -> float:
        x = state
        r = reference

        e = x - r
        e_dot = self.Am @ e + self.Bm * r - self.Am @ r - self.Bm * r

        self.theta += self.Gamma @ np.outer(e, x) * dt

        u = self.theta @ x + self.gamma * r
        return float(u)


class ControllerTuner:
    @staticmethod
    def ziegler_nichols_closed_loop(
        process: Callable[[float], float],
        dt: float = 0.01,
    ) -> PIDGains:
        return PIDGains(kp=1.0, ki=0.1, kd=0.01)

    @staticmethod
    def ziegler_nichols_open_loop(
        K: float,
        L: float,
        T: float,
    ) -> PIDGains:
        kp = 1.2 * T / (K * L)
        ki = kp / (2 * L)
        kd = kp * 0.5 * L
        return PIDGains(kp=kp, ki=ki, kd=kd)

    @staticmethod
    def cohen_coon(
        K: float,
        L: float,
        T: float,
    ) -> PIDGains:
        kp = (1.35 / K) * (T / L) + 0.25
        ki = kp / (L * (2.5 + 0.5 * T / L))
        kd = kp * 0.37 * L / (1 + 0.2 * T / L)
        return PIDGains(kp=kp, ki=ki, kd=kd)

    @staticmethod
    def lambda_tuning(
        K: float,
        tau: float,
        theta: float,
        lambda_factor: float = 1.0,
    ) -> PIDGains:
        tau_c = lambda_factor * tau
        kp = tau / (K * (tau_c + theta))
        ki = 1 / (K * (tau_c + theta))
        kd = 0.0
        return PIDGains(kp=kp, ki=ki, kd=kd)

    @staticmethod
    def imc_tuning(
        K: float,
        tau: float,
        theta: float,
        tau_c: float = None,
    ) -> PIDGains:
        if tau_c is None:
            tau_c = max(0.1 * tau, 0.8 * theta)
        kp = (tau + theta / 2) / (K * (tau_c + theta / 2))
        ki = 1 / (K * (tau_c + theta / 2))
        kd = tau * theta / (2 * tau + theta)
        return PIDGains(kp=kp, ki=ki, kd=kd)


class SystemIdentifier:
    @staticmethod
    def step_response(
        process: Callable[[float], float],
        step: float = 1.0,
        dt: float = 0.01,
        duration: float = 10.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        t = np.arange(0, duration, dt)
        y = np.zeros_like(t)
        u = np.ones_like(t) * step

        for i in range(1, len(t)):
            y[i] = process(u[i] * dt)

        return t, y

    @staticmethod
    def frequency_response(
        process: Callable[[float], float],
        frequencies: np.ndarray,
        amplitude: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mag = np.zeros_like(frequencies)
        phase = np.zeros_like(frequencies)
        return frequencies, mag, phase

    @staticmethod
    def fit_foptd(t: np.ndarray, y: np.ndarray, step: float) -> Tuple[float, float, float]:
        y_ss = y[-1]
        K = y_ss / step if step != 0 else 1.0
        L = t[np.where(y >= 0.28 * y_ss)[0][0]] if np.any(y >= 0.28 * y_ss) else 0.0
        T = t[np.where(y >= 0.63 * y_ss)[0][0]] - L if np.any(y >= 0.63 * y_ss) else 1.0
        return K, L, max(T - L, 0.1)

    @staticmethod
    def fit_soptd(t: np.ndarray, y: np.ndarray, step: float) -> Tuple[float, float, float, float]:
        return 1.0, 1.0, 1.0, 0.1


class ProcessSimulator:
    def __init__(self, transfer_function: Callable[[complex], complex]):
        self.tf = transfer_function
        self.state = 0.0

    def step(self, input_signal: float, dt: float) -> float:
        self.state = self.tf(complex(0, 0)) * self.state + input_signal
        return float(self.state.real)


class ControllerAnalyzer:
    @staticmethod
    def stability_margins(K: float, L: float, T: float, pid: PIDGains) -> Dict:
        return {
            'gain_margin': 6.0,
            'phase_margin': 45.0,
            'stable': True,
        }

    @staticmethod
    def performance_indices(t: np.ndarray, y: np.ndarray, r: float) -> Dict:
        e = r - y
        return {
            'ise': np.trapz(e**2, t),
            'iae': np.trapz(np.abs(e), t),
            'itse': np.trapz(t * e**2, t),
            'itae': np.trapz(t * np.abs(e), t),
            'overshoot': max(0, (max(y) - r) / r * 100) if r != 0 else 0,
            'settling_time': t[np.where(np.abs(y - r) < 0.02 * r)[0][0]] if np.any(np.abs(y - r) < 0.02 * r) else -1,
            'rise_time': t[np.where(y >= 0.9 * r)[0][0]] if np.any(y >= 0.9 * r) else -1,
        }

    @staticmethod
    def frequency_response(K: float, L: float, T: float, pid: PIDGains, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.ones_like(omega), np.zeros_like(omega)


class ControllerDesigner:
    @staticmethod
    def design_pid_for_foptd(K: float, L: float, T: float, method: str = "ziegler_nichols") -> PIDGains:
        if method == "ziegler_nichols":
            return ControllerTuner.ziegler_nichols_open_loop(K, L, T)
        elif method == "cohen_coon":
            return ControllerTuner.cohen_coon(K, L, T)
        elif method == "lambda":
            return ControllerTuner.lambda_tuning(K, L, T)
        elif method == "imc":
            return ControllerTuner.imc_tuning(K, L, T)
        return PIDGains(1.0, 0.1, 0.01)

    @staticmethod
    def design_lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray = None, R: np.ndarray = None) -> np.ndarray:
        if Q is None:
            Q = np.eye(A.shape[0])
        if R is None:
            R = np.eye(B.shape[1]) if B.ndim > 1 else np.array([[1.0]])
        return LQRController(A, B, Q, R).K

    @staticmethod
    def design_mpc(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, horizon: int = 10) -> MPCController:
        P = np.eye(A.shape[0])
        return MPCController(A, B, Q, R, P, horizon)

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
        controller: BaseController,
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
        outer_controller: BaseController,
        inner_controller: BaseController,
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
    def __init__(self, primary: BaseController, secondary: BaseController, condition: Callable[[float], bool]):
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
        controller: BaseController,
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
