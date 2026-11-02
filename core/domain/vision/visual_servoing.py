from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict


class VisualServoingType(Enum):
    IBVS = "image_based_visual_servoing"
    PBVS = "position_based_visual_servoing"
    HYBRID = "hybrid_visual_servoing"
    PARTITIONED = "partitioned_visual_servoing"


class FeatureType(Enum):
    POINT = "point"
    LINE = "line"
    PLANE = "plane"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    MOMENTS = "moments"
    FOURIER = "fourier"
    HU_MOMENTS = "hu_moments"


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                        [0, self.fy, self.cy],
                        [0, 0, 1]], dtype=np.float32)

    @property
    def inv_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.matrix)


@dataclass(frozen=True, slots=True)
class FeatureError:
    current: np.ndarray
    desired: np.ndarray
    error: np.ndarray
    interaction_matrix: np.ndarray
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class VisualServoingState:
    current_features: np.ndarray
    desired_features: np.ndarray
    camera_pose: np.ndarray
    velocity: np.ndarray
    error_norm: float
    iteration: int
    converged: bool


class InteractionMatrix(ABC):
    @abstractmethod
    def compute(self, features: np.ndarray, depth: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
        pass

    @abstractmethod
    def feature_dim(self) -> int:
        pass


class PointInteractionMatrix(InteractionMatrix):
    def feature_dim(self) -> int:
        return 2

    def compute(self, features: np.ndarray, depth: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
        n_points = features.shape[0] // 2
        L = np.zeros((2 * n_points, 6), dtype=np.float32)

        for i in range(n_points):
            u = features[2*i]
            v = features[2*i + 1]
            z = depth[i] if i < len(depth) else 1.0

            if z <= 0:
                z = 1.0

            fu = intrinsics.fx
            fv = intrinsics.fy

            L[2*i, :] = [-fu/z, 0, u/z, u*v/fu, -(fu**2 + u**2)/fu, v]
            L[2*i+1, :] = [0, -fv/z, v/fv, (fv**2 + v**2)/fv, -u*v/fv, -u]

        return L


class LineInteractionMatrix(InteractionMatrix):
    def feature_dim(self) -> int:
        return 4

    def compute(self, features: np.ndarray, depth: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
        n_lines = features.shape[0] // 4
        L = np.zeros((4 * n_lines, 6), dtype=np.float32)

        for i in range(n_lines):
            rho = features[4*i]
            theta = features[4*i + 1]
            rho_d = features[4*i + 2]
            theta_d = features[4*i + 3]

            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            L[4*i, :] = [cos_t/z, sin_t/z, 0, -rho_d*sin_t/z, rho_d*cos_t/z, 0]
            L[4*i+1, :] = [-sin_t/z, cos_t/z, 0, -rho_d*cos_t/z, -rho_d*sin_t/z, 0]
            L[4*i+2, :] = [0, 0, -1/z, 0, 0, 0]
            L[4*i+3, :] = [0, 0, 0, 0, 0, -1/z]

        return L


class MomentsInteractionMatrix(InteractionMatrix):
    def __init__(self, order: int = 2):
        self.order = order

    def feature_dim(self) -> int:
        return (self.order + 1) * (self.order + 2) // 2 - 1

    def compute(self, features: np.ndarray, depth: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
        n_moments = self.feature_dim()
        L = np.zeros((n_moments, 6), dtype=np.float32)

        for i in range(n_moments):
            p = i // (self.order + 1)
            q = i % (self.order + 1)
            if p + q > self.order:
                continue

            L[i, 0] = -(p + 1) * features[i] / depth[0] if depth[0] > 0 else 0
            L[i, 1] = -(q + 1) * features[i] / depth[0] if depth[0] > 0 else 0
            L[i, 2] = -(p + q + 2) * features[i] / depth[0] if depth[0] > 0 else 0

        return L


class VisualServoingController(ABC):
    def __init__(
        self,
        camera_intrinsics: CameraIntrinsics,
        lambda_gain: float = 0.5,
        max_velocity: float = 1.0,
        convergence_threshold: float = 1e-3,
        max_iterations: int = 1000
    ):
        self.camera_intrinsics = camera_intrinsics
        self.lambda_gain = lambda_gain
        self.max_velocity = max_velocity
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations

    @abstractmethod
    def compute_velocity(self, state: VisualServoingState) -> np.ndarray:
        pass

    @abstractmethod
    def update_desired_features(self, features: np.ndarray):
        pass


class IBVSController(VisualServoingController):
    def __init__(
        self,
        camera_intrinsics: CameraIntrinsics,
        interaction_matrix: InteractionMatrix,
        lambda_gain: float = 0.5,
        max_velocity: float = 1.0,
        convergence_threshold: float = 1e-3,
        max_iterations: int = 1000,
        damping: float = 0.01
    ):
        super().__init__(camera_intrinsics, lambda_gain, max_velocity, convergence_threshold, max_iterations)
        self.interaction_matrix = interaction_matrix
        self.damping = damping
        self.desired_features: Optional[np.ndarray] = None

    def update_desired_features(self, features: np.ndarray):
        self.desired_features = features.copy()

    def compute_velocity(self, state: VisualServoingState) -> np.ndarray:
        if self.desired_features is None:
            return np.zeros(6)

        error = state.current_features - self.desired_features
        depth = self._estimate_depth(state.current_features, state.camera_pose)

        L = self.interaction_matrix.compute(state.current_features, depth, self.camera_intrinsics)

        L_pinv = np.linalg.pinv(L.T @ L + self.damping * np.eye(6)) @ L.T
        velocity = -self.lambda_gain * L_pinv @ error

        velocity_norm = np.linalg.norm(velocity)
        if velocity_norm > self.max_velocity:
            velocity = velocity * self.max_velocity / velocity_norm

        return velocity

    def _estimate_depth(self, features: np.ndarray, pose: np.ndarray) -> np.ndarray:
        n_features = features.shape[0] // self.interaction_matrix.feature_dim()
        return np.ones(n_features) * 1.0


class PBVSController(VisualServoingController):
    def __init__(
        self,
        camera_intrinsics: CameraIntrinsics,
        lambda_gain: float = 0.5,
        max_velocity: float = 1.0,
        convergence_threshold: float = 1e-3,
        max_iterations: int = 1000,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0
    ):
        super().__init__(camera_intrinsics, lambda_gain, max_velocity, convergence_threshold, max_iterations)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_error = np.zeros(6)
        self.prev_error = np.zeros(6)
        self.desired_pose: Optional[np.ndarray] = None

    def update_desired_features(self, features: np.ndarray):
        self.desired_pose = self._features_to_pose(features)

    def _features_to_pose(self, features: np.ndarray) -> np.ndarray:
        pose = np.eye(4)
        if len(features) >= 6:
            pose[:3, 3] = features[:3]
            pose[:3, :3] = cv2.Rodrigues(features[3:6])[0]
        return pose

    def compute_velocity(self, state: VisualServoingState) -> np.ndarray:
        if self.desired_pose is None:
            return np.zeros(6)

        current_pose = state.camera_pose
        error = self._pose_error(current_pose, self.desired_pose)

        self.integral_error += error
        derivative = error - self.prev_error
        self.prev_error = error

        velocity = self.kp * error + self.ki * self.integral_error + self.kd * derivative
        velocity = self.lambda_gain * velocity

        velocity_norm = np.linalg.norm(velocity)
        if velocity_norm > self.max_velocity:
            velocity = velocity * self.max_velocity / velocity_norm

        return velocity

    def _pose_error(self, current: np.ndarray, desired: np.ndarray) -> np.ndarray:
        R_err = desired[:3, :3] @ current[:3, :3].T
        t_err = desired[:3, 3] - current[:3, 3]

        angle_axis = cv2.Rodrigues(R_err)[0].flatten()
        return np.concatenate([t_err, angle_axis])


class HybridVSController(VisualServoingController):
    def __init__(
        self,
        camera_intrinsics: CameraIntrinsics,
        ibvs_controller: IBVSController,
        pbvs_controller: PBVSController,
        alpha: float = 0.5
    ):
        super().__init__(camera_intrinsics)
        self.ibvs_controller = ibvs_controller
        self.pbvs_controller = pbvs_controller
        self.alpha = alpha

    def update_desired_features(self, features: np.ndarray):
        self.ibvs_controller.update_desired_features(features)
        self.pbvs_controller.update_desired_features(features)

    def compute_velocity(self, state: VisualServoingState) -> np.ndarray:
        v_ibvs = self.ibvs_controller.compute_velocity(state)
        v_pbvs = self.pbvs_controller.compute_velocity(state)
        return self.alpha * v_ibvs + (1 - self.alpha) * v_pbvs


class PartitionedVSController(VisualServoingController):
    def __init__(
        self,
        camera_intrinsics: CameraIntrinsics,
        translational_controller: VisualServoingController,
        rotational_controller: VisualServoingController
    ):
        super().__init__(camera_intrinsics)
        self.translational_controller = translational_controller
        self.rotational_controller = rotational_controller

    def update_desired_features(self, features: np.ndarray):
        self.translational_controller.update_desired_features(features[:3])
        self.rotational_controller.update_desired_features(features[3:])

    def compute_velocity(self, state: VisualServoingState) -> np.ndarray:
        v_trans = self.translational_controller.compute_velocity(state)
        v_rot = self.rotational_controller.compute_velocity(state)
        return np.concatenate([v_trans[:3], v_rot[3:]])


class VisualServoingPipeline:
    def __init__(
        self,
        controller: VisualServoingController,
        feature_extractor: Any,
        depth_estimator: Optional[Any] = None
    ):
        self.controller = controller
        self.feature_extractor = feature_extractor
        self.depth_estimator = depth_estimator

        self.state: Optional[VisualServoingState] = None
        self.history: List[VisualServoingState] = []
        self.iteration = 0

    def initialize(self, image: np.ndarray, desired_features: Optional[np.ndarray] = None):
        features = self.feature_extractor.extract(image)
        depth = self.depth_estimator.estimate(image) if self.depth_estimator else None

        if desired_features is None:
            desired_features = features

        self.controller.update_desired_features(desired_features)

        self.state = VisualServoingState(
            current_features=features,
            desired_features=desired_features,
            camera_pose=np.eye(4),
            velocity=np.zeros(6),
            error_norm=np.linalg.norm(features - desired_features),
            iteration=0,
            converged=False
        )
        self.history = [self.state]
        self.iteration = 0

    def step(self, image: np.ndarray, current_pose: np.ndarray) -> Tuple[np.ndarray, bool]:
        features = self.feature_extractor.extract(image)
        depth = self.depth_estimator.estimate(image) if self.depth_estimator else None

        self.state = VisualServoingState(
            current_features=features,
            desired_features=self.state.desired_features,
            camera_pose=current_pose,
            velocity=np.zeros(6),
            error_norm=np.linalg.norm(features - self.state.desired_features),
            iteration=self.iteration,
            converged=False
        )

        velocity = self.controller.compute_velocity(self.state)
        self.state.velocity = velocity
        self.state.converged = self.state.error_norm < self.controller.convergence_threshold

        self.history.append(self.state)
        self.iteration += 1

        return velocity, self.state.converged or self.iteration >= self.controller.max_iterations

    def get_trajectory(self) -> List[np.ndarray]:
        return [s.camera_pose for s in self.history]

    def get_error_history(self) -> List[float]:
        return [s.error_norm for s in self.history]


class VisualServoingSimulator:
    def __init__(
        self,
        camera_intrinsics: CameraIntrinsics,
        controller: VisualServoingController,
        robot_model: Any,
        dt: float = 0.01
    ):
        self.camera_intrinsics = camera_intrinsics
        self.controller = controller
        self.robot_model = robot_model
        self.dt = dt

        self.pose = np.eye(4)
        self.velocity = np.zeros(6)

    def simulate_step(self, target_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image = self.robot_model.render(self.pose, self.camera_intrinsics)

        if self.iteration == 0:
            self.controller.update_desired_features(target_features)

        features = self._extract_features(image)
        depth = self._estimate_depth(image)

        state = VisualServoingState(
            current_features=features,
            desired_features=target_features,
            camera_pose=self.pose,
            velocity=self.velocity,
            error_norm=np.linalg.norm(features - target_features),
            iteration=self.iteration,
            converged=False
        )

        self.velocity = self.controller.compute_velocity(state)
        self.pose = self._integrate_pose(self.pose, self.velocity, self.dt)
        self.iteration += 1

        return self.pose, self.velocity

    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        return np.random.randn(20).astype(np.float32)

    def _estimate_depth(self, image: np.ndarray) -> np.ndarray:
        return np.ones(10)

    def _integrate_pose(self, pose: np.ndarray, velocity: np.ndarray, dt: float) -> np.ndarray:
        v = velocity[:3]
        w = velocity[3:]

        R = pose[:3, :3]
        t = pose[:3, 3]

        t_new = t + R @ v * dt
        R_new = R @ cv2.Rodrigues(w * dt)[0]

        new_pose = np.eye(4)
        new_pose[:3, :3] = R_new
        new_pose[:3, 3] = t_new

        return new_pose


class VisualServoingMetrics:
    @staticmethod
    def compute_convergence_time(history: List[VisualServoingState], threshold: float = 1e-3) -> float:
        for i, state in enumerate(history):
            if state.error_norm < threshold:
                return i
        return len(history)

    @staticmethod
    def compute_steady_state_error(history: List[VisualServoingState], window: int = 10) -> float:
        if len(history) < window:
            return history[-1].error_norm if history else 0.0
        return np.mean([s.error_norm for s in history[-window:]])

    @staticmethod
    def compute_overshoot(history: List[VisualServoingState]) -> float:
        errors = [s.error_norm for s in history]
        min_error = min(errors)
        final_error = errors[-1]
        return max(0, final_error - min_error) / min_error if min_error > 0 else 0.0

    @staticmethod
    def compute_integral_error(history: List[VisualServoingState]) -> float:
        return sum(s.error_norm for s in history)

    @staticmethod
    def compute_control_effort(history: List[VisualServoingState]) -> float:
        return sum(np.linalg.norm(s.velocity) for s in history)


class AdaptiveGainScheduler:
    def __init__(
        self,
        initial_gain: float = 0.5,
        min_gain: float = 0.01,
        max_gain: float = 2.0,
        adaptation_rate: float = 0.1
    ):
        self.gain = initial_gain
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.adaptation_rate = adaptation_rate
        self.prev_error = float('inf')

    def update(self, error: float) -> float:
        if error < self.prev_error:
            self.gain = min(self.gain * (1 + self.adaptation_rate), self.max_gain)
        else:
            self.gain = max(self.gain * (1 - self.adaptation_rate), self.min_gain)
        self.prev_error = error
        return self.gain


class VisualServoingObstacleAvoidance:
    def __init__(self, safety_distance: float = 0.5, influence_distance: float = 1.0):
        self.safety_distance = safety_distance
        self.influence_distance = influence_distance

    def compute_avoidance_velocity(
        self,
        current_pose: np.ndarray,
        obstacles: List[Tuple[np.ndarray, float]],
        goal_velocity: np.ndarray
    ) -> np.ndarray:
        avoidance = np.zeros(6)

        for obs_pos, obs_radius in obstacles:
            rel_pos = obs_pos - current_pose[:3, 3]
            dist = np.linalg.norm(rel_pos)

            if dist < self.influence_distance:
                if dist < self.safety_distance:
                    repulsion = (1.0/self.safety_distance - 1.0/dist) * rel_pos / (dist + 1e-6)
                else:
                    repulsion = (1.0/dist - 1.0/self.influence_distance) * rel_pos / (dist + 1e-6)
                avoidance[:3] += repulsion

        return goal_velocity + avoidance


class VisualServoingPathPlanner:
    def __init__(
        self,
        camera_intrinsics: CameraIntrinsics,
        controller: VisualServoingController,
        max_steps: int = 100
    ):
        self.camera_intrinsics = camera_intrinsics
        self.controller = controller
        self.max_steps = max_steps

    def plan(
        self,
        initial_features: np.ndarray,
        desired_features: np.ndarray,
        initial_pose: np.ndarray
    ) -> List[VisualServoingState]:
        self.controller.update_desired_features(desired_features)

        path = []
        current_features = initial_features.copy()
        current_pose = initial_pose.copy()

        for step in range(self.max_steps):
            state = VisualServoingState(
                current_features=current_features,
                desired_features=desired_features,
                camera_pose=current_pose,
                velocity=np.zeros(6),
                error_norm=np.linalg.norm(current_features - desired_features),
                iteration=step,
                converged=False
            )

            velocity = self.controller.compute_velocity(state)
            state.velocity = velocity
            state.converged = state.error_norm < self.controller.convergence_threshold

            path.append(state)

            if state.converged:
                break

            current_pose = self._integrate_pose(current_pose, velocity, 0.1)
            current_features = self._predict_features(current_features, velocity, 0.1)

        return path

    def _integrate_pose(self, pose: np.ndarray, velocity: np.ndarray, dt: float) -> np.ndarray:
        v = velocity[:3]
        w = velocity[3:]

        R = pose[:3, :3]
        t = pose[:3, 3]

        t_new = t + R @ v * dt
        R_new = R @ cv2.Rodrigues(w * dt)[0]

        new_pose = np.eye(4)
        new_pose[:3, :3] = R_new
        new_pose[:3, 3] = t_new

        return new_pose

    def _predict_features(self, features: np.ndarray, velocity: np.ndarray, dt: float) -> np.ndarray:
        return features


class VisualServoingCalibration:
    @staticmethod
    def calibrate_interaction_matrix(
        features: np.ndarray,
        depths: np.ndarray,
        velocities: np.ndarray,
        feature_changes: np.ndarray
    ) -> np.ndarray:
        n_samples = velocities.shape[0]
        L_est = np.zeros((features.shape[0], 6))

        for i in range(features.shape[0]):
            A = velocities
            b = feature_changes[:, i]
            L_est[i, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        return L_est.T

    @staticmethod
    def estimate_camera_intrinsics(
        image_points: List[np.ndarray],
        world_points: List[np.ndarray]
    ) -> CameraIntrinsics:
        all_img_pts = np.vstack(image_points)
        all_wld_pts = np.vstack(world_points)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            [p.reshape(-1, 1, 3) for p in world_points],
            [p.reshape(-1, 1, 2) for p in image_points],
            (640, 480), None, None
        )

        return CameraIntrinsics(
            fx=mtx[0, 0], fy=mtx[1, 1],
            cx=mtx[0, 2], cy=mtx[1, 2],
            width=640, height=480
        )


class VisualServoingSafetyMonitor:
    def __init__(
        self,
        max_velocity: float = 2.0,
        max_error: float = 100.0,
        max_iterations: int = 1000,
        joint_limits: Optional[List[Tuple[float, float]]] = None
    ):
        self.max_velocity = max_velocity
        self.max_error = max_error
        self.max_iterations = max_iterations
        self.joint_limits = joint_limits

    def check(self, state: VisualServoingState) -> Tuple[bool, str]:
        if np.linalg.norm(state.velocity) > self.max_velocity:
            return False, f"Velocity limit exceeded: {np.linalg.norm(state.velocity):.2f} > {self.max_velocity}"

        if state.error_norm > self.max_error:
            return False, f"Error limit exceeded: {state.error_norm:.2f} > {self.max_error}"

        if state.iteration > self.max_iterations:
            return False, f"Iteration limit exceeded: {state.iteration} > {self.max_iterations}"

        return True, "OK"


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet