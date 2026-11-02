from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
import random
from collections import defaultdict


class GraphType(Enum):
    POSE_GRAPH = "pose_graph"
    FACTOR_GRAPH = "factor_graph"
    FEATURE_GRAPH = "feature_graph"
    SEMANTIC_GRAPH = "semantic_graph"
    TEMPORAL_GRAPH = "temporal_graph"
    TOPOLOGICAL_GRAPH = "topological_graph"


class FactorType(Enum):
    PRIOR = "prior"
    ODOMETRY = "odometry"
    LOOP_CLOSURE = "loop_closure"
    LANDMARK = "landmark"
    GPS = "gps"
    IMU = "imu"
    VISUAL = "visual"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class Factor:
    factor_type: FactorType
    keys: List[int]
    measurement: np.ndarray
    information: np.ndarray
    loss_function: Optional[Callable] = None

    def error(self, values: Dict[int, np.ndarray]) -> np.ndarray:
        if self.factor_type == FactorType.PRIOR:
            return values[self.keys[0]] - self.measurement
        elif self.factor_type == FactorType.ODOMETRY:
            return np.linalg.inv(values[self.keys[0]]) @ values[self.keys[1]] - self.measurement
        return np.zeros_like(self.measurement)


@dataclass(frozen=True, slots=True)
class Variable:
    key: int
    dim: int
    value: np.ndarray
    fixed: bool = False


class FactorGraph:
    def __init__(self):
        self.factors: List[Factor] = []
        self.variables: Dict[int, Variable] = {}
        self.variable_counter = 0

    def add_variable(self, dim: int, initial_value: Optional[np.ndarray] = None, fixed: bool = False) -> int:
        key = self.variable_counter
        self.variable_counter += 1

        value = initial_value if initial_value is not None else np.zeros(dim)
        self.variables[key] = Variable(key, dim, value, fixed)
        return key

    def add_factor(self, factor: Factor):
        self.factors.append(factor)

    def add_prior_factor(self, key: int, measurement: np.ndarray, information: np.ndarray):
        factor = Factor(FactorType.PRIOR, [key], measurement, information)
        self.add_factor(factor)

    def add_odometry_factor(self, key1: int, key2: int, measurement: np.ndarray, information: np.ndarray):
        factor = Factor(FactorType.ODOMETRY, [key1, key2], measurement, information)
        self.add_factor(factor)

    def add_loop_closure_factor(self, key1: int, key2: int, measurement: np.ndarray, information: np.ndarray):
        factor = Factor(FactorType.LOOP_CLOSURE, [key1, key2], measurement, information)
        self.add_factor(factor)

    def add_landmark_factor(self, pose_key: int, landmark_key: int, measurement: np.ndarray, information: np.ndarray):
        factor = Factor(FactorType.LANDMARK, [pose_key, landmark_key], measurement, information)
        self.add_factor(factor)

    def get_values(self) -> Dict[int, np.ndarray]:
        return {key: var.value for key, var in self.variables.items()}

    def set_values(self, values: Dict[int, np.ndarray]):
        for key, value in values.items():
            if key in self.variables and not self.variables[key].fixed:
                self.variables[key] = Variable(key, self.variables[key].dim, value, self.variables[key].fixed)


class FactorGraphOptimizer:
    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        lambda_init: float = 1e-3,
        lambda_factor: float = 10.0
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.lambda_init = lambda_init
        self.lambda_factor = lambda_factor

    def optimize(self, graph: FactorGraph) -> Dict[int, np.ndarray]:
        variables = list(graph.variables.keys())
        n_vars = len(variables)

        key_to_idx = {key: i for i, key in enumerate(variables)}

        dims = [graph.variables[key].dim for key in variables]
        total_dim = sum(dims)

        x = np.zeros(total_dim)
        offset = 0
        for i, key in enumerate(variables):
            x[offset:offset + dims[i]] = graph.variables[key].value
            offset += dims[i]

        for iteration in range(self.max_iterations):
            A = np.zeros((total_dim, total_dim))
            b = np.zeros(total_dim)

            for factor in graph.factors:
                if all(k in key_to_idx for k in factor.keys):
                    self._add_factor_jacobian(A, b, factor, graph, key_to_idx, dims)

            lam = self.lambda_init
            for _ in range(10):
                A_reg = A + lam * np.eye(total_dim)
                try:
                    dx = np.linalg.solve(A_reg, b)
                    break
                except np.linalg.LinAlgError:
                    lam *= self.lambda_factor
            else:
                dx = np.linalg.lstsq(A + lam * np.eye(total_dim), b, rcond=None)[0]

            offset = 0
            for i, key in enumerate(variables):
                if not graph.variables[key].fixed:
                    graph.variables[key] = Variable(
                        key, dims[i],
                        graph.variables[key].value + dx[offset:offset + dims[i]],
                        graph.variables[key].fixed
                    )
                offset += dims[i]

            if np.linalg.norm(dx) < self.tolerance:
                break

        return graph.get_values()

    def _add_factor_jacobian(
        self,
        A: np.ndarray,
        b: np.ndarray,
        factor: Factor,
        graph: FactorGraph,
        key_to_idx: Dict[int, int],
        dims: List[int]
    ):
        offsets = {}
        offset = 0
        for key in sorted(key_to_idx.keys()):
            idx = key_to_idx[key]
            offsets[key] = offset
            offset += dims[idx]

        if factor.factor_type == FactorType.PRIOR:
            key = factor.keys[0]
            offset = offsets[key]
            dim = graph.variables[key].dim

            error = graph.variables[key].value - factor.measurement
            J = np.eye(dim)

            A[offset:offset+dim, offset:offset+dim] += J.T @ factor.information @ J
            b[offset:offset+dim] -= J.T @ factor.information @ error

        elif factor.factor_type == FactorType.ODOMETRY:
            key1, key2 = factor.keys
            offset1 = offsets[key1]
            offset2 = offsets[key2]
            dim1 = graph.variables[key1].dim
            dim2 = graph.variables[key2].dim

            error = np.linalg.inv(graph.variables[key1].value) @ graph.variables[key2].value - factor.measurement
            J1 = -np.eye(dim1)
            J2 = np.eye(dim2)

            A[offset1:offset1+dim1, offset1:offset1+dim1] += J1.T @ factor.information @ J1
            A[offset1:offset1+dim1, offset2:offset2+dim2] += J1.T @ factor.information @ J2
            A[offset2:offset2+dim2, offset1:offset1+dim1] += J2.T @ factor.information @ J1
            A[offset2:offset2+dim2, offset2:offset2+dim2] += J2.T @ factor.information @ J2

            b[offset1:offset1+dim1] -= J1.T @ factor.information @ error
            b[offset2:offset2+dim2] -= J2.T @ factor.information @ error


class ISAM2:
    def __init__(
        self,
        relinearize_threshold: float = 0.1,
        relinearize_skip: int = 10
    ):
        self.relinearize_threshold = relinearize_threshold
        self.relinearize_skip = relinearize_skip
        self.graph = FactorGraph()
        self.counter = 0
        self.last_relinearize = 0

    def add_factor(self, factor: Factor):
        self.graph.add_factor(factor)

    def add_variable(self, dim: int, initial_value: Optional[np.ndarray] = None, fixed: bool = False) -> int:
        return self.graph.add_variable(dim, initial_value, fixed)

    def update(self) -> Dict[int, np.ndarray]:
        self.counter += 1
        if self.counter - self.last_relinearize >= self.relinearize_skip:
            self.last_relinearize = self.counter

        optimizer = FactorGraphOptimizer()
        return optimizer.optimize(self.graph)

    def get_estimate(self, key: int) -> Optional[np.ndarray]:
        return self.graph.variables.get(key, Variable(0, 0, np.zeros(0))).value


class IncrementalSmoother:
    def __init__(self, isam2: ISAM2):
        self.isam2 = isam2

    def add_pose(self, pose: np.ndarray, information: np.ndarray, fixed: bool = False) -> int:
        key = self.isam2.add_variable(6, pose, fixed)

        if hasattr(self, 'last_key') and self.last_key is not None:
            odom = np.linalg.inv(self.isam2.graph.variables[self.last_key].value) @ pose
            self.isam2.add_factor(Factor(
                FactorType.ODOMETRY, [self.last_key, key], odom, information
            ))

        self.last_key = key
        return key

    def add_landmark(self, position: np.ndarray, information: np.ndarray, fixed: bool = False) -> int:
        key = self.isam2.add_variable(3, position, fixed)
        return key

    def add_landmark_factor(self, pose_key: int, landmark_key: int, measurement: np.ndarray, information: np.ndarray):
        self.isam2.add_factor(Factor(
            FactorType.LANDMARK, [pose_key, landmark_key], measurement, information
        ))

    def add_loop_closure(self, key1: int, key2: int, measurement: np.ndarray, information: np.ndarray):
        self.isam2.add_factor(Factor(
            FactorType.LOOP_CLOSURE, [key1, key2], measurement, information
        ))

    def update(self):
        self.isam2.update()


class GraphSLAM:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.graph = FactorGraph()
        self.optimizer = FactorGraphOptimizer()
        self.poses: Dict[int, np.ndarray] = {}
        self.landmarks: Dict[int, np.ndarray] = {}
        self.current_pose_id = 0
        self.current_landmark_id = 0

    def add_odometry(self, odom: np.ndarray, information: np.ndarray = None):
        information = information or np.eye(6) * 100

        if self.current_pose_id == 0:
            self.graph.add_variable(6, np.eye(4), fixed=True)
        else:
            self.graph.add_variable(6, self.poses[self.current_pose_id])

        self.current_pose_id += 1
        new_id = self.current_pose_id

        if new_id > 1:
            self.graph.add_odometry_factor(new_id - 1, new_id, odom, information)

        self.poses[new_id] = self.poses[new_id - 1] @ odom

    def add_landmark(self, position: np.ndarray, information: np.ndarray = None):
        information = information or np.eye(3) * 100
        self.current_landmark_id += 1
        lm_id = self.current_landmark_id

        self.graph.add_variable(3, position)
        self.landmarks[lm_id] = position

        if self.current_pose_id > 0:
            measurement = np.linalg.inv(self.poses[self.current_pose_id]) @ np.hstack([position, 1])[:3]
            self.graph.add_landmark_factor(self.current_pose_id, lm_id, measurement, information)

    def add_loop_closure(self, pose_id1: int, pose_id2: int, measurement: np.ndarray, information: np.ndarray):
        self.graph.add_loop_closure_factor(pose_id1, pose_id2, measurement, information)

    def optimize(self) -> Dict[int, np.ndarray]:
        optimized = self.optimizer.optimize(self.graph)
        for key, value in optimized.items():
            if key <= self.current_pose_id:
                self.poses[key] = value
            else:
                self.landmarks[key] = value
        return optimized

    def get_trajectory(self) -> List[np.ndarray]:
        return [self.poses[i] for i in range(1, self.current_pose_id + 1)]


class FactorGraphVisualizer:
    @staticmethod
    def draw_factor_graph(
        graph: FactorGraph,
        image_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if not graph.variables:
            return vis

        positions = {}
        for key, var in graph.variables.items():
            if var.dim >= 2:
                positions[key] = var.value[:2]
            elif var.dim >= 3:
                positions[key] = var.value[:2]

        if not positions:
            return vis

        pos_array = np.array(list(positions.values()))
        min_coords = np.min(pos_array, axis=0)
        max_coords = np.max(pos_array, axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1

        scale = min(image_size[0], image_size[1]) * 0.8 / max(range_coords)

        node_positions = {}
        for key, pos in positions.items():
            x = int((pos[0] - min_coords[0]) * scale + image_size[0] * 0.1)
            y = int(image_size[1] - (pos[1] - min_coords[1]) * scale - image_size[1] * 0.1)
            node_positions[key] = (x, y)

        for factor in graph.factors:
            if len(factor.keys) == 2:
                k1, k2 = factor.keys
                if k1 in node_positions and k2 in node_positions:
                    color = {
                        FactorType.PRIOR: (255, 0, 0),
                        FactorType.ODOMETRY: (0, 255, 0),
                        FactorType.LOOP_CLOSURE: (0, 0, 255),
                        FactorType.LANDMARK: (255, 255, 0),
                        FactorType.GPS: (255, 0, 255),
                        FactorType.IMU: (0, 255, 255)
                    }.get(factor.factor_type, (255, 255, 255))

                    cv2.line(vis, node_positions[k1], node_positions[k2], color, 2)

        for key, pos in node_positions.items():
            color = (0, 0, 255) if graph.variables[key].fixed else (255, 255, 255)
            cv2.circle(vis, pos, 5, color, -1)
            cv2.putText(vis, str(key), (pos[0] + 8, pos[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return vis


class FactorGraphMetrics:
    @staticmethod
    def compute_chi2_error(graph: FactorGraph, values: Dict[int, np.ndarray]) -> float:
        total_error = 0.0
        for factor in graph.factors:
            if all(k in values for k in factor.keys):
                error = factor.error(values)
                total_error += error.T @ factor.information @ error
        return float(total_error)

    @staticmethod
    def compute_rmse(graph: FactorGraph, values: Dict[int, np.ndarray]) -> float:
        chi2 = FactorGraphMetrics.compute_chi2_error(graph, values)
        n_factors = len(graph.factors)
        return math.sqrt(chi2 / max(n_factors, 1))

    @staticmethod
    def check_consistency(graph: FactorGraph, values: Dict[int, np.ndarray], confidence: float = 0.95) -> Tuple[bool, float, float]:
        from scipy.stats import chi2
        chi2_val = FactorGraphMetrics.compute_chi2_error(graph, values)
        dof = sum(factor.information.shape[0] for factor in graph.factors)
        lower = chi2.ppf((1 - confidence) / 2, dof)
        upper = chi2.ppf((1 + confidence) / 2, dof)
        return lower <= chi2_val <= upper, lower, upper


class FactorGraphExporter:
    @staticmethod
    def export_g2o(graph: FactorGraph, path: str):
        with open(path, 'w') as f:
            for key, var in graph.variables.items():
                if var.dim == 6:
                    R = var.value[:3, :3] if var.value.shape == (4, 4) else np.eye(3)
                    t = var.value[:3, 3] if var.value.shape == (4, 4) else np.zeros(3)
                    q = cv2.Rodrigues(R)[0].flatten()
                    if np.linalg.norm(q) > 0:
                        q = np.array([math.cos(np.linalg.norm(q)/2), *(q/np.linalg.norm(q)*math.sin(np.linalg.norm(q)/2))])
                    else:
                        q = np.array([1, 0, 0, 0])
                    f.write(f"VERTEX_SE3:QUAT {key} {t[0]} {t[1]} {t[2]} {q[1]} {q[2]} {q[3]} {q[0]}\n")

            for factor in graph.factors:
                if factor.factor_type == FactorType.ODOMETRY:
                    k1, k2 = factor.keys
                    R = factor.measurement[:3, :3] if factor.measurement.shape == (4, 4) else np.eye(3)
                    t = factor.measurement[:3, 3] if factor.measurement.shape == (4, 4) else np.zeros(3)
                    q = cv2.Rodrigues(R)[0].flatten()
                    if np.linalg.norm(q) > 0:
                        q = np.array([math.cos(np.linalg.norm(q)/2), *(q/np.linalg.norm(q)*math.sin(np.linalg.norm(q)/2))])
                    else:
                        q = np.array([1, 0, 0, 0])

                    info = factor.information.flatten()
                    f.write(f"EDGE_SE3:QUAT {k1} {k2} {t[0]} {t[1]} {t[2]} {q[1]} {q[2]} {q[3]} {q[0]} ")
                    f.write(" ".join(map(str, info)) + "\n")


import cv2
from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet