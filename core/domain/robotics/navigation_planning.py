from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math
import time


@dataclass(frozen=True, slots=True)
class Obstacle:
    position: np.ndarray
    shape: str
    dimensions: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    static: bool = True


@dataclass(frozen=True, slots=True)
class PlanningConfig:
    max_iterations: int = 5000
    step_size: float = 0.5
    goal_bias: float = 0.1
    collision_check_resolution: float = 0.1
    goal_tolerance: float = 0.5
    bounds: Tuple[np.ndarray, np.ndarray] = field(default_factory=lambda: (np.array([-10, -10, -10]), np.array([10, 10, 10])))


@dataclass(frozen=True, slots=True)
class Path:
    waypoints: List[np.ndarray]
    cost: float
    length: float
    duration: float
    smooth: bool = False


class Planner(ABC):
    @abstractmethod
    def plan(self, start: np.ndarray, goal: np.ndarray, obstacles: List[Callable] = None) -> Optional[List[np.ndarray]]:
        pass


class AStarPlanner:
    def __init__(self, grid_resolution: float = 0.5, heuristic_weight: float = 1.0):
        self.grid_resolution = grid_resolution
        self.heuristic_weight = heuristic_weight

    def plan(self, start: np.ndarray, goal: np.ndarray, obstacles: List[Callable] = None) -> Optional[List[np.ndarray]]:
        start_cell = tuple((start / 0.5).astype(int))
        goal_cell = tuple((goal / 0.5).astype(int))

        open_set = [(0, start_cell)]
        came_from = {}
        g_score = {start_cell: 0}
        f_score = {start_cell: self._heuristic(start_cell, goal_cell)}

        while open_set:
            current = min(open_set, key=lambda x: f_score[x])[1]
            if current == goal_cell:
                return self._reconstruct_path(came_from, current)

            for neighbor in self._neighbors(current):
                if self._collision(neighbor):
                    continue
                tentative_g = g_score[current] + self._distance(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic_weight * self._heuristic(neighbor, goal_cell)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def _heuristic(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def _neighbors(self, cell):
        x, y, z = cell
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbors.append((x+dx, y+dy, z+dz))
        return neighbors

    def _distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def _collision(self, cell):
        return False

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return [np.array(p) for p in path]


class RRTPlanner:
    def __init__(
        self,
        max_iterations: int = 5000,
        step_size: float = 1.0,
        goal_bias: float = 0.1,
        bounds: Tuple[np.ndarray, np.ndarray] = None,
        collision_checker: Callable = None,
    ):
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.bounds = bounds or (np.array([-10, -10, -10]), np.array([10, 10, 10]))
        self.collision_checker = collision_checker or (lambda x: False)

    def plan(self, start: np.ndarray, goal: np.ndarray, obstacles: List[Callable] = None) -> Optional[List[np.ndarray]]:
        self.tree = [start]
        self.parents = {0: -1}

        for _ in range(self.max_iterations):
            if random.random() < self.goal_bias:
                sample = goal
            else:
                sample = np.random.uniform(self.bounds[0], self.bounds[1])

            if self.collision_checker(sample):
                continue

            nearest_idx = self._nearest(sample)
            nearest = self.tree[nearest_idx]
            new_node = self._steer(nearest, sample)

            if not self._collision_free(nearest, new_node):
                continue

            self.tree.append(new_node)
            self.parents[len(self.tree) - 1] = nearest_idx

            if np.linalg.norm(new_node - goal) < 0.5:
                return self._reconstruct_path(len(self.tree) - 1)

        return None

    def _nearest(self, point: np.ndarray) -> int:
        return int(np.argmin([np.linalg.norm(p - point) for p in self.tree]))

    def _steer(self, from_point: np.ndarray, to_point: np.ndarray) -> np.ndarray:
        direction = to_point - from_point
        dist = np.linalg.norm(direction)
        if dist <= self.step_size:
            return to_point
        return from_point + direction / dist * self.step_size

    def _collision_free(self, a: np.ndarray, b: np.ndarray) -> bool:
        steps = max(1, int(np.linalg.norm(b - a) / 0.1))
        for i in range(steps + 1):
            t = i / steps
            point = a + (b - a) * t
            if self.collision_checker(point):
                return False
        return True

    def _reconstruct_path(self, goal_idx: int) -> List[np.ndarray]:
        path = []
        idx = goal_idx
        while idx != -1:
            path.append(self.tree[idx])
            idx = self.parents.get(idx, -1)
        path.reverse()
        return path


class RRTStarPlanner(RRTPlanner):
    def __init__(
        self,
        max_iterations: int = 5000,
        step_size: float = 1.0,
        goal_bias: float = 0.1,
        rewire_radius: float = 5.0,
        bounds: Tuple[np.ndarray, np.ndarray] = None,
    ):
        super().__init__(max_iterations, step_size)
        self.rewire_radius = rewire_radius
        self.costs: Dict[int, float] = {0: 0.0}

    def plan(self, start: np.ndarray, goal: np.ndarray, obstacles: List[Callable] = None) -> Optional[List[np.ndarray]]:
        self.tree = [start]
        self.parents = {0: -1}
        self.costs = {0: 0.0}

        for _ in range(self.max_iterations):
            if random.random() < self.goal_bias:
                sample = goal
            else:
                sample = np.random.uniform(self.bounds[0], self.bounds[1])

            if self.collision_checker(sample):
                continue

            nearest_idx = self._nearest(sample)
            nearest = self.tree[nearest_idx]
            new_node = self._steer(nearest, sample)

            if not self._collision_free(nearest, new_node):
                continue

            near_indices = self._near(len(self.tree) - 1)
            best_parent = nearest
            min_cost = self.costs[nearest_idx] + np.linalg.norm(new_node - nearest)

            for near_idx in near_indices:
                if self._collision_free(self.tree[near_idx], new_node):
                    cost = self.costs[near_idx] + np.linalg.norm(new_node - self.tree[near_idx])
                    if cost < min_cost:
                        min_cost = cost
                        best_parent = near_idx

            self.tree.append(new_node)
            self.parents[len(self.tree) - 1] = best_parent
            self.costs[len(self.tree) - 1] = min_cost

            for near_idx in near_indices:
                if near_idx == best_parent:
                    continue
                new_cost = min_cost + np.linalg.norm(new_node - self.tree[near_idx])
                if new_cost < self.costs.get(near_idx, float('inf')) and self._collision_free(new_node, self.tree[near_idx]):
                    self.parents[near_idx] = len(self.tree) - 1
                    self.costs[near_idx] = new_cost

            if np.linalg.norm(new_node - goal) < 0.5:
                return self._reconstruct_path(len(self.tree) - 1)

        return None

    def _near(self, node_idx: int) -> List[int]:
        return [i for i, node in enumerate(self.tree) if np.linalg.norm(self.tree[node_idx] - self.tree[i]) <= self.rewire_radius]


class AStarPlanner:
    def __init__(self, grid_resolution: float = 0.5, heuristic_weight: float = 1.0):
        self.grid_resolution = grid_resolution
        self.heuristic_weight = heuristic_weight

    def plan(self, start: np.ndarray, goal: np.ndarray, obstacles: List[Callable] = None) -> Optional[List[np.ndarray]]:
        start_cell = (int(start[0] / 0.5), int(start[1] / 0.5))
        goal_cell = (int(goal[0] / 0.5), int(goal[1] / 0.5))

        open_set = [(0, start_cell)]
        came_from = {}
        g_score = {start_cell: 0}
        f_score = {start_cell: self._heuristic(start_cell, goal_cell)}

        while open_set:
            current = min(f_score, key=f_score.get)
            if current == goal_cell:
                return self._reconstruct_path(came_from, current)

            for neighbor in self._neighbors(current):
                if self._collision(neighbor):
                    continue
                tentative_g = g_score[current] + self._distance(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic_weight * self._heuristic(neighbor, goal_cell)

        return None

    def _heuristic(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def _neighbors(self, cell):
        x, y = cell
        return [(x+dx, y+dy) for dx, dy in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]]

    def _distance(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def _collision(self, cell):
        return False

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


class RRTConnectPlanner(RRTPlanner):
    def __init__(self, max_iterations: int = 5000, step_size: float = 1.0, goal_bias: float = 0.1, bounds: Tuple = None):
        super().__init__(max_iterations, step_size)
        self.tree_a = {}
        self.tree_b = {}
        self.parents_a = {}
        self.parents_b = {}

    def plan(self, start, goal, obstacles=None):
        self.tree_a = {tuple(start): None}
        self.tree_b = {tuple(goal): None}
        self.parents_a = {tuple(start): None}
        self.parents_b = {tuple(goal): None}
        nodes_a = [start]
        nodes_b = [goal]

        for _ in range(self.max_iterations):
            if random.random() < 0.5:
                sample = self._random_config()
                nearest_a = self._nearest(nodes_a, sample)
                new_node = self._steer(nearest_a, sample)
                if self._collision_free(nearest_a, new_node):
                    self.tree_a[tuple(new_node)] = tuple(nearest_a)
                    self.parents_a[tuple(new_node)] = tuple(nearest_a)
                    if self._connect_trees(new_node):
                        return self._reconstruct_path()
            else:
                sample = self._random_config()
                nearest_b = self._nearest(nodes_b, sample)
                new_node = self._steer(nearest_b, sample)
                if self._collision_free(nearest_b, new_node):
                    self.tree_b[tuple(new_node)] = tuple(nearest_b)
                    self.parents_b[tuple(new_node)] = tuple(nearest_b)
                    if self._connect_trees(new_node):
                        return self._reconstruct_path()

        return None

    def _connect_trees(self, node):
        return False

    def _reconstruct_path(self):
        return []


class PRMPlanner:
    def __init__(self, num_samples: int = 1000, connection_radius: float = 5.0):
        self.num_samples = num_samples
        self.connection_radius = connection_radius
        self.graph = {}
        self.nodes = []

    def build(self, bounds: Tuple[np.ndarray, np.ndarray], obstacles: List[Callable]):
        self.nodes = []
        self.graph = {}
        for _ in range(self.num_samples):
            point = np.random.uniform(bounds[0], bounds[1])
            if not self._collision(point):
                self.nodes.append(point)
                self.graph[len(self.nodes) - 1] = []

        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                if np.linalg.norm(self.nodes[i] - self.nodes[j]) < self.connection_radius:
                    if not self._collision(self.nodes[i], self.nodes[j]):
                        self.graph[i].append(j)
                        self.graph[j].append(i)

    def plan(self, start, goal, obstacles=None):
        start_idx = len(self.nodes)
        goal_idx = len(self.nodes) + 1
        self.nodes.append(start)
        self.nodes.append(goal)
        self.graph[start_idx] = []
        self.graph[goal_idx] = []

        for i in range(len(self.nodes) - 2):
            if np.linalg.norm(start - self.nodes[i]) < 5.0:
                if not self._collision(start, self.nodes[i]):
                    self.graph[start_idx].append(i)
                    self.graph[i].append(start_idx)
            if np.linalg.norm(goal - self.nodes[i]) < 5.0:
                if not self._collision(goal, self.nodes[i]):
                    self.graph[goal_idx].append(i)
                    self.graph[i].append(goal_idx)

        return self._a_star(start_idx, goal_idx)

    def _a_star(self, start, goal):
        return []


class DWAPlanner:
    def __init__(
        self,
        max_velocity: float = 1.0,
        max_angular: float = 1.0,
        prediction_time: float = 2.0,
        dt: float = 0.1,
        velocity_resolution: int = 10,
        angular_resolution: int = 20,
    ):
        self.max_velocity = max_velocity
        self.max_angular = max_angular
        self.prediction_time = prediction_time
        self.dt = dt
        self.velocity_resolution = velocity_resolution
        self.angular_resolution = angular_resolution

    def plan(self, current_state, goal, obstacles):
        best_score = -float('inf')
        best_v = 0
        best_w = 0

        for v in np.linspace(0, self.max_velocity, self.velocity_resolution):
            for w in np.linspace(-self.max_angular, self.max_angular, self.angular_resolution):
                if self._collision(current_state, v, w):
                    continue

                score = self._evaluate_trajectory(current_state, v, w, goal)
                if score > best_score:
                    best_score = score
                    best_v = v
                    best_w = w

        return self._simulate_trajectory(current_state, best_v, best_w)

    def _collision(self, state, v, w, obstacles):
        return False

    def _evaluate_trajectory(self, state, v, w, goal, obstacles):
        return 0

    def _simulate_trajectory(self, state, v, w):
        return []


class TrajectoryOptimizer:
    def __init__(self, robot_model, dt=0.01, horizon=50):
        self.robot_model = robot_model
        self.dt = dt
        self.horizon = horizon

    def optimize(self, trajectory, max_iter=100):
        return trajectory


class CHOMPOptimizer:
    def __init__(self, num_iterations=100, step_size=0.1, smoothness_weight=1.0, obstacle_weight=10.0):
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.smoothness_weight = smoothness_weight
        self.obstacle_weight = obstacle_weight

    def optimize(self, trajectory, obstacles):
        return trajectory


class TrajectoryOptimizer:
    def __init__(self, robot_model, dt=0.01, horizon=100):
        self.robot_model = robot_model
        self.dt = dt
        self.horizon = horizon
        self.Q = np.eye(6)
        self.R = np.eye(2)

    def optimize(self, start, goal, x0=None, u0=None):
        return [], []


class PathOptimizer:
    @staticmethod
    def smooth(path, weight_data=0.1, weight_smooth=0.3):
        return path

    @staticmethod
    def simplify(path):
        return path

    @staticmethod
    def _line_of_sight(a, b):
        return True


class PurePursuitController:
    def __init__(self, lookahead_distance=1.0, max_velocity=1.0):
        self.lookahead_distance = lookahead_distance
        self.max_velocity = max_velocity

    def compute(self, current, path):
        return 0.0, 0.0


class StanleyController:
    def __init__(self, k=1.0, max_velocity=1.0):
        self.k = k
        self.max_velocity = max_velocity

    def compute(self, current, path):
        return 0.0, 0.0


class DWAController:
    def __init__(self, max_velocity=1.0, max_angular=1.0, prediction_time=2.0, dt=0.1):
        self.max_velocity = max_velocity
        self.max_angular = max_angular
        self.prediction_time = prediction_time
        self.dt = dt

    def compute(self, state, goal, obstacles):
        return 0, 0


class MPPIController:
    def __init__(self, num_samples=1000, horizon=20, noise_sigma=0.1, lambda_=1.0):
        self.num_samples = num_samples
        self.horizon = horizon
        self.noise_sigma = noise_sigma
        self.lambda_ = lambda_

    def compute(self, state, goal, obstacles):
        return 0, 0


class LQRController:
    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.K = self._compute_gain()

    def _compute_gain(self):
        return np.linalg.inv(self.R) @ self.B.T @ np.linalg.solve(self.A.T, self.Q)

    def compute(self, state, goal):
        return np.zeros(2)


class MPCController:
    def __init__(self, model, Q, R, horizon=20):
        self.model = model
        self.Q = Q
        self.R = R
        self.horizon = horizon

    def solve(self, state, goal):
        return np.zeros(2)


class Controller:
    @abstractmethod
    def compute(self, state, goal):
        pass


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class TrajectoryFollower:
    def __init__(self, controller: Controller, lookahead: float = 1.0):
        self.controller = controller
        self.lookahead = lookahead
        self.path_idx = 0

    def follow(self, path, current_state):
        return 0, 0


class PathFollower:
    def __init__(self, controller: Controller, lookahead: float = 1.0):
        self.controller = controller
        self.lookahead = lookahead

    def follow(self, path, current_state):
        return 0, 0


class PurePursuitController:
    def __init__(self, lookahead=1.0, max_velocity=1.0):
        self.lookahead = lookahead
        self.max_velocity = max_velocity

    def compute(self, state, path):
        return 0, 0


class StanleyController:
    def __init__(self, k=1.0, max_velocity=1.0):
        self.k = k
        self.max_velocity = max_velocity

    def compute(self, state, path):
        return 0, 0


class MPCController:
    def __init__(self, model, Q, R, horizon=10):
        self.model = model
        self.Q = Q
        self.R = R
        self.horizon = horizon

    def solve(self, state, goal):
        return np.zeros(2)


class LQRController:
    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.K = self._compute_gain()

    def _compute_gain(self):
        P = np.linalg.solve(self.A.T @ self.Q @ self.A, self.A.T @ self.Q @ self.B)
        return np.linalg.inv(self.R) @ self.B.T @ P

    def compute(self, state, goal):
        return np.zeros(2)


class MPCController:
    def __init__(self, model, Q, R, horizon=10):
        self.model = model
        self.Q = Q
        self.R = R
        self.horizon = horizon

    def solve(self, state, goal):
        return np.zeros(2)


class TrajectoryTracker:
    def __init__(self, controller: Controller, path: List[np.ndarray]):
        self.controller = controller
        self.path = path
        self.current_idx = 0

    def update(self, state):
        if self.current_idx >= len(self.path) - 1:
            return np.zeros(2)

        target = self.path[self.current_idx]
        return self.controller.compute(self.path[self.current_idx], self.path[-1])


import random
import heapq
