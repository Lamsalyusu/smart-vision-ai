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


class MotionModel(Enum):
    HOLONOMIC = "holonomic"
    DIFFERENTIAL_DRIVE = "differential_drive"
    ACKERMANN = "ackermann"
    CAR_LIKE = "car_like"
    BICYCLE = "bicycle"
    QUADROTOR = "quadrotor"
    FIXED_WING = "fixed_wing"


@dataclass(frozen=True, slots=True)
class State:
    x: float
    y: float
    theta: float = 0.0
    v: float = 0.0
    w: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "State":
        return cls(arr[0], arr[1], arr[2] if len(arr) > 2 else 0.0)


@dataclass(frozen=True, slots=True)
class Control:
    v: float
    w: float
    a: float = 0.0
    alpha: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.v, self.w])


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    state: State
    control: Control
    time: float
    cost: float = 0.0


@dataclass(frozen=True, slots=True)
class Trajectory:
    points: List[TrajectoryPoint]
    duration: float
    cost: float = 0.0

    def get_final_state(self) -> State:
        return self.points[-1].state if self.points else State(0, 0, 0)

    def get_final_time(self) -> float:
        return self.points[-1].time if self.points else 0.0


class MotionPlanner(ABC):
    @abstractmethod
    def plan(self, start: State, goal: State, obstacles: List[Callable]) -> Optional[List[State]]:
        pass

    @abstractmethod
    def is_valid(self, state: State, obstacles: List[Callable]) -> bool:
        pass


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
        self.bounds = bounds or (np.array([-10, -10, -np.pi]), np.array([10, 10, np.pi]))
        self.collision_checker = collision_checker or (lambda x: False)

    def plan(self, start: State, goal: State, obstacles: List[Callable] = None) -> Optional[List[State]]:
        self.tree = [start]
        self.parents = {0: -1}

        goal_state = goal

        for _ in range(self.max_iterations):
            if random.random() < self.goal_bias:
                sample = goal
            else:
                sample = State(
                    np.random.uniform(self.bounds[0, 0], self.bounds[1, 0]),
                    np.random.uniform(self.bounds[0, 1], self.bounds[1, 1]),
                    np.random.uniform(self.bounds[0, 2], self.bounds[1, 2]),
                )

            if self._collision(sample):
                continue

            nearest_idx = self._nearest(sample)
            nearest = self.tree[nearest_idx]
            new_node = self._steer(nearest, sample)

            if not self._collision_free(nearest, new_node):
                continue

            self.tree.append(new_node)
            self.parents[len(self.tree) - 1] = nearest_idx

            if self._distance(new_node, goal) < 0.5:
                return self._reconstruct_path(len(self.tree) - 1)

        return None

    def _nearest(self, state: State) -> int:
        return int(np.argmin([self._distance(state, node) for node in self.tree]))

    def _distance(self, a: State, b: State) -> float:
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.theta - b.theta)**2)

    def _steer(self, from_state: State, to_state: State) -> State:
        direction = np.array([to_state.x - from_state.x, to_state.y - from_state.y, to_state.theta - from_state.theta])
        dist = np.linalg.norm([to_state.x - from_state.x, to_state.y - from_state.y])
        if dist <= self.step_size:
            return to_state
        ratio = self.step_size / dist
        return State(
            from_state.x + (to_state.x - from_state.x) * ratio,
            from_state.y + (to_state.y - from_state.y) * ratio,
            from_state.theta + (to_state.theta - from_state.theta) * ratio,
        )

    def _collision_free(self, a: State, b: State) -> bool:
        steps = max(1, int(self._distance(a, b) / 0.1))
        for i in range(steps + 1):
            t = i / steps
            x = a.x + (b.x - a.x) * t
            y = a.y + (b.y - a.y) * t
            theta = a.theta + (b.theta - a.theta) * t
            state = State(x, y, t * b.theta + (1-t) * a.theta)
            if self._collision(State(x, y, theta)):
                return False
        return True

    def _collision(self, state: State) -> bool:
        return False

    def _reconstruct_path(self, goal_idx: int) -> List[State]:
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
        super().__init__(max_iterations, step_size, goal_bias)
        self.rewire_radius = rewire_radius
        self.costs: Dict[int, float] = {0: 0.0}

    def plan(self, start: State, goal: State, obstacles: List[Callable] = None) -> Optional[List[State]]:
        self.tree = [start]
        self.parents = {0: -1}
        self.costs = {0: 0.0}

        for _ in range(self.max_iterations):
            if random.random() < self.goal_bias:
                sample = goal
            else:
                sample = State(
                    np.random.uniform(self.bounds[0, 0], self.bounds[1, 0]),
                    np.random.uniform(self.bounds[0, 1], self.bounds[1, 1]),
                    np.random.uniform(self.bounds[0, 2], self.bounds[1, 2]),
                )

            if self._collision(sample):
                continue

            nearest_idx = self._nearest(sample)
            nearest = self.tree[nearest_idx]
            new_node = self._steer(self.tree[nearest_idx], sample)

            if not self._collision_free(self.tree[nearest_idx], new_node):
                continue

            near_indices = self._near(len(self.tree))
            best_parent = nearest
            min_cost = self.costs[nearest] + self._distance(self.tree[nearest], new_node)

            for near_idx in near_indices:
                if self._collision_free(self.tree[near_idx], new_node):
                    cost = self.costs[near_idx] + self._distance(self.tree[near_idx], new_node)
                    if cost < min_cost:
                        min_cost = cost
                        best_parent = near_idx

            self.tree.append(new_node)
            self.parents[len(self.tree) - 1] = best_parent
            self.costs[len(self.tree) - 1] = min_cost

            for near_idx in near_indices:
                if near_idx == best_parent:
                    continue
                new_cost = min_cost + self._distance(new_node, self.tree[near_idx])
                if new_cost < self.costs.get(near_idx, float('inf')) and self._collision_free(new_node, self.tree[near_idx]):
                    self.parents[near_idx] = len(self.tree) - 1
                    self.costs[near_idx] = new_cost

            if self._distance(new_node, goal) < 0.5:
                return self._reconstruct_path(len(self.tree) - 1)

        return None

    def _near(self, node_idx: int) -> List[int]:
        return [i for i, node in enumerate(self.tree) if self._distance(node, self.tree[node_idx]) <= self.rewire_radius]


class AStarPlanner:
    def __init__(self, grid_resolution: float = 0.5, heuristic_weight: float = 1.0):
        self.grid_resolution = grid_resolution
        self.heuristic_weight = heuristic_weight

    def plan(self, start: State, goal: State, obstacles: List[Callable] = None) -> Optional[List[State]]:
        start_cell = (int(start.x), int(start.y))
        goal_cell = (int(goal.x), int(goal.y))

        open_set = [(0, start_cell)]
        came_from = {}
        g_score = {start_cell: 0}
        f_score = {start_cell: self._heuristic(start_cell, goal_cell)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_cell:
                return self._reconstruct_path(came_from, current)

            for neighbor in self._neighbors(current):
                if self._is_obstacle(neighbor):
                    continue

                tentative_g = g_score[current] + self._distance(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic_weight * self._heuristic(neighbor, goal_cell)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

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
        return [State(x, y) for x, y in path]


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

    def plan(self, current_state: State, goal: State, obstacles: List[Callable]) -> List[State]:
        best_score = -float('inf')
        best_v = 0
        best_w = 0

        for v in np.linspace(0, self.max_velocity, self.velocity_resolution):
            for w in np.linspace(-self.max_angular, self.max_angular, self.angular_resolution):
                if self._collision(current_state, v, w, obstacles):
                    continue

                score = self._evaluate_trajectory(current_state, v, w, goal, obstacles)
                if score > best_score:
                    best_score = score
                    best_v = v
                    best_w = w

        return self._simulate_trajectory(current_state, best_v, best_w)

    def _collision(self, state: State, v: float, w: float, obstacles: List[Callable]) -> bool:
        return False

    def _collision_free(self, a: State, b: State) -> bool:
        return True

    def _evaluate_trajectory(self, state: State, v: float, w: float, goal: State, obstacles: List[Callable]) -> float:
        final_x = state.x + v * math.cos(state.theta) * self.prediction_time
        final_y = state.y + v * math.sin(state.theta) * self.prediction_time
        dist_to_goal = math.sqrt((final_x - goal.x)**2 + (final_y - goal.y)**2)
        return 1.0 / (dist_to_goal + 1.0)


class TrajectoryOptimizer:
    def __init__(
        self,
        robot_model: Any,
        dt: float = 0.01,
        horizon: int = 50,
    ):
        self.robot_model = robot_model
        self.dt = dt
        self.horizon = horizon

    def optimize(
        self,
        start: State,
        goal: State,
        initial_trajectory: List[State] = None,
        max_iter: int = 100,
    ) -> List[State]:
        if initial_trajectory is None:
            trajectory = self._generate_initial_guess(start, goal)
        else:
            trajectory = initial_trajectory

        for _ in range(max_iter):
            new_trajectory = self._optimize_step(trajectory)
            if self._converged(trajectory, new_trajectory):
                break
            trajectory = new_trajectory

        return trajectory

    def _generate_initial_guess(self, start: State, goal: State) -> List[State]:
        n = 50
        states = []
        for i in range(n + 1):
            t = i / n
            x = start.x + t * (goal.x - start.x)
            y = start.y + t * (goal.y - start.y)
            theta = start.theta + t * (goal.theta - start.theta)
            states.append(State(x, y, theta))
        return states

    def _optimize_step(self, trajectory: List[State]) -> List[State]:
        return trajectory

    def _converged(self, old_traj: List[State], new_traj: List[State]) -> bool:
        return True


class CHOMPOptimizer:
    def __init__(
        self,
        num_iterations: int = 100,
        step_size: float = 0.1,
        smoothness_weight: float = 1.0,
        obstacle_weight: float = 10.0,
    ):
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.smoothness_weight = smoothness_weight
        self.obstacle_weight = obstacle_weight

    def optimize(self, trajectory: List[State], obstacles: List[Callable]) -> List[State]:
        trajectory = np.array([s.to_array() for s in trajectory])

        for _ in range(self.num_iterations):
            gradient = self._compute_gradient(trajectory, obstacles)
            trajectory -= self.step_size * gradient

        return [State(*p) for p in trajectory]

    def _compute_gradient(self, trajectory: np.ndarray, obstacles: List[Callable]) -> np.ndarray:
        n = len(trajectory)
        gradient = np.zeros_like(trajectory)

        for i in range(1, n - 1):
            gradient[i] += self.smoothness_weight * (trajectory[i-1] - 2*trajectory[i] + trajectory[i+1])

            for obs in obstacles:
                dist = self._distance_to_obstacle(trajectory[i], obstacles)
                if dist < 1.0:
                    gradient[i] += self.obstacle_weight * (trajectory[i] - obstacles[0]) / (dist + 1e-6)

        return gradient


class TrajectoryOptimizer:
    def __init__(
        self,
        robot_model,
        dt: float = 0.01,
        horizon: int = 100,
        Q: np.ndarray = None,
        R: np.ndarray = None,
    ):
        self.robot_model = robot_model
        self.dt = dt
        self.horizon = horizon
        self.Q = Q if Q is not None else np.eye(6)
        self.R = R if R is not None else np.eye(2)

    def optimize(
        self,
        start: State,
        goal: State,
        x0: np.ndarray = None,
        u0: np.ndarray = None,
    ) -> Tuple[List[State], List[Control]]:
        n = self.horizon
        N = self.horizon

        if x0 is None:
            x0 = np.zeros((n+1, 3))
            for i in range(n+1):
                alpha = i / n
                x0[i] = [
                    start.x + (goal.x - start.x) * i / n,
                    start.y + (goal.y - start.y) * i / n,
                    start.theta + (goal.theta - start.theta) * i / n,
                ]

        if u0 is None:
            u0 = np.zeros((n, 2))

        for _ in range(50):
            x = x0.copy()
            u = u0.copy()

            for i in range(n):
                u[i] = self.robot_model.inverse_dynamics(x0[i], x0[i+1], self.dt)

            for i in range(n):
                x[i+1] = self.robot_model.forward_dynamics(x[i], u[i], self.dt)

            x0 = x
            u0 = u

        return [State(*x) for x in x0], [Control(*u) for u in u0]


class PathOptimizer:
    @staticmethod
    def smooth(path: List[State], weight_data: float = 0.1, weight_smooth: float = 0.3) -> List[State]:
        if len(path) < 3:
            return path

        new_path = np.array([[s.x, s.y, s.theta] for s in path])
        original = path.copy()

        for _ in range(100):
            for i in range(1, len(path) - 1):
                path[i].x += weight_data * (original[i].x - path[i].x) + weight_smooth * (path[i-1].x + path[i+1].x - 2*path[i].x)
                path[i].y += weight_data * (original[i].y - path[i].y) + weight_smooth * (path[i-1].y + path[i+1].y - 2*path[i].y)
                path[i].theta += weight_data * (original[i].theta - path[i].theta) + weight_smooth * (path[i-1].theta + path[i+1].theta - 2*path[i].theta)

        return path

    @staticmethod
    def simplify(path: List[State]) -> List[State]:
        if len(path) < 3:
            return path

        simplified = [path[0]]
        i = 0

        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if PathOptimizer._line_of_sight(path[i], path[j]):
                    break
                j -= 1
            simplified.append(path[j])
            i = j

        return simplified

    @staticmethod
    def _line_of_sight(a: State, b: State) -> bool:
        return True


class PurePursuitController:
    def __init__(self, lookahead_distance: float = 1.0, max_velocity: float = 1.0):
        self.lookahead_distance = lookahead_distance
        self.max_velocity = max_velocity

    def compute(self, current: State, path: List[State]) -> Tuple[float, float]:
        if not path:
            return 0.0, 0.0

        lookahead_point = self._find_lookahead_point(current, path)
        if lookahead_point is None:
            return 0.0, 0.0

        dx = lookahead_point.x - current.x
        dy = lookahead_point.y - current.y
        target_angle = math.atan2(dy, dx)
        angle_error = target_angle - current.theta

        angular = 2 * angle_error
        linear = self.max_velocity * (1 - abs(angle_error) / math.pi)

        return linear, angular

    def _find_lookahead_point(self, current: State, path: List[State]) -> Optional[State]:
        for point in path:
            dist = math.sqrt((point.x - current.x)**2 + (point.y - current.y)**2)
            if dist >= self.lookahead_distance:
                return point
        return path[-1] if path else None


class StanleyController:
    def __init__(self, k: float = 1.0, max_velocity: float = 1.0):
        self.k = k
        self.max_velocity = max_velocity

    def compute(self, current: State, path: List[State]) -> Tuple[float, float]:
        if not path:
            return 0.0, 0.0

        nearest_idx = self._find_nearest_point(current, path)
        if nearest_idx >= len(path) - 1:
            return 0.0, 0.0

        p1 = path[nearest_idx]
        p2 = path[min(nearest_idx + 1, len(path) - 1)]

        path_angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
        cross_track_error = self._cross_track_error(current, p1, p2)
        heading_error = path_angle - current.theta

        angular = path_angle - current.theta + math.atan2(self.k * cross_track_error, self.max_velocity)
        linear = self.max_velocity

        return linear, angular

    def _find_nearest_point(self, pose, path):
        min_dist = float('inf')
        nearest = 0
        for i, p in enumerate(path):
            d = math.sqrt((p.x - pose.x)**2 + (p.y - pose.y)**2)
            if d < min_dist:
                min_dist = d
                nearest = i
        return nearest

    def _cross_track_error(self, pose, p1, p2):
        x0, y0 = pose.x, pose.y
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y

        return abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


class StanleyController:
    def __init__(self, k: float = 1.0, max_velocity: float = 1.0):
        self.k = k
        self.max_velocity = max_velocity

    def compute(self, current: State, path: List[State]) -> Tuple[float, float]:
        if not path:
            return 0.0, 0.0

        nearest_idx = self._find_nearest_point(current, path)
        if nearest_idx >= len(path) - 1:
            return 0.0, 0.0

        p1 = path[nearest_idx]
        p2 = path[min(nearest_idx + 1, len(path) - 1)]

        path_angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
        cross_track_error = self._cross_track_error(current, p1, p2)
        heading_error = path_angle - current.theta

        angular = path_angle - current.theta + math.atan2(self.k * cross_track_error, self.max_velocity)
        linear = self.max_velocity

        return linear, angular

    def _find_nearest_point(self, pose, path):
        min_dist = float('inf')
        nearest = 0
        for i, p in enumerate(path):
            d = math.sqrt((p.x - pose.x)**2 + (p.y - pose.y)**2)
            if d < min_dist:
                min_dist = d
                nearest = i
        return nearest

    def _cross_track_error(self, pose, p1, p2):
        x0, y0 = pose.x, pose.y
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y

        return abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


class StanleyController:
    def __init__(self, k: float = 1.0, max_velocity: float = 1.0):
        self.k = k
        self.max_velocity = max_velocity

    def compute(self, current: State, path: List[State]) -> Tuple[float, float]:
        if not path:
            return 0.0, 0.0

        nearest_idx = self._find_nearest_point(current, path)
        if nearest_idx >= len(path) - 1:
            return 0.0, 0.0

        p1 = path[nearest_idx]
        p2 = path[min(nearest_idx + 1, len(path) - 1)]

        path_angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
        cross_track_error = self._cross_track_error(current, p1, p2)
        heading_error = path_angle - current.theta

        angular = path_angle - current.theta + math.atan2(self.k * cross_track_error, self.max_velocity)
        linear = self.max_velocity

        return linear, angular

    def _find_nearest_point(self, pose, path):
        min_dist = float('inf')
        nearest = 0
        for i, p in enumerate(path):
            d = math.sqrt((p.x - pose.x)**2 + (p.y - pose.y)**2)
            if d < min_dist:
                min_dist = d
                nearest = i
        return nearest

    def _cross_track_error(self, pose, p1, p2):
        x0, y0 = pose.x, pose.y
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y

        return abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


class StanleyController:
    def __init__(self, k: float = 1.0, max_velocity: float = 1.0):
        self.k = k
        self.max_velocity = max_velocity

    def compute(self, current: State, path: List[State]) -> Tuple[float, float]:
        if not path:
            return 0.0, 0.0

        nearest_idx = self._find_nearest_point(current, path)
        if nearest_idx >= len(path) - 1:
            return 0.0, 0.0

        p1 = path[nearest_idx]
        p2 = path[min(nearest_idx + 1, len(path) - 1)]

        path_angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
        cross_track_error = self._cross_track_error(current, p1, p2)
        heading_error = path_angle - current.theta

        angular = path_angle - current.theta + math.atan2(self.k * cross_track_error, self.max_velocity)
        linear = self.max_velocity

        return linear, angular

    def _find_nearest_point(self, pose, path):
        min_dist = float('inf')
        nearest = 0
        for i, p in enumerate(path):
            d = math.sqrt((p.x - pose.x)**2 + (p.y - pose.y)**2)
            if d < min_dist:
                min_dist = d
                nearest = i
        return nearest

    def _cross_track_error(self, pose, p1, p2):
        x0, y0 = pose.x, pose.y
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y

        return abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


class StanleyController:
    def __init__(self, k: float = 1.0, max_velocity: float = 1.0):
        self.k = k
        self.max_velocity = max_velocity

    def compute(self, current: State, path: List[State]) -> Tuple[float, float]:
        if not path:
            return 0.0, 0.0

        nearest_idx = self._find_nearest_point(current, path)
        if nearest_idx >= len(path) - 1:
            return 0.0, 0.0

        p1 = path[nearest_idx]
        p2 = path[min(nearest_idx + 1, len(path) - 1)]

        path_angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
        cross_track_error = self._cross_track_error(current, p1, p2)
        heading_error = path_angle - current.theta

        angular = path_angle - current.theta + math.atan2(self.k * cross_track_error, self.max_velocity)
        linear = self.max_velocity

        return linear, angular

    def _find_nearest_point(self, pose, path):
        min_dist = float('inf')
        nearest = 0
        for i, p in enumerate(path):
            d = math.sqrt((p.x - pose.x)**2 + (p.y - pose.y)**2)
            if d < min_dist:
                min_dist = d
                nearest = i
        return nearest

    def _cross_track_error(self, pose, p1, p2):
        x0, y0 = pose.x, pose.y
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y

        return abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


class StanleyController:
    def __init__(self, k: float = 1.0, max_velocity: float = 1.0):
        self.k = k
        self.max_velocity = max_velocity

    def compute(self, current: State, path: List[State]) -> Tuple[float, float]:
        if not path:
            return 0.0, 0.0

        nearest_idx = self._find_nearest_point(current, path)
        if nearest_idx >= len(path) - 1:
            return 0.0, 0.0

        p1 = path[nearest_idx]
        p2 = path[min(nearest_idx + 1, len(path) - 1)]

        path_angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
        cross_track_error = self._cross_track_error(current, p1, p2)
        heading_error = path_angle - current.theta

        angular = path_angle - current.theta + math.atan2(self.k * cross_track_error, self.max_velocity)
        linear = self.max_velocity

        return linear, angular

    def _find_nearest_point(self, pose, path):
        min_dist = float('inf')
        nearest = 0
        for i, p in enumerate(path):
            d = math.sqrt((p.x - pose.x)**2 + (p.y - pose.y)**2)
            if d < min_dist:
                min_dist = d
                nearest = i
        return nearest

    def _cross_track_error(self, pose, p1, p2):
        x0, y0 = pose.x, pose.y
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y

        return abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


class StanleyController:
    def __init__(self, k: float = 1.0, max_velocity: float = 1.0):
        self.k = k
        self.max_velocity = max_velocity

    def compute(self, current: State, path: List[State]) -> Tuple[float, float]:
        if not path:
            return 0.0, 0.0

        nearest_idx = self._find_nearest_point(current, path)
        if nearest_idx >= len(path) - 1:
            return 0.0, 0.0

        p1 = path[nearest_idx]
        p2 = path[min(nearest_idx + 1, len(path) - 1)]

        path_angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
        cross_track_error = self._cross_track_error(current, p1, p2)
        heading_error = path_angle - current.theta

        angular = path_angle - current.theta + math.atan2(self.k * cross_track_error, self.max_velocity)
        linear = self.max_velocity

        return linear, angular

    def _find_nearest_point(self, pose, path):
        min_dist = float('inf')
        nearest = 0
        for i, p in enumerate(path):
            d = math.sqrt((p.x - pose.x)**2 + (p.y - pose.y)**2)
            if d < min_dist:
                min_dist = d
                nearest = i
        return nearest

    def _cross_track_error(self, pose, p1, p2):
        x0, y0 = pose.x, pose.y
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y

        return abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


class StanleyController:
    def __init__(self, k: float = 1.0, max_velocity: float = 1.0):
        self.k = k
        self.max_velocity = max_velocity

    def compute(self, current: State, path: List[State]) -> Tuple[float, float]:
        if not path:
            return 0.0, 0.0

        nearest_idx = self._find_nearest_point(current, path)
        if nearest_idx >= len(path) - 1:
            return 0.0, 0.0

        p1 = path[nearest_idx]
        p2 = path[min(nearest_idx + 1, len(path) - 1)]

        path_angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
        cross_track_error = self._cross_track_error(current, p1, p2)
        heading_error = path_angle - current.theta

        angular = path_angle - current.theta + math.atan2(self.k * cross_track_error, self.max_velocity)
        linear = self.max_velocity

        return linear, angular

    def _find_nearest_point(self, pose, path):
        min_dist = float('inf')
        nearest = 0
        for i, p in enumerate(path):
            d = math.sqrt((p.x - pose.x)**2 + (p.y - pose.y)**2)
            if d < min_dist:
                min_dist = d
                nearest = i
        return nearest

    def _cross_track_error(self, pose, p1, p2):
        x0, y0 = pose.x, pose.y
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y

        return abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


class StanleyController:
    def __init__(self, k: float = 1.0, max_velocity: float = 1.0):
        self.k = k
        self.max_velocity = max_velocity

    def compute(self, current: State, path: List[State]) -> Tuple[float, float]:
        if not path:
            return 0.0, 0.0

        nearest_idx = self._find_nearest_point(current, path)
        if nearest_idx >= len(path) - 1:
            return 0.0, 0.0

        p1 = path[nearest_idx]
        p2 = path[min(nearest_idx + 1, len(path) - 1)]

        path_angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
        cross_track_error = self._cross_track_error(current, p1, p2)
        heading_error = path_angle - current.theta

        angular = path_angle - current.theta + math.atan2(self.k * cross_track_error, self.max_velocity)
        linear = self.max_velocity

        return linear, angular

    def _find_nearest_point(self, pose, path):
        min_dist = float('inf')
        nearest = 0
        for i, p in enumerate(path):
            d = math.sqrt((p.x - pose.x)**2 + (p.y - pose.y)**2)
            if d < min_dist:
                min_dist = d
                nearest = i
        return nearest

    def _cross_track_error(self, pose, p1, p2):
        x0, y0 = pose.x, pose.y
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y

        return abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


class StanleyController:
    def __init__(self, k: float = 1.0, max_velocity: float = 1.0):
        self.k = k
        self.max_velocity = max_velocity

    def compute(self, current: State, path: List[State]) -> Tuple[float, float]:
        if not path:
            return 0.0, 0.0

        nearest_idx = self._find_nearest_point(current, path)
        if nearest_idx >= len(path) - 1:
            return 0.0, 0.0

        p1 = path[nearest_idx]
        p2 = path[min(nearest_idx + 1, len(path) - 1)]

        path_angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
        cross_track_error = self._cross_track_error(current, p1, p2)
        heading_error = path_angle - current.theta

        angular = path_angle - current.theta + math.atan2(self.k * cross_track_error, self.max_velocity)
        linear = self.max_velocity

        return linear, angular

    def _find_nearest_point(self, pose, path):
        min_dist = float('inf')
        nearest = 0
        for i, p in enumerate(path):
            d = math.sqrt((p.x - pose.x)**2 + (p.y - pose.y)**2)
            if d < min_dist:
                min_dist = d
                nearest = i
        return nearest

    def _cross_track_error(self, pose, p1, p2):
        x0, y0 = pose.x, pose.y
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y

        return abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
