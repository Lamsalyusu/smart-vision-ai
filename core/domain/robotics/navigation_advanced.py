from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict, deque
import heapq


class NavigationAlgorithm(Enum):
    ASTAR = "astar"
    DIJKSTRA = "dijkstra"
    RRT = "rrt"
    RRT_STAR = "rrt_star"
    RRT_CONNECT = "rrt_connect"
    PRM = "prm"
    PRM_STAR = "prm_star"
    DWA = "dwa"
    MPC = "mpc"
    POTENTIAL_FIELD = "potential_field"
    VORONOI = "voronoi"
    VISIBILITY_GRAPH = "visibility_graph"
    ANYTIME = "anytime"
    ANYTIME_RRT = "anytime_rrt"
    INFORMED_RRT_STAR = "informed_rrt_star"
    BIT_STAR = "bit_star"
    AIT_STAR = "ait_star"
    VFH = "vfh"
    VFH_STAR = "vfh_star"
    TEMPORAL_ASTAR = "temporal_astar"
    MULTI_GOAL = "multi_goal"


class RobotType(Enum):
    DIFFERENTIAL_DRIVE = "differential_drive"
    ACKERMANN = "ackermann"
    OMNIDIRECTIONAL = "omnidirectional"
    QUADROTOR = "quadrotor"
    FIXED_WING = "fixed_wing"
    HUMANOID = "humanoid"
    MANIPULATOR = "manipulator"


@dataclass(frozen=True, slots=True)
class RobotState:
    position: np.ndarray
    velocity: np.ndarray
    orientation: float = 0.0
    angular_velocity: float = 0.0
    time: float = 0.0


@dataclass(frozen=True, slots=True)
class Path:
    waypoints: List[np.ndarray]
    total_length: float = 0.0
    total_time: float = 0.0
    is_valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Obstacle:
    center: np.ndarray
    radius: float
    velocity: Optional[np.ndarray] = None
    shape: str = "circle"


class BasePlanner(ABC):
    @abstractmethod
    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Obstacle],
        robot_state: Optional[RobotState] = None
    ) -> Path:
        pass


class AStarPlanner:
    def __init__(
        self,
        grid_resolution: float = 0.1,
        heuristic_weight: float = 1.0,
        diagonal_movement: bool = True
    ):
        self.grid_resolution = grid_resolution
        self.heuristic_weight = heuristic_weight
        self.diagonal_movement = diagonal_movement

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Obstacle],
        robot_state: Optional[RobotState] = None
    ) -> Path:
        start_node = Node(start)
        goal_node = Node(goal)

        open_set = []
        heapq.heappush(open_set, (0, id(start_node), start_node))

        closed_set = set()
        g_score = {start_node: 0.0}
        f_score = {start_node: self._heuristic(start, goal)}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if np.linalg.norm(current.position - goal) < self.grid_resolution:
                path = self._reconstruct_path(current, goal)
                return path

            closed_set.add(id(current))

            neighbors = self._get_neighbors(current.position, obstacles)

            for neighbor_pos in neighbors:
                neighbor = Node(neighbor_pos, current)
                tentative_g = g_score[current] + np.linalg.norm(neighbor_pos - current.position)

                if tentative_g < g_score.get(neighbor, float('inf')):
                    neighbor.parent = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor_pos, goal)

                    if neighbor not in closed_set:
                        heapq.heappush(open_set, (f_score[neighbor], id(neighbor), neighbor))

        return Path(waypoints=[], is_valid=False)

    def _heuristic(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.diagonal_movement:
            return np.linalg.norm(a - b)
        return np.sum(np.abs(a - b))

    def _get_neighbors(self, pos: np.ndarray, obstacles: List[Obstacle]) -> List[np.ndarray]:
        neighbors = []
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ] if self.diagonal_movement else [
            (1, 0), (-1, 0), (0, 1), (0, -1)
        ]

        for dx, dy in directions:
            new_pos = pos + np.array([dx, dy, 0]) * 0.1
            if not self._is_collision(new_pos, obstacles):
                neighbors.append(new_pos)

        return neighbors

    def _is_collision(self, pos: np.ndarray, obstacles: List[Obstacle]) -> bool:
        for obs in obstacles:
            if obs.shape == "circle":
                if np.linalg.norm(pos[:2] - obs.center[:2]) < obs.radius:
                    return True
        return False

    def _reconstruct_path(self, node: 'Node', goal: np.ndarray) -> Path:
        waypoints = []
        current = node

        while current:
            waypoints.append(current.position)
            current = current.parent

        waypoints.reverse()
        waypoints.append(goal)

        total_length = sum(
            np.linalg.norm(np.array(waypoints[i+1]) - np.array(waypoints[i]))
            for i in range(len(waypoints) - 1)
        )

        return Path(
            waypoints=waypoints,
            total_length=total_length,
            is_valid=True
        )


class RRTPlanner:
    def __init__(
        self,
        max_iterations: int = 5000,
        step_size: float = 1.0,
        goal_bias: float = 0.1,
        goal_radius: float = 1.0,
        max_distance: float = 100.0
    ):
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.goal_radius = goal_radius
        self.max_distance = max_distance

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Obstacle],
        robot_state: Optional[RobotState] = None
    ) -> Path:
        nodes = [Node(start)]

        for _ in range(self.max_iterations):
            if random.random() < self.goal_bias:
                rand_point = goal
            else:
                rand_point = np.random.uniform(-100, 100, 3)

            nearest = self._find_nearest(nodes, rand_point)
            new_point = self._steer(nearest.position, rand_point)

            if self._is_collision_free(nearest.position, new_point, obstacles):
                new_node = Node(new_point, nearest)
                nearest.children.append(new_node)
                nodes.append(new_node)

                if np.linalg.norm(new_point - goal) < 1.0:
                    goal_node = Node(goal, new_node)
                    return self._reconstruct_path(goal_node)

        return Path(waypoints=[], is_valid=False)

    def _find_nearest(self, nodes: List['Node'], point: np.ndarray) -> 'Node':
        min_dist = float('inf')
        nearest = nodes[0]

        for node in nodes:
            dist = np.linalg.norm(node.position - point)
            if dist < min_dist:
                min_dist = dist
                nearest = node

        return nearest

    def _steer(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        direction = to_pos - from_pos
        dist = np.linalg.norm(direction)

        if dist <= self.step_size:
            return to_pos

        return from_pos + direction / dist * self.step_size

    def _is_collision_free(self, from_pos: np.ndarray, to_pos: np.ndarray, obstacles: List[Obstacle]) -> bool:
        steps = max(int(np.linalg.norm(to_pos - from_pos) / 0.1), 1)

        for i in range(steps + 1):
            t = i / steps
            point = from_pos + t * (to_pos - from_pos)

            for obs in obstacles:
                if obs.shape == "circle":
                    if np.linalg.norm(point[:2] - obs.center[:2]) < obs.radius:
                        return False

        return True

    def _reconstruct_path(self, node: 'Node') -> Path:
        waypoints = []
        current = node

        while current:
            waypoints.append(current.position)
            current = current.parent

        waypoints.reverse()

        total_length = sum(
            np.linalg.norm(np.array(waypoints[i+1]) - np.array(waypoints[i]))
            for i in range(len(waypoints) - 1)
        )

        return Path(
            waypoints=waypoints,
            total_length=total_length,
            is_valid=True
        )


class RRTStarPlanner(RRTPlanner):
    def __init__(
        self,
        max_iterations: int = 5000,
        step_size: float = 1.0,
        goal_bias: float = 0.1,
        goal_radius: float = 1.0,
        max_distance: float = 100.0,
        rewire_radius: float = 5.0
    ):
        super().__init__(
            max_iterations=max_iterations,
            step_size=step_size,
            goal_bias=goal_bias,
            goal_radius=goal_radius,
            max_distance=max_distance
        )
        self.rewire_radius = rewire_radius

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Obstacle],
        robot_state: Optional[RobotState] = None
    ) -> Path:
        nodes = [Node(start)]

        for _ in range(self.max_iterations):
            if random.random() < self.goal_bias:
                rand_point = goal
            else:
                rand_point = np.random.uniform(-100, 100, 3)

            nearest = self._find_nearest(nodes, rand_point)
            new_point = self._steer(nearest.position, rand_point)

            if self._is_collision_free(nearest.position, new_point, obstacles):
                new_node = Node(new_point, nearest)
                nodes.append(new_node)

                near_nodes = self._find_near_nodes(nodes, new_point)
                self._rewire(new_node, near_nodes)

                if np.linalg.norm(new_point - goal) < self.goal_radius:
                    goal_node = Node(goal, new_node)
                    return self._reconstruct_path(goal_node)

        return Path(waypoints=[], is_valid=False)

    def _find_near_nodes(self, nodes: List['Node'], point: np.ndarray) -> List['Node']:
        near = []
        for node in nodes:
            if np.linalg.norm(node.position - point) < self.rewire_radius:
                near.append(node)
        return near

    def _rewire(self, new_node: 'Node', near_nodes: List['Node']):
        for near_node in near_nodes:
            if near_node == new_node.parent:
                continue

            cost_through_new = new_node.cost + np.linalg.norm(new_node.position - near_node.position)
            if cost_through_new < near_node.cost and self._is_collision_free(new_node.position, near_node.position, []):
                near_node.parent = new_node
                near_node.cost = cost_through_new

    def _is_collision_free(self, from_pos: np.ndarray, to_pos: np.ndarray, obstacles: List[Obstacle]) -> bool:
        return super()._is_collision_free(from_pos, to_pos, obstacles)

    def _reconstruct_path(self, node: 'Node') -> Path:
        waypoints = []
        current = node

        while current:
            waypoints.append(current.position)
            current = current.parent

        waypoints.reverse()

        total_length = sum(
            np.linalg.norm(np.array(waypoints[i+1]) - np.array(waypoints[i]))
            for i in range(len(waypoints) - 1)
        )

        return Path(
            waypoints=waypoints,
            total_length=total_length,
            is_valid=True
        )


class PRMPlanner:
    def __init__(
        self,
        num_samples: int = 1000,
        connection_radius: float = 5.0,
        max_edge_length: float = 10.0
    ):
        self.num_samples = num_samples
        self.connection_radius = connection_radius
        self.max_edge_length = max_edge_length

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Obstacle],
        robot_state: Optional[RobotState] = None
    ) -> Path:
        nodes = [Node(start), Node(goal)]

        for _ in range(self.num_samples):
            point = np.random.uniform(-50, 50, 3)
            if not self._is_collision(point, obstacles):
                nodes.append(Node(point))

        edges = self._build_edges(nodes)

        path_nodes = self._a_star(nodes, edges, 0, 1)

        if not path_nodes:
            return Path(waypoints=[], is_valid=False)

        waypoints = [n.position for n in path_nodes]
        total_length = sum(
            np.linalg.norm(np.array(waypoints[i+1]) - np.array(waypoints[i]))
            for i in range(len(waypoints) - 1)
        )

        return Path(
            waypoints=waypoints,
            total_length=total_length,
            is_valid=True
        )

    def _build_edges(self, nodes: List['Node']) -> Dict[int, List[Tuple[int, float]]]:
        edges = {i: [] for i in range(len(nodes))}

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                dist = np.linalg.norm(nodes[i].position - nodes[j].position)
                if dist <= self.connection_radius and dist <= self.max_edge_length:
                    if self._is_collision_free(nodes[i].position, nodes[j].position, []):
                        edges[i].append((j, dist))
                        edges[j].append((i, dist))

        return edges

    def _a_star(self, nodes: List['Node'], edges: Dict[int, List[Tuple[int, float]]], start_idx: int, goal_idx: int) -> List['Node']:
        open_set = [(0, start_idx)]
        g_score = {start_idx: 0.0}
        f_score = {start_idx: self._heuristic(nodes[start_idx].position, nodes[goal_idx].position)}
        came_from = {}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_idx:
                path = []
                while current in came_from:
                    path.append(nodes[current])
                    current = came_from[current]
                path.append(nodes[start_idx])
                return path[::-1]

            for neighbor, cost in edges[current]:
                tentative_g = g_score[current] + cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(nodes[neighbor].position, nodes[goal_idx].position)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def _heuristic(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.linalg.norm(a - b)

    def _is_collision(self, pos: np.ndarray, obstacles: List[Obstacle]) -> bool:
        for obs in obstacles:
            if obs.shape == "circle":
                if np.linalg.norm(pos[:2] - obs.center[:2]) < obs.radius:
                    return True
        return False

    def _is_collision_free(self, from_pos: np.ndarray, to_pos: np.ndarray, obstacles: List[Obstacle]) -> bool:
        steps = max(int(np.linalg.norm(to_pos - from_pos) / 0.1), 1)

        for i in range(steps + 1):
            t = i / steps
            point = from_pos + t * (to_pos - from_pos)

            for obs in obstacles:
                if obs.shape == "circle":
                    if np.linalg.norm(point[:2] - obs.center[:2]) < obs.radius:
                        return False

        return True


class PotentialFieldPlanner:
    def __init__(
        self,
        attractive_gain: float = 1.0,
        repulsive_gain: float = 10.0,
        repulsive_radius: float = 5.0,
        step_size: float = 0.1,
        max_iterations: int = 1000
    ):
        self.attractive_gain = attractive_gain
        self.repulsive_gain = repulsive_gain
        self.repulsive_radius = repulsive_radius
        self.step_size = step_size
        self.max_iterations = max_iterations

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Obstacle],
        robot_state: Optional[RobotState] = None
    ) -> Path:
        current = start.copy()
        waypoints = [Waypoint(position=current)]

        for i in range(self.max_iterations):
            attractive_force = self._attractive_force(current, goal)
            repulsive_force = self._repulsive_force(current, obstacles)

            total_force = attractive_force + repulsive_force

            if np.linalg.norm(total_force) < 1e-6:
                break

            direction = total_force / (np.linalg.norm(total_force) + 1e-6)
            current = current + direction * 0.1

            waypoints.append(Waypoint(position=current.copy()))

            if np.linalg.norm(current - goal) < 0.5:
                break

        if np.linalg.norm(current - goal) > 1.0:
            return Path(waypoints=[], is_valid=False)

        waypoints.append(Waypoint(position=goal))

        total_length = sum(
            np.linalg.norm(waypoints[i+1].position - waypoints[i].position)
            for i in range(len(waypoints) - 1)
        )

        return Path(
            waypoints=waypoints,
            total_length=total_length,
            is_valid=True
        )

    def _attractive_force(self, pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return self.attractive_gain * (goal - pos)

    def _repulsive_force(self, pos: np.ndarray, obstacles: List[Obstacle]) -> np.ndarray:
        force = np.zeros(3)

        for obs in obstacles:
            if obs.shape == "circle":
                diff = pos[:2] - obs.center[:2]
                dist = np.linalg.norm(diff)

                if dist < self.repulsive_radius and dist > 1e-6:
                    force_2d = self.repulsive_gain * (1.0/dist - 1.0/self.repulsive_radius) * (1.0/dist**2) * diff / dist
                    force = np.array([force_2d[0], force_2d[1], 0.0])
            elif obs.shape == "rectangle":
                closest = np.clip(pos[:2], obs.center[:2] - obs.radius, obs.center[:2] + obs.radius)
                diff = pos[:2] - closest
                dist = np.linalg.norm(diff)

                if dist < self.repulsive_radius and dist > 1e-6:
                    force_2d = self.repulsive_gain * (1.0/dist - 1.0/self.repulsive_radius) * (1.0/dist**2) * diff / dist
                    force = np.array([force_2d[0], force_2d[1], 0.0])

        return force


class DWAPlanner:
    def __init__(
        self,
        max_speed: float = 1.0,
        max_angular_speed: float = 1.0,
        max_acceleration: float = 0.5,
        max_angular_acceleration: float = 1.0,
        velocity_resolution: float = 0.1,
        angular_velocity_resolution: float = 0.1,
        prediction_horizon: float = 3.0,
        weights: Dict[str, float] = None
    ):
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed
        self.max_acceleration = max_acceleration
        self.max_angular_acceleration = max_angular_acceleration
        self.velocity_resolution = velocity_resolution
        self.angular_velocity_resolution = angular_velocity_resolution
        self.prediction_horizon = prediction_horizon
        self.weights = weights or {
            'heading': 0.5,
            'clearance': 0.3,
            'velocity': 0.2
        }

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Obstacle],
        robot_state: RobotState
    ) -> Path:
        best_trajectory = None
        best_score = -float('inf')

        v_min = max(0, robot_state.velocity[0] - self.max_acceleration * 0.1)
        v_max = min(self.max_speed, robot_state.velocity[0] + self.max_acceleration * 0.1)
        w_min = max(-self.max_angular_speed, robot_state.angular_velocity - self.max_angular_acceleration * 0.1)
        w_max = min(self.max_angular_speed, robot_state.angular_velocity + self.max_angular_acceleration * 0.1)

        for v in np.arange(v_min, v_max + self.velocity_resolution, self.velocity_resolution):
            for w in np.arange(w_min, w_max + self.angular_velocity_resolution, self.angular_velocity_resolution):
                trajectory = self._simulate_trajectory(robot_state, v, w)

                if self._is_trajectory_valid(trajectory, obstacles):
                    score = self._evaluate_trajectory(trajectory, goal)
                    if score > best_score:
                        best_score = score
                        best_trajectory = trajectory

        if best_trajectory is None:
            return Path(waypoints=[], is_valid=False)

        waypoints = [Waypoint(
            position=point,
            velocity=np.array([v, 0, 0])
        ) for point in best_trajectory]

        total_length = sum(
            np.linalg.norm(waypoints[i+1].position - waypoints[i].position)
            for i in range(len(waypoints) - 1)
        )

        return Path(
            waypoints=waypoints,
            path_type="time_parameterized",
            total_length=total_length,
            total_time=self.prediction_horizon,
            is_valid=True
        )

    def _simulate_trajectory(self, robot_state: RobotState, v: float, w: float) -> List[np.ndarray]:
        trajectory = []
        x, y = robot_state.position[:2]
        theta = robot_state.orientation

        dt = 0.1
        steps = int(self.prediction_horizon / dt)

        for _ in range(steps):
            x += v * math.cos(theta) * 0.1
            y += v * math.sin(theta) * 0.1
            theta += w * 0.1
            trajectory.append(np.array([x, y, 0]))

        return trajectory

    def _is_trajectory_valid(self, trajectory: List[np.ndarray], obstacles: List[Obstacle]) -> bool:
        for point in trajectory:
            if self._is_collision(point, obstacles):
                return False
        return True

    def _evaluate_trajectory(self, trajectory: List[np.ndarray], goal: np.ndarray) -> float:
        if not trajectory:
            return -float('inf')

        final_pos = trajectory[-1]
        goal_dist = np.linalg.norm(final_pos - goal)
        heading_score = 1.0 / (1.0 + goal_dist)

        clearance_score = 1.0
        velocity_score = 1.0

        return (self.weights.get('heading', 0.5) * heading_score +
                self.weights.get('clearance', 0.3) * clearance_score +
                self.weights.get('velocity', 0.2) * velocity_score)

    def _is_collision(self, pos: np.ndarray, obstacles: List[Obstacle]) -> bool:
        for obs in obstacles:
            if obs.shape == "circle":
                if np.linalg.norm(pos[:2] - obs.center[:2]) < obs.radius:
                    return True
        return False


class PathSmoother:
    @staticmethod
    def smooth_bspline(
        waypoints: List[np.ndarray],
        num_points: int = 50,
        degree: int = 3
    ) -> List[np.ndarray]:
        if len(waypoints) < 2:
            return waypoints

        positions = np.array(waypoints)

        try:
            from scipy.interpolate import splprep, splev
            tck, u = splprep([positions[:, 0], positions[:, 1], positions[:, 2]], k=min(degree, len(waypoints) - 1), s=0)
            u_new = np.linspace(0, 1, num_points)
            x_new, y_new, z_new = splev(u_new, tck)

            smooth_positions = np.column_stack([x_new, y_new, z_new])
            return list(smooth_positions)
        except ImportError:
            return waypoints

    @staticmethod
    def smooth_bezier(waypoints: List[np.ndarray], num_points: int = 50) -> List[np.ndarray]:
        if len(waypoints) < 4:
            return waypoints

        positions = np.array(waypoints)
        smooth_positions = []

        for i in range(len(waypoints) - 3):
            p0, p1, p2, p3 = positions[i:i+4]

            for t in np.linspace(0, 1, num_points // (len(waypoints) - 3)):
                point = (
                    (1-t)**3 * p0 +
                    3*(1-t)**2*t * p1 +
                    3*(1-t)*t**2 * p2 +
                    t**3 * p3
                )
                smooth_positions.append(point)

        return smooth_positions

    @staticmethod
    def remove_sharp_corners(
        waypoints: List[np.ndarray],
        max_angle: float = math.pi / 4
    ) -> List[np.ndarray]:
        if len(waypoints) < 3:
            return waypoints

        smoothed = [waypoints[0]]

        for i in range(1, len(waypoints) - 1):
            v1 = smoothed[-1] - waypoints[i-1]
            v2 = waypoints[i+1] - waypoints[i]

            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm > 0 and v2_norm > 0:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                angle = math.acos(np.clip(cos_angle, -1, 1))

                if angle > max_angle:
                    interp = 0.5
                    new_pos = (1 - interp) * waypoints[i] + interp * waypoints[i+1]
                    smoothed.append(new_pos)
                else:
                    smoothed.append(waypoints[i])
            else:
                smoothed.append(waypoints[i])

        smoothed.append(waypoints[-1])
        return smoothed


class TrajectoryOptimizer:
    def __init__(
        self,
        max_velocity: float = 1.0,
        max_acceleration: float = 0.5,
        max_jerk: float = 1.0
    ):
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk

    def time_parameterize(
        self,
        path: List[np.ndarray],
        dt: float = 0.1
    ) -> List[Tuple[np.ndarray, float]]:
        if len(path) < 2:
            return [(path[0], 0.0)] if path else []

        positions = np.array(path)
        distances = np.zeros(len(positions))

        for i in range(1, len(positions)):
            distances[i] = distances[i-1] + np.linalg.norm(positions[i] - positions[i-1])

        total_distance = distances[-1]
        if total_distance == 0:
            return [(path[0], 0.0)]

        time_points = self._compute_time_schedule(total_distance)
        if len(time_points) < 2:
            return [(path[0], 0.0)]

        new_path = []

        for t in np.linspace(0, time_points[-1], int(time_points[-1] / 0.1)):
            idx = np.searchsorted(np.linspace(0, 1, len(path)), t / time_points[-1])
            idx = min(idx, len(path) - 1)

            if idx > 0:
                alpha = (t / time_points[-1] - distances[idx-1] / distances[-1]) / (distances[idx] / distances[-1] - distances[idx-1] / distances[-1] + 1e-6)
                pos = (1 - alpha) * np.array(path[idx-1]) + alpha * np.array(path[idx])
            else:
                pos = np.array(path[0])

            new_path.append((pos, t))

        return new_path

    def _compute_time_schedule(self, distance: float) -> np.ndarray:
        v_max = self.max_velocity
        a_max = self.max_acceleration
        j_max = self.max_jerk

        t_acc = v_max / a_max
        d_acc = 0.5 * a_max * t_acc**2

        if distance < 2 * d_acc:
            v_peak = math.sqrt(distance * a_max)
            t_acc = v_peak / a_max
            t_const = 0
        else:
            t_const = (distance - 2 * d_acc) / v_max

        total_time = 2 * t_acc + t_const
        return np.linspace(0, total_time, int(total_time / 0.1))


class NavigationPipeline:
    def __init__(
        self,
        planner: Any,
        smoother: PathSmoother = None,
        optimizer: TrajectoryOptimizer = None
    ):
        self.planner = planner
        self.smoother = smoother or PathSmoother()
        self.optimizer = optimizer or TrajectoryOptimizer()

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Obstacle],
        robot_state: Optional[RobotState] = None
    ) -> List[Tuple[np.ndarray, float]]:
        raw_path = self.planner.plan(start, goal, obstacles, robot_state)

        if not raw_path.waypoints:
            return []

        smoothed = self.smoother.smooth_bspline(raw_path.waypoints)
        time_param = self.optimizer.time_parameterize(smoothed)

        return time_param


class NavigationMetrics:
    @staticmethod
    def path_length(path: List[np.ndarray]) -> float:
        if len(path) < 2:
            return 0.0
        return sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) for i in range(len(path) - 1))

    @staticmethod
    def path_smoothness(path: List[np.ndarray]) -> float:
        if len(path) < 3:
            return 1.0

        angles = []
        for i in range(1, len(path) - 1):
            v1 = path[i] - path[i-1]
            v2 = path[i+1] - path[i]

            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm > 0 and v2_norm > 0:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                angle = math.acos(np.clip(cos_angle, -1, 1))
                angles.append(angle)

        if not angles:
            return 1.0

        return 1.0 - np.mean(angles) / math.pi

    @staticmethod
    def clearance(path: List[np.ndarray], obstacles: List[Obstacle]) -> float:
        min_clearance = float('inf')

        for wp in path:
            for obs in obstacles:
                if obs.shape == "circle":
                    dist = np.linalg.norm(np.array(wp[:2]) - obs.center[:2]) - obs.radius
                    min_clearance = min(min_clearance, dist)
                elif obs.shape == "rectangle":
                    pass

        return min_clearance if min_clearance != float('inf') else 100.0

    @staticmethod
    def path_efficiency(path: List[np.ndarray], direct_distance: float) -> float:
        if direct_distance == 0:
            return 1.0
        total = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) for i in range(len(path) - 1))
        return direct_distance / total if total > 0 else 0.0


class Node:
    def __init__(self, position: np.ndarray, parent: 'Node' = None, cost: float = 0.0):
        self.position = position
        self.parent = parent
        self.cost = cost
        self.children: List['Node'] = []


class Waypoint:
    def __init__(
        self,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        acceleration: Optional[np.ndarray] = None,
        yaw: float = 0.0,
        time_from_start: float = 0.0,
        constraints: Dict[str, Any] = None
    ):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.yaw = yaw
        self.time_from_start = time_from_start
        self.constraints = constraints or {}


class TrajectoryOptimizer:
    def __init__(
        self,
        max_velocity: float = 1.0,
        max_acceleration: float = 0.5,
        max_jerk: float = 1.0
    ):
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk

    def time_parameterize(
        self,
        path: List[np.ndarray],
        dt: float = 0.1
    ) -> List[Tuple[np.ndarray, float]]:
        if len(path) < 2:
            return [(path[0], 0.0)] if path else []

        positions = np.array(path)
        distances = np.zeros(len(positions))

        for i in range(1, len(positions)):
            distances[i] = distances[i-1] + np.linalg.norm(positions[i] - positions[i-1])

        total_distance = distances[-1]
        if total_distance == 0:
            return [(path[0], 0.0)]

        time_points = self._compute_time_schedule(total_distance)
        if len(time_points) < 2:
            return [(path[0], 0.0)]

        new_path = []

        for t in np.linspace(0, time_points[-1], int(time_points[-1] / 0.1)):
            idx = np.searchsorted(distances, t * total_distance / time_points[-1])
            idx = min(idx, len(positions) - 1)

            if idx > 0:
                alpha = (t * total_distance / time_points[-1] - distances[idx-1]) / (distances[idx] - distances[idx-1] + 1e-6)
                pos = (1 - alpha) * positions[idx-1] + alpha * positions[idx]
            else:
                pos = positions[0]

            new_path.append((pos, t))

        return new_path

    def _compute_time_schedule(self, distance: float) -> np.ndarray:
        v_max = self.max_velocity
        a_max = self.max_acceleration
        j_max = self.max_jerk

        t_acc = v_max / a_max
        d_acc = 0.5 * a_max * t_acc**2

        if distance < 2 * d_acc:
            v_peak = math.sqrt(distance * a_max)
            t_acc = v_peak / a_max
            t_const = 0
        else:
            t_const = (distance - 2 * d_acc) / v_max

        total_time = 2 * t_acc + t_const
        return np.linspace(0, total_time, int(total_time / 0.1))


class PathMetrics:
    @staticmethod
    def path_length(path: List[np.ndarray]) -> float:
        if len(path) < 2:
            return 0.0
        return sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) for i in range(len(path) - 1))

    @staticmethod
    def path_smoothness(path: List[np.ndarray]) -> float:
        if len(path) < 3:
            return 1.0

        angles = []
        for i in range(1, len(path) - 1):
            v1 = np.array(path[i]) - np.array(path[i-1])
            v2 = np.array(path[i+1]) - np.array(path[i])

            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm > 0 and v2_norm > 0:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                angle = math.acos(np.clip(cos_angle, -1, 1))
                angles.append(angle)

        if not angles:
            return 1.0

        return 1.0 - np.mean(angles) / math.pi

    @staticmethod
    def clearance(path: List[np.ndarray], obstacles: List[Obstacle]) -> float:
        min_clearance = float('inf')

        for wp in path:
            for obs in obstacles:
                if obs.shape == "circle":
                    dist = np.linalg.norm(np.array(wp[:2]) - obs.center[:2]) - obs.radius
                    min_clearance = min(min_clearance, dist)

        return min_clearance if min_clearance != float('inf') else 100.0

    @staticmethod
    def path_efficiency(path: List[np.ndarray], direct_distance: float) -> float:
        if direct_distance == 0:
            return 1.0
        total = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) for i in range(len(path) - 1))
        return direct_distance / total if total > 0 else 0.0


class PathPlannerFactory:
    @staticmethod
    def create_planner(
        algorithm: NavigationAlgorithm,
        **kwargs
    ) -> Any:
        if algorithm == NavigationAlgorithm.ASTAR:
            return AStarPlanner(**kwargs)
        elif algorithm == NavigationAlgorithm.RRT:
            return RRTPlanner(**kwargs)
        elif algorithm == NavigationAlgorithm.RRT_STAR:
            return RRTStarPlanner(**kwargs)
        elif algorithm == NavigationAlgorithm.PRM:
            return PRMPlanner(**kwargs)
        elif algorithm == NavigationAlgorithm.POTENTIAL_FIELD:
            return PotentialFieldPlanner(**kwargs)
        elif algorithm == NavigationAlgorithm.DWA:
            return DWAPlanner(**kwargs)
        else:
            return AStarPlanner(**kwargs)


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
