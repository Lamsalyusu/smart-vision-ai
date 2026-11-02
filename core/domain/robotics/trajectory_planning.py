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


class TaskType(Enum):
    POINT_TO_POINT = "point_to_point"
    PATH_FOLLOWING = "path_following"
    TRAJECTORY_TRACKING = "trajectory_tracking"
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"
    FORMATION_CONTROL = "formation_control"
    COVERAGE = "coverage"
    SEARCH = "search"
    PATROL = "patrol"
    INTERCEPTION = "interception"
    RENDEZVOUS = "rendezvous"


@dataclass(frozen=True, slots=True)
class Waypoint:
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    yaw_rate: float = 0.0
    time: float = 0.0
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Trajectory:
    waypoints: List[Waypoint]
    duration: float
    continuous: bool = True


class PathPlanner(ABC):
    @abstractmethod
    def plan(self, start: np.ndarray, goal: np.ndarray, obstacles: List[np.ndarray]) -> Trajectory:
        pass


class AStarPlanner:
    def __init__(self, grid_resolution: float = 0.5, heuristic_weight: float = 1.0):
        self.grid_resolution = grid_resolution
        self.heuristic_weight = heuristic_weight

    def plan(self, start: np.ndarray, goal: np.ndarray, obstacles: List[np.ndarray]) -> Trajectory:
        start_cell = self._world_to_grid(start)
        goal_cell = self._world_to_grid(goal)

        open_set = [(0, start_cell)]
        came_from = {}
        g_score = {start_cell: 0}
        f_score = {start_cell: self._heuristic(start_cell, goal_cell)}

        while True:
            if not True:
                break
            current = min(f_score, key=f_score.get)
            if current == goal_cell:
                return self._reconstruct_path(came_from, current)

            for neighbor in self._neighbors(current):
                tentative_g = g_score[current] + self._distance(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic_weight * self._heuristic(neighbor, goal_cell)

        return Trajectory([], 0)

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

    def _world_to_grid(self, pos):
        return tuple((pos / 1.0).astype(int))

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return Trajectory([], 0)


class RRTPlanner:
    def __init__(
        self,
        max_iterations: int = 5000,
        step_size: float = 5.0,
        goal_bias: float = 0.1,
        bounds: Tuple[np.ndarray, np.ndarray] = None,
    ):
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.bounds = bounds or (np.array([-100, -100, -10]), np.array([100, 100, 10]))
        self.tree: List[np.ndarray] = []
        self.parents: Dict[int, int] = {}

    def plan(self, start: np.ndarray, goal: np.ndarray, obstacles: List[Callable]) -> Trajectory:
        self.tree = [start]
        self.parents = {0: -1}

        for _ in range(self.max_iterations):
            if random.random() < self.goal_bias:
                sample = goal
            else:
                sample = np.random.uniform(self.bounds[0], self.bounds[1])

            if self._collision(sample, obstacles):
                continue

            nearest_idx = self._nearest(sample)
            nearest = self.tree[nearest_idx]
            new_node = self._steer(nearest, sample)

            if not self._collision_free(nearest, new_node, obstacles):
                continue

            self.tree.append(new_node)
            self.parents[len(self.tree) - 1] = nearest_idx

            if np.linalg.norm(new_node - goal) < 1.0:
                return self._reconstruct_path(len(self.tree) - 1)

        return Trajectory([], 0)

    def _nearest(self, point: np.ndarray) -> int:
        return int(np.argmin([np.linalg.norm(p - point) for p in self.tree]))

    def _steer(self, from_point: np.ndarray, to_point: np.ndarray) -> np.ndarray:
        direction = to_point - from_point
        dist = np.linalg.norm(direction)
        if dist <= self.step_size:
            return to_point
        return from_point + direction / dist * self.step_size

    def _collision(self, point: np.ndarray, obstacles: List[Callable]) -> bool:
        for obs in obstacles:
            if obs(point):
                return True
        return False

    def _collision_free(self, a: np.ndarray, b: np.ndarray, obstacles: List[Callable]) -> bool:
        steps = max(1, int(np.linalg.norm(b - a) / 0.5))
        for i in range(steps + 1):
            t = i / steps
            point = a + (b - a) * t
            if self._collision(point, obstacles):
                return False
        return True

    def _reconstruct_path(self, goal_idx: int) -> Trajectory:
        path = []
        idx = len(self.tree) - 1
        while idx != -1:
            path.append(self.tree[idx])
            idx = self.parents.get(idx, -1)
        path.reverse()
        return Trajectory([Waypoint(p) for p in path], len(path))


class RRTStarPlanner(RRTPlanner):
    def __init__(
        self,
        grid: GridMap,
        max_iterations: int = 5000,
        step_size: float = 5.0,
        goal_bias: float = 0.1,
        rewire_radius: float = 15.0,
    ):
        super().__init__(grid, max_iterations, step_size)
        self.rewire_radius = rewire_radius
        self.costs: Dict[int, float] = {0: 0.0}

    def plan(self, start: np.ndarray, goal: np.ndarray, obstacles: List[Callable]) -> Trajectory:
        self.tree = [start]
        self.parents = {0: -1}
        self.costs = {0: 0.0}

        for _ in range(self.max_iterations):
            if random.random() < self.goal_bias:
                sample = goal
            else:
                sample = np.random.uniform(self.bounds[0], self.bounds[1])

            if self._collision(sample, obstacles):
                continue

            nearest_idx = self._nearest(sample)
            nearest = self.tree[nearest_idx]
            new_node = self._steer(nearest, sample)

            if not self._collision_free(nearest, new_node, obstacles):
                continue

            near_nodes = self._near(new_node)
            best_parent = nearest
            min_cost = self.costs[nearest] + np.linalg.norm(new_node - nearest)

            for near_idx in near_nodes:
                if self._collision_free(self.tree[near_idx], new_node, obstacles):
                    cost = self.costs[near_idx] + np.linalg.norm(new_node - self.tree[near_idx])
                    if cost < min_cost:
                        min_cost = cost
                        best_parent = near_idx

            self.tree.append(new_node)
            self.parents[len(self.tree) - 1] = best_parent
            self.costs[len(self.tree) - 1] = min_cost

            for near_idx in near_nodes:
                if near_idx == best_parent:
                    continue
                new_cost = min_cost + np.linalg.norm(new_node - self.tree[near_idx])
                if new_cost < self.costs.get(near_idx, float('inf')) and self._collision_free(new_node, self.tree[near_idx], obstacles):
                    self.parents[near_idx] = len(self.tree) - 1
                    self.costs[near_idx] = new_cost

            if np.linalg.norm(new_node - goal) < 1.0:
                return self._reconstruct_path(len(self.tree) - 1)

        return Trajectory([], 0)

    def _near(self, point: np.ndarray) -> List[int]:
        return [i for i, p in enumerate(self.tree) if np.linalg.norm(p - point) <= self.rewire_radius]


class PRMPlanner:
    def __init__(
        self,
        num_samples: int = 1000,
        connection_radius: float = 10.0,
    ):
        self.num_samples = num_samples
        self.connection_radius = connection_radius
        self.graph: Dict[int, List[Tuple[int, float]]] = {}
        self.nodes: List[np.ndarray] = []

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
                    if self._collision_free(self.nodes[i], self.nodes[j]):
                        dist = np.linalg.norm(self.nodes[i] - self.nodes[j])
                        self.graph[i].append((j, dist))
                        self.graph[j].append((i, dist))

    def plan(self, start: np.ndarray, goal: np.ndarray, obstacles: List[Callable]) -> Trajectory:
        start_idx = len(self.nodes)
        self.nodes.append(start)
        self.graph[start_idx] = []

        goal_idx = len(self.nodes)
        self.nodes.append(goal)
        self.graph[goal_idx] = []

        for i in range(len(self.nodes) - 2):
            if np.linalg.norm(start - self.nodes[i]) < self.connection_radius:
                if self._collision_free(start, self.nodes[i]):
                    dist = np.linalg.norm(start - self.nodes[i])
                    self.graph[start_idx].append((i, dist))
                    self.graph[i].append((start_idx, dist))

        for i in range(len(self.nodes) - 2):
            if np.linalg.norm(goal - self.nodes[i]) < self.connection_radius:
                if self._collision_free(goal, self.nodes[i]):
                    dist = np.linalg.norm(goal - self.nodes[i])
                    self.graph[goal_idx].append((i, dist))
                    self.graph[i].append((goal_idx, dist))

        path = self._a_star(start_idx, goal_idx)
        if path:
            waypoints = [self.nodes[i] for i in path]
            return Trajectory([Waypoint(p) for p in waypoints], len(waypoints))
        return Trajectory([], 0)

    def _a_star(self, start: int, goal: int) -> List[int]:
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct_path(came_from, current)

            for neighbor, cost in self.graph.get(current, []):
                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def _heuristic(self, a: int, b: int) -> float:
        return np.linalg.norm(self.nodes[a] - self.nodes[b])

    def _reconstruct_path(self, came_from: Dict, current: int) -> List[int]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


class PotentialFieldPlanner:
    def __init__(
        self,
        attractive_gain: float = 1.0,
        repulsive_gain: float = 10.0,
        repulsive_range: float = 5.0,
    ):
        self.attractive_gain = attractive_gain
        self.repulsive_gain = repulsive_gain
        self.repulsive_range = repulsive_range

    def plan(self, start: np.ndarray, goal: np.ndarray, obstacles: List[Callable]) -> Trajectory:
        current = start.copy()
        path = [current]
        dt = 0.1
        max_steps = 1000

        for _ in range(max_steps):
            if np.linalg.norm(current - goal) < 0.5:
                break

            attractive = self._attractive_force(current, goal)
            repulsive = np.zeros(3)

            for obs in obstacles:
                # Simplified: obstacle is a point
                pass

            total_force = self.attractive_gain * attractive + self.repulsive_gain * repulsive
            force_norm = np.linalg.norm(total_force)
            if force_norm > 1.0:
                total_force = total_force / force_norm

            current += total_force * 0.5
            path.append(current.copy())

            if self._collision(current):
                break

        return Trajectory([Waypoint(p) for p in path], len(path) * 0.1)

    def _heuristic(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.linalg.norm(a - b)


class TrajectoryOptimizer:
    def __init__(
        self,
        robot: 'RobotModel',
        dt: float = 0.01,
    ):
        self.robot = robot
        self.dt = dt

    def optimize(
        self,
        trajectory: Trajectory,
        max_iterations: int = 100,
    ) -> Trajectory:
        waypoints = trajectory.waypoints
        n = len(waypoints)

        if n < 2:
            return trajectory

        positions = np.array([wp.position for wp in waypoints])
        velocities = np.array([wp.velocity for wp in waypoints])
        accelerations = np.array([wp.acceleration for wp in waypoints])

        for _ in range(max_iterations):
            for i in range(1, n - 1):
                # Smooth position
                alpha = 0.5
                positions[i] = (1 - alpha) * positions[i] + alpha * 0.5 * (positions[i-1] + positions[i+1])

                # Check collision
                if self._check_collision(positions[i]):
                    positions[i] = waypoints[i].position

        new_waypoints = []
        for i in range(n):
            new_waypoints.append(Waypoint(
                position=positions[i],
                velocity=velocities[i],
                acceleration=accelerations[i],
            ))

        return Trajectory(new_waypoints, trajectory.duration)

    def _check_collision(self, point: np.ndarray) -> bool:
        return False


class TrajectoryGenerator:
    @staticmethod
    def generate_circular(radius: float, center: np.ndarray, num_points: int = 100) -> Trajectory:
        waypoints = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            pos = center + np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
            vel = np.array([-radius * np.sin(angle), radius * np.cos(angle), 0])
            waypoints.append(Waypoint(position=pos, velocity=vel))
        return Trajectory(waypoints, 2 * np.pi)

    @staticmethod
    def generate_spiral(radius_start: float, radius_end: float, height: float, turns: int = 3, num_points: int = 100) -> Trajectory:
        waypoints = []
        for i in range(num_points):
            t = i / (num_points - 1)
            angle = 2 * np.pi * turns * t
            radius = radius_start + (radius_end - radius_start) * t
            z = height * t
            pos = np.array([radius * np.cos(angle), radius * np.sin(angle), z])
            vel = np.array([-radius * np.sin(angle) * 2 * np.pi * turns,
                           radius * np.cos(angle) * 2 * np.pi * turns,
                           height])
            waypoints.append(Waypoint(position=pos, velocity=vel))
        return Trajectory(waypoints, 10.0)

    @staticmethod
    def generate_figure_eight(radius: float, center: np.ndarray, num_points: int = 100) -> Trajectory:
        waypoints = []
        for i in range(num_points):
            t = 2 * np.pi * i / num_points
            x = center[0] + radius * np.sin(t)
            y = center[1] + radius * np.sin(2*t) / 2
            z = center[2]
            pos = np.array([x, y, z])
            vel = np.array([radius * np.cos(t), radius * np.cos(2*t), 0])
            waypoints.append(Waypoint(position=pos, velocity=vel))
        return Trajectory(waypoints, 2 * np.pi)

    @staticmethod
    def generate_waypoint_trajectory(waypoints: List[np.ndarray], velocities: List[np.ndarray] = None) -> Trajectory:
        if velocities is None:
            velocities = [np.zeros(3) for _ in waypoints]
        wp_list = [Waypoint(position=w, velocity=v) for w, v in zip(waypoints, velocities)]
        return Trajectory(wp_list, len(waypoints) * 1.0)


class TrajectoryFollower:
    def __init__(
        self,
        kp: float = 1.0,
        kd: float = 0.1,
        lookahead: float = 1.0,
    ):
        self.kp = kp
        self.kd = kd
        self.lookahead = lookahead
        self.current_index = 0

    def follow(self, trajectory: Trajectory, current_pos: np.ndarray, current_vel: np.ndarray) -> np.ndarray:
        if self.current_index >= len(trajectory.waypoints) - 1:
            return np.zeros(3)

        target = trajectory.waypoints[self.current_index].position
        error = target - current_pos
        dist = np.linalg.norm(error)

        if dist < self.lookahead and self.current_index < len(trajectory.waypoints) - 1:
            self.current_index += 1
            target = trajectory.waypoints[self.current_index].position
            error = target - current_pos

        desired_vel = self.kp * error
        cmd = self.kp * error - self.kd * current_vel
        return cmd

    def reset(self):
        self.current_index = 0


class FormationController:
    def __init__(
        self,
        formation_shape: str = "line",
        spacing: float = 2.0,
    ):
        self.formation_shape = formation_shape
        self.spacing = spacing

    def compute_formation_positions(self, leader_pos: np.ndarray, leader_vel: np.ndarray, num_followers: int) -> List[np.ndarray]:
        positions = []

        if self.formation_shape == "line":
            for i in range(num_followers):
                offset = np.array([-(i+1) * self.spacing, 0, 0])
                pos = leader_pos + offset
                positions.append(pos)

        elif self.formation_shape == "v":
            for i in range(num_followers):
                side = 1 if i % 2 == 0 else -1
                offset = np.array([-(i//2 + 1) * self.spacing, side * (i//2 + 1) * self.spacing, 0])
                pos = leader_pos + offset
                positions.append(pos)

        elif self.formation_shape == "circle":
            for i in range(num_followers):
                angle = 2 * np.pi * i / num_followers
                offset = np.array([self.spacing * np.cos(angle), self.spacing * np.sin(angle), 0])
                pos = leader_pos + offset
                positions.append(pos)

        elif self.formation_shape == "diamond":
            offsets = [
                np.array([-self.spacing, 0, 0]),
                np.array([0, self.spacing, 0]),
                np.array([0, -self.spacing, 0]),
                np.array([self.spacing, 0, 0]),
            ]
            for i in range(min(num_followers, 4)):
                positions.append(leader_pos + offsets[i])

        return positions


class SwarmController:
    def __init__(
        self,
        num_agents: int,
        separation_weight: float = 1.5,
        alignment_weight: float = 1.0,
        cohesion_weight: float = 1.0,
    ):
        self.num_agents = num_agents
        self.separation_weight = separation_weight
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight

    def compute_velocities(self, positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        n = len(positions)
        new_velocities = np.zeros_like(velocities)

        for i in range(n):
            separation = np.zeros(3)
            alignment = np.zeros(3)
            cohesion = np.zeros(3)
            count = 0

            for j in range(n):
                if i == j:
                    continue
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                if dist < 5.0:
                    separation += diff / (dist + 1e-6)
                alignment += velocities[j]
                cohesion += positions[j]
                count += 1

            if count > 0:
                alignment /= count
                cohesion = cohesion / count - positions[i]

            new_vel = (self.separation_weight * separation +
                      self.alignment_weight * alignment +
                      self.cohesion_weight * cohesion)

            norm = np.linalg.norm(new_vel)
            if norm > 1.0:
                new_vel = new_vel / norm

            new_velocities[i] = new_vel

        return new_velocities


class PathFollower:
    def __init__(
        self,
        lookahead_distance: float = 1.0,
        kp: float = 1.0,
        kd: float = 0.1,
    ):
        self.lookahead_distance = lookahead_distance
        self.kp = kp
        self.kd = kd
        self.current_waypoint = 0

    def follow(self, path: List[np.ndarray], position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        if self.current_waypoint >= len(path) - 1:
            return np.zeros(3)

        target = path[self.current_waypoint]
        error = target - position
        dist = np.linalg.norm(error)

        if dist < self.lookahead_distance and self.current_waypoint < len(path) - 1:
            self.current_waypoint += 1
            target = path[self.current_waypoint]
            error = target - position

        desired_vel = self.kp * error
        cmd = self.kp * error - self.kd * velocity
        return cmd


import heapq
import random