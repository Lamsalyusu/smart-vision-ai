from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math
import time


class PlannerType(Enum):
    A_STAR = "a_star"
    DIJKSTRA = "dijkstra"
    RRT = "rrt"
    RRT_STAR = "rrt_star"
    PRM = "prm"
    D_STAR = "d_star"
    THETA_STAR = "theta_star"
    JUMP_POINT = "jump_point"


@dataclass(frozen=True, slots=True)
class GridCell:
    x: int
    y: int
    z: int = 0

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __eq__(self, other):
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def neighbors(self, diagonal: bool = True) -> List["GridCell"]:
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if diagonal:
            dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        return [GridCell(self.x + dx, self.y + dy, self.z) for dx, dy in dirs]

    def distance(self, other: "GridCell") -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)


@dataclass(frozen=True, slots=True)
class PathNode:
    cell: GridCell
    g_cost: float = 0.0
    h_cost: float = 0.0
    parent: Optional["PathNode"] = None

    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost


class GridMap:
    def __init__(self, width: int, height: int, resolution: float = 1.0, origin: Tuple[float, float] = (0, 0)):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = origin
        self.grid = np.zeros((height, width), dtype=np.uint8)
        self.cost_map = np.ones((height, width), dtype=np.float32)

    def set_obstacle(self, x: int, y: int, cost: float = 255):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = 255
            self.cost_map[y, x] = cost

    def set_free(self, x: int, y: int):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = 0
            self.cost_map[y, x] = 1.0

    def is_obstacle(self, x: int, y: int) -> bool:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y, x] > 128
        return True

    def get_cost(self, x: int, y: int) -> float:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.cost_map[y, x]
        return float('inf')

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        gx = int((wx - self.origin[0]) / self.resolution)
        gy = int((wy - self.origin[1]) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        wx = gx * self.resolution + self.origin[0]
        wy = gy * self.resolution + self.origin[1]
        return wx, wy


class AStarPlanner:
    def __init__(self, grid: GridMap, diagonal: bool = True, heuristic_weight: float = 1.0):
        self.grid = grid
        self.diagonal = diagonal
        self.heuristic_weight = heuristic_weight

    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        start_cell = GridCell(start[0], start[1])
        goal_cell = GridCell(goal[0], goal[1])

        if self.grid.is_obstacle(start_cell.x, start_cell.y) or self.grid.is_obstacle(goal_cell.x, goal_cell.y):
            return None

        open_set: List[PathNode] = []
        closed_set: Set[GridCell] = set()
        g_scores: Dict[GridCell, float] = {}

        start_node = PathNode(start_cell, 0, self._heuristic(start_cell, goal_cell))
        heapq.heappush(open_set, (start_node.f_cost, id(start_node), start_node))
        g_scores[start_cell] = 0

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current.cell == goal_cell:
                return self._reconstruct_path(current)

            closed_set.add(current.cell)

            for neighbor_cell in current.cell.neighbors(self.diagonal):
                if neighbor_cell in closed_set:
                    continue
                if self.grid.is_obstacle(neighbor_cell.x, neighbor_cell.y):
                    continue

                move_cost = self.grid.get_cost(neighbor_cell.x, neighbor_cell.y)
                if self.diagonal and abs(neighbor_cell.x - current.cell.x) == 1 and abs(neighbor_cell.y - current.cell.y) == 1:
                    move_cost *= 1.414

                tentative_g = current.g_cost + move_cost

                if neighbor_cell not in g_scores or tentative_g < g_scores[neighbor_cell]:
                    g_scores[neighbor_cell] = tentative_g
                    h = self._heuristic(neighbor_cell, goal_cell)
                    neighbor = PathNode(neighbor_cell, tentative_g, h * self.heuristic_weight, current)
                    heapq.heappush(open_set, (neighbor.f_cost, id(neighbor), neighbor))

        return None

    def _heuristic(self, a: GridCell, b: GridCell) -> float:
        return a.distance(b)

    def _reconstruct_path(self, node: PathNode) -> List[Tuple[int, int]]:
        path = []
        while node:
            path.append((node.cell.x, node.cell.y))
            node = node.parent
        path.reverse()
        return path


class RRTPlanner:
    def __init__(
        self,
        grid: GridMap,
        max_iterations: int = 5000,
        step_size: float = 5.0,
        goal_bias: float = 0.1,
        connect_radius: float = 10.0,
    ):
        self.grid = grid
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.connect_radius = connect_radius
        self.tree: List[GridCell] = []
        self.parents: Dict[GridCell, GridCell] = {}

    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        start_cell = GridCell(start[0], start[1])
        goal_cell = GridCell(goal[0], goal[1])

        if self.grid.is_obstacle(start_cell.x, start_cell.y) or self.grid.is_obstacle(goal_cell.x, goal_cell.y):
            return None

        self.tree = [start_cell]
        self.parents = {start_cell: None}

        for _ in range(self.max_iterations):
            if random.random() < self.goal_bias:
                sample = goal_cell
            else:
                sample = GridCell(
                    random.randint(0, self.grid.width - 1),
                    random.randint(0, self.grid.height - 1),
                )

            if self.grid.is_obstacle(sample.x, sample.y):
                continue

            nearest = self._nearest(sample)
            new_node = self._steer(nearest, sample)

            if self._collision_free(nearest, new_node):
                self.tree.append(new_node)
                self.parents[new_node] = nearest

                if new_node.distance(goal_cell) <= self.connect_radius:
                    self.parents[goal_cell] = new_node
                    return self._reconstruct_path(goal_cell)

        return None

    def _nearest(self, cell: GridCell) -> GridCell:
        return min(self.tree, key=lambda n: n.distance(cell))

    def _steer(self, from_cell: GridCell, to_cell: GridCell) -> GridCell:
        dist = from_cell.distance(to_cell)
        if dist <= self.step_size:
            return to_cell
        ratio = self.step_size / dist
        x = int(from_cell.x + (to_cell.x - from_cell.x) * ratio)
        y = int(from_cell.y + (to_cell.y - from_cell.y) * ratio)
        return GridCell(x, y)

    def _collision_free(self, a: GridCell, b: GridCell) -> bool:
        steps = max(abs(a.x - b.x), abs(a.y - b.y))
        if steps == 0:
            return not self.grid.is_obstacle(a.x, a.y)
        for i in range(steps + 1):
            t = i / steps
            x = int(a.x + (b.x - a.x) * t)
            y = int(a.y + (b.y - a.y) * t)
            if self.grid.is_obstacle(x, y):
                return False
        return True

    def _reconstruct_path(self, goal: GridCell) -> List[Tuple[int, int]]:
        path = []
        current = goal
        while current is not None:
            path.append((current.x, current.y))
            current = self.parents.get(current)
        path.reverse()
        return path


class RRTStarPlanner(RRTPlanner):
    def __init__(
        self,
        grid: GridMap,
        max_iterations: int = 5000,
        step_size: float = 5.0,
        goal_bias: float = 0.1,
        rewire_radius: float = 15.0,
    ):
        super().__init__(grid, max_iterations, step_size, goal_bias)
        self.rewire_radius = rewire_radius
        self.costs: Dict[GridCell, float] = {}

    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        start_cell = GridCell(start[0], start[1])
        goal_cell = GridCell(goal[0], goal[1])

        if self.grid.is_obstacle(start_cell.x, start_cell.y) or self.grid.is_obstacle(goal_cell.x, goal_cell.y):
            return None

        self.tree = [start_cell]
        self.parents = {start_cell: None}
        self.costs = {start_cell: 0.0}

        for _ in range(self.max_iterations):
            if random.random() < self.goal_bias:
                sample = goal_cell
            else:
                sample = GridCell(
                    random.randint(0, self.grid.width - 1),
                    random.randint(0, self.grid.height - 1),
                )

            if self.grid.is_obstacle(sample.x, sample.y):
                continue

            nearest = self._nearest(sample)
            new_node = self._steer(nearest, sample)

            if not self._collision_free(nearest, new_node):
                continue

            near_nodes = self._near(new_node)
            best_parent = nearest
            min_cost = self.costs[nearest] + nearest.distance(new_node)

            for near in near_nodes:
                if self._collision_free(near, new_node):
                    cost = self.costs[near] + near.distance(new_node)
                    if cost < min_cost:
                        min_cost = cost
                        best_parent = near

            self.tree.append(new_node)
            self.parents[new_node] = best_parent
            self.costs[new_node] = min_cost

            for near in near_nodes:
                if near == best_parent:
                    continue
                new_cost = min_cost + new_node.distance(near)
                if new_cost < self.costs.get(near, float('inf')) and self._collision_free(new_node, near):
                    self.parents[near] = new_node
                    self.costs[near] = new_cost

            if new_node.distance(goal_cell) <= self.connect_radius:
                self.parents[goal_cell] = new_node
                self.costs[goal_cell] = self.costs[new_node] + new_node.distance(goal_cell)
                return self._reconstruct_path(goal_cell)

        return None

    def _near(self, cell: GridCell) -> List[GridCell]:
        return [n for n in self.tree if n.distance(cell) <= self.rewire_radius]


class ThetaStarPlanner:
    def __init__(self, grid: GridMap, diagonal: bool = True):
        self.grid = grid
        self.diagonal = diagonal

    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        start_cell = GridCell(start[0], start[1])
        goal_cell = GridCell(goal[0], goal[1])

        if self.grid.is_obstacle(start_cell.x, start_cell.y) or self.grid.is_obstacle(goal_cell.x, goal_cell.y):
            return None

        open_set: List[Tuple[float, int, PathNode]] = []
        closed_set: Set[GridCell] = set()
        g_scores: Dict[GridCell, float] = {}

        start_node = PathNode(start_cell, 0, self._heuristic(start_cell, goal_cell))
        heapq.heappush(open_set, (start_node.f_cost, id(start_node), start_node))
        g_scores[start_cell] = 0

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current.cell == goal_cell:
                return self._reconstruct_path(current)

            closed_set.add(current.cell)

            for neighbor_cell in current.cell.neighbors(self.diagonal):
                if neighbor_cell in closed_set:
                    continue
                if self.grid.is_obstacle(neighbor_cell.x, neighbor_cell.y):
                    continue

                parent = current.parent if current.parent else current
                if self._line_of_sight(parent.cell, neighbor_cell):
                    tentative_g = parent.g_cost + parent.cell.distance(neighbor_cell)
                    parent_node = parent
                else:
                    tentative_g = current.g_cost + current.cell.distance(neighbor_cell)
                    parent_node = current

                if self.diagonal and abs(neighbor_cell.x - current.cell.x) == 1 and abs(neighbor_cell.y - current.cell.y) == 1:
                    tentative_g *= 1.414

                if neighbor_cell not in g_scores or tentative_g < g_scores[neighbor_cell]:
                    g_scores[neighbor_cell] = tentative_g
                    h = self._heuristic(neighbor_cell, goal_cell)
                    neighbor = PathNode(neighbor_cell, tentative_g, h, parent_node)
                    heapq.heappush(open_set, (neighbor.f_cost, id(neighbor), neighbor))

        return None

    def _line_of_sight(self, a: GridCell, b: GridCell) -> bool:
        x0, y0 = a.x, a.y
        x1, y1 = b.x, b.y
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if self.grid.is_obstacle(x0, y0):
                return False
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return True

    def _heuristic(self, a: GridCell, b: GridCell) -> float:
        return a.distance(b)

    def _reconstruct_path(self, node: PathNode) -> List[Tuple[int, int]]:
        path = []
        while node:
            path.append((node.cell.x, node.cell.y))
            node = node.parent
        path.reverse()
        return path


class JumpPointSearch:
    def __init__(self, grid: GridMap):
        self.grid = grid

    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        start_cell = GridCell(start[0], start[1])
        goal_cell = GridCell(goal[0], goal[1])

        if self.grid.is_obstacle(start_cell.x, start_cell.y) or self.grid.is_obstacle(goal_cell.x, goal_cell.y):
            return None

        open_set: List[Tuple[float, int, GridCell, GridCell]] = []
        closed_set: Set[GridCell] = set()
        g_scores: Dict[GridCell, float] = {}

        heapq.heappush(open_set, (self._heuristic(start_cell, goal_cell), id(start_cell), start_cell, None))
        g_scores[start_cell] = 0

        while open_set:
            _, _, current, parent = heapq.heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == goal_cell:
                return self._reconstruct_path(current, parent)

            jump_points = self._jump(current, parent, goal_cell)

            for jp in jump_points:
                if jp in closed_set:
                    continue
                tentative_g = g_scores[current] + current.distance(jp)
                if jp not in g_scores or tentative_g < g_scores[jp]:
                    g_scores[jp] = tentative_g
                    f = tentative_g + self._heuristic(jp, goal_cell)
                    heapq.heappush(open_set, (f, id(jp), jp, current))

        return None

    def _jump(self, current: GridCell, parent: Optional[GridCell], goal: GridCell) -> List[GridCell]:
        if current == goal:
            return [current]

        if parent is None:
            return []

        dx = current.x - parent.x
        dy = current.y - parent.y

        if dx != 0 and dy != 0:
            if self._is_forced(current.x - dx, current.y, dx, 0) or self._is_forced(current.x, current.y - dy, 0, dy):
                return [current]
            next_diag = self._jump(GridCell(current.x + dx, current.y + dy), current, goal)
            next_x = self._jump(GridCell(current.x + dx, current.y), current, goal)
            next_y = self._jump(GridCell(current.x, current.y + dy), current, goal)
            return next_diag + next_x + next_y
        else:
            if dx != 0:
                if self._is_forced(current.x, current.y + 1, dx, 0) or self._is_forced(current.x, current.y - 1, dx, 0):
                    return [current]
            if dy != 0:
                if self._is_forced(current.x + 1, current.y, 0, dy) or self._is_forced(current.x - 1, current.y, 0, dy):
                    return [current]

        next_cell = GridCell(current.x + dx, current.y + dy)
        if not self.grid.is_obstacle(next_cell.x, next_cell.y):
            return self._jump(next_cell, current, goal)
        return []

    def _is_forced(self, x: int, y: int, dx: int, dy: int) -> bool:
        if self.grid.is_obstacle(x + dx, y):
            return True
        if self.grid.is_obstacle(x, y + dy):
            return True
        if dx != 0 and dy != 0:
            if self.grid.is_obstacle(x - dx, y) or self.grid.is_obstacle(x, y - dy):
                return True
        return False

    def _heuristic(self, a: GridCell, b: GridCell) -> float:
        return max(abs(a.x - b.x), abs(a.y - b.y))

    def _reconstruct_path(self, current: GridCell, parent: Optional[GridCell]) -> List[Tuple[int, int]]:
        path = []
        while current is not None:
            path.append((current.x, current.y))
            current = parent
        path.reverse()
        return path


class PathOptimizer:
    @staticmethod
    def smooth(path: List[Tuple[int, int]], grid: GridMap, weight_data: float = 0.1, weight_smooth: float = 0.3) -> List[Tuple[float, float]]:
        if len(path) < 3:
            return [(float(x), float(y)) for x, y in path]

        new_path = np.array(path, dtype=np.float64)
        original = new_path.copy()

        for _ in range(100):
            for i in range(1, len(new_path) - 1):
                new_path[i] += weight_data * (original[i] - new_path[i])
                new_path[i] += weight_smooth * (new_path[i-1] + new_path[i+1] - 2 * new_path[i])

                gx, gy = int(new_path[i, 0]), int(new_path[i, 1])
                if grid.is_obstacle(gx, gy):
                    new_path[i] = original[i]

        return [(float(x), float(y)) for x, y in new_path]

    @staticmethod
    def simplify(path: List[Tuple[int, int]], grid: GridMap) -> List[Tuple[int, int]]:
        if len(path) < 3:
            return path

        simplified = [path[0]]
        i = 0

        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if grid._line_of_sight(GridCell(path[i][0], path[i][1]), GridCell(path[j][0], path[j][1])):
                    break
                j -= 1
            simplified.append(path[j])
            i = j

        return simplified


class DynamicPlanner:
    def __init__(self, grid: GridMap):
        self.grid = grid
        self.obstacles: Dict[Tuple[int, int], float] = {}

    def add_dynamic_obstacle(self, x: int, y: int, duration: float):
        self.obstacles[(x, y)] = time.time() + duration
        self.grid.set_obstacle(x, y)

    def remove_expired_obstacles(self):
        current = time.time()
        expired = [(x, y) for (x, y), expiry in self.obstacles.items() if current > expiry]
        for x, y in expired:
            self.grid.set_free(x, y)
            del self.obstacles[(x, y)]

    def replan(self, start: Tuple[int, int], goal: Tuple[int, int], planner: AStarPlanner) -> Optional[List[Tuple[int, int]]]:
        self.remove_expired_obstacles()
        return planner.plan(start, goal)


import heapq
import random
