from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


class AStarPlanner:
    """2D Grid A* path planner with Euclidean heuristics."""

    @staticmethod
    def plan(
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        h, w = grid.shape
        sx, sy = start
        gx, gy = goal

        if grid[sy, sx] > 0 or grid[gy, gx] > 0:
            return None

        open_set: List[Tuple[float, int, int]] = []
        heapq.heappush(open_set, (0.0, sx, sy))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}

        def heuristic(x: int, y: int) -> float:
            return math.hypot(gx - x, gy - y)

        motions = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)), (-1, -1, math.sqrt(2))
        ]

        while open_set:
            _, cx, cy = heapq.heappop(open_set)

            if (cx, cy) == (gx, gy):
                # Reconstruct path
                path = [(cx, cy)]
                while (cx, cy) in came_from:
                    cx, cy = came_from[(cx, cy)]
                    path.append((cx, cy))
                path.reverse()
                return path

            for dx, dy, cost in motions:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if grid[ny, nx] > 0:
                    continue  # Obstacle

                tentative_g = g_score[(cx, cy)] + cost
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = (cx, cy)
                    g_score[(nx, ny)] = tentative_g
                    f_score = tentative_g + heuristic(nx, ny)
                    heapq.heappush(open_set, (f_score, nx, ny))

        return None


class RRTStarPlanner:
    """Rapidly-exploring Random Tree Star (RRT*) with asymptotic optimality and near-neighbor rewiring."""

    @dataclass
    class Node:
        x: float
        y: float
        cost: float = 0.0
        parent: Optional[RRTStarPlanner.Node] = None

    def __init__(
        self,
        bounds: Tuple[float, float, float, float] = (0.0, 10.0, 0.0, 10.0),
        step_size: float = 0.5,
        search_radius: float = 1.2,
        max_iters: int = 500,
    ) -> None:
        self.min_x, self.max_x, self.min_y, self.max_y = bounds
        self.step = step_size
        self.radius = search_radius
        self.max_iters = max_iters

    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: List[Tuple[float, float, float]],  # (x, y, radius)
    ) -> Optional[List[Tuple[float, float]]]:
        root = self.Node(start[0], start[1], cost=0.0)
        nodes = [root]

        def is_collision_free(x: float, y: float) -> bool:
            for ox, oy, r in obstacles:
                if math.hypot(x - ox, y - oy) <= r:
                    return False
            return True

        for _ in range(self.max_iters):
            # Sample random point (with 10% goal bias)
            if np.random.rand() < 0.1:
                rand_x, rand_y = goal
            else:
                rand_x = np.random.uniform(self.min_x, self.max_x)
                rand_y = np.random.uniform(self.min_y, self.max_y)

            # Find nearest node
            dists = [math.hypot(n.x - rand_x, n.y - rand_y) for n in nodes]
            nearest_idx = int(np.argmin(dists))
            nearest = nodes[nearest_idx]

            # Steer towards sample
            theta = math.atan2(rand_y - nearest.y, rand_x - nearest.x)
            new_x = nearest.x + self.step * math.cos(theta)
            new_y = nearest.y + self.step * math.sin(theta)

            if not is_collision_free(new_x, new_y):
                continue

            new_node = self.Node(new_x, new_y, cost=nearest.cost + self.step, parent=nearest)

            # RRT* Near-neighbor search & rewiring
            near_nodes = [
                n for n in nodes
                if math.hypot(n.x - new_x, n.y - new_y) <= self.radius
            ]

            # Choose parent with minimum cost
            for near in near_nodes:
                c = near.cost + math.hypot(near.x - new_x, near.y - new_y)
                if c < new_node.cost:
                    new_node.cost = c
                    new_node.parent = near

            nodes.append(new_node)

            # Rewire nearby neighbors
            for near in near_nodes:
                c = new_node.cost + math.hypot(near.x - new_x, near.y - new_y)
                if c < near.cost:
                    near.cost = c
                    near.parent = new_node

            # Check goal reachability
            if math.hypot(new_x - goal[0], new_y - goal[1]) <= self.step:
                goal_node = self.Node(goal[0], goal[1], cost=new_node.cost + math.hypot(new_x - goal[0], new_y - goal[1]), parent=new_node)
                path = []
                curr: Optional[RRTStarPlanner.Node] = goal_node
                while curr is not None:
                    path.append((curr.x, curr.y))
                    curr = curr.parent
                path.reverse()
                return path

        return None
