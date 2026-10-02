from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict
import copy


class CoordinationStrategy(Enum):
    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"
    HIERARCHICAL = "hierarchical"
    MARKET_BASED = "market_based"
    CONSENSUS = "consensus"
    SWARM = "swarm"
    COALITION = "coalition"
    BEHAVIOR_BASED = "behavior_based"


class FormationType(Enum):
    LINE = "line"
    VEE = "vee"
    WEDGE = "wedge"
    CIRCLE = "circle"
    DIAMOND = "diamond"
    COLUMN = "column"
    ECHELON = "echelon"
    CUSTOM = "custom"


class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class RobotState:
    robot_id: str
    position: np.ndarray
    velocity: np.ndarray
    orientation: float
    battery: float = 1.0
    status: str = "active"
    capabilities: List[str] = field(default_factory=list)
    current_task: Optional[str] = None


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    task_type: str
    priority: int
    location: np.ndarray
    requirements: List[str] = field(default_factory=list)
    deadline: Optional[float] = None
    reward: float = 1.0
    assigned_robot: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    dependencies: List[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Formation:
    formation_type: FormationType
    leader_id: str
    positions: Dict[str, np.ndarray]
    spacing: float
    orientation: float = 0.0


@dataclass(frozen=True, slots=True)
class Coalition:
    coalition_id: str
    members: List[str]
    tasks: List[str]
    leader: Optional[str] = None
    shared_resources: Dict[str, Any] = field(default_factory=dict)


class MultiRobotCoordinator:
    def __init__(
        self,
        strategy: CoordinationStrategy = CoordinationStrategy.DECENTRALIZED,
        communication_range: float = 50.0
    ):
        self.strategy = strategy
        self.communication_range = communication_range
        self.robots: Dict[str, RobotState] = {}
        self.tasks: Dict[str, Task] = {}
        self.formations: Dict[str, Formation] = {}
        self.coalitions: Dict[str, Coalition] = {}
        self.task_assignments: Dict[str, str] = {}
        self.robot_id_counter = 0

    def add_robot(
        self,
        robot_id: str,
        position: np.ndarray,
        capabilities: List[str] = None
    ):
        self.robots[robot_id] = RobotState(
            robot_id=robot_id,
            position=position,
            velocity=np.zeros(3),
            orientation=0.0,
            capabilities=capabilities or []
        )

    def remove_robot(self, robot_id: str):
        if robot_id in self.robots:
            del self.robots[robot_id]

    def add_task(self, task: Task):
        self.tasks[task.task_id] = task

    def remove_task(self, task_id: str):
        if task_id in self.tasks:
            del self.tasks[task_id]

    def assign_task(self, task_id: str, robot_id: str) -> bool:
        if task_id not in self.tasks or robot_id not in self.robots:
            return False

        task = self.tasks[task_id]
        robot = self.robots[robot_id]

        if not all(req in robot.capabilities for req in task.requirements):
            return False

        if task.assigned_robot is not None:
            return False

        task = Task(
            task_id=task.task_id,
            task_type=task.task_type,
            priority=task.priority,
            location=task.location,
            requirements=task.requirements,
            deadline=task.deadline,
            reward=task.reward,
            assigned_robot=robot_id,
            status=TaskStatus.ASSIGNED,
            progress=0.0,
            dependencies=task.dependencies
        )
        self.tasks[task_id] = task
        self.task_assignments[task_id] = robot_id
        self.robots[robot_id] = RobotState(
            robot_id=robot.robot_id,
            position=robot.position,
            velocity=robot.velocity,
            orientation=robot.orientation,
            battery=robot.battery,
            status=robot.status,
            capabilities=robot.capabilities,
            current_task=task_id
        )
        return True

    def step(self, dt: float):
        for robot_id, robot in self.robots.items():
            if robot.current_task:
                self._execute_task(robot_id, dt)

        self._update_formations(dt)
        self._manage_coalitions()

    def _execute_task(self, robot_id: str, dt: float):
        robot = self.robots[robot_id]
        task_id = robot.current_task
        if not task_id or task_id not in self.tasks:
            return

        task = self.tasks[task_id]
        direction = task.location - self.robots[robot_id].position
        distance = np.linalg.norm(direction)

        if distance < 1.0:
            self._complete_task(robot_id)
        else:
            direction = direction / (distance + 1e-6)
            new_position = self.robots[robot_id].position + direction * min(dt, distance)
            self.robots[robot_id] = RobotState(
                robot_id=self.robots[robot_id].robot_id,
                position=new_position,
                velocity=direction * 0.5,
                orientation=self.robots[robot_id].orientation,
                battery=max(0, self.robots[robot_id].battery - dt * 0.01),
                status=self.robots[robot_id].status,
                capabilities=self.robots[robot_id].capabilities,
                current_task=task_id
            )

    def _complete_task(self, robot_id: str):
        robot = self.robots[robot_id]
        task_id = robot.current_task
        if task_id and task_id in self.tasks:
            task = self.tasks[task_id]
            self.tasks[task_id] = Task(
                task_id=task.task_id,
                task_type=task.task_type,
                priority=task.priority,
                location=task.location,
                requirements=task.requirements,
                deadline=task.deadline,
                reward=task.reward,
                assigned_robot=task.assigned_robot,
                status=TaskStatus.COMPLETED,
                progress=1.0,
                dependencies=task.dependencies
            )
            self.robots[robot_id] = RobotState(
                robot_id=robot.robot_id,
                position=robot.position,
                velocity=robot.velocity,
                orientation=robot.orientation,
                battery=robot.battery,
                status=robot.status,
                capabilities=robot.capabilities,
                current_task=None
            )
            del self.task_assignments[task_id]

    def create_formation(
        self,
        formation_type: FormationType,
        leader_id: str,
        member_ids: List[str],
        spacing: float = 2.0
    ) -> Formation:
        if leader_id not in self.robots:
            raise ValueError("Leader not found")

        leader_pos = self.robots[leader_id].position
        positions = {leader_id: leader_pos}

        for i, member_id in enumerate(member_ids):
            if member_id not in self.robots:
                continue

            if formation_type == FormationType.LINE:
                offset = np.array([-(i + 1) * spacing, 0, 0])
            elif formation_type == FormationType.VEE:
                side = 1 if i % 2 == 0 else -1
                row = (i // 2) + 1
                offset = np.array([-row * spacing, side * row * spacing, 0])
            elif formation_type == FormationType.CIRCLE:
                angle = 2 * math.pi * i / len(member_ids)
                offset = np.array([spacing * math.cos(angle), spacing * math.sin(angle), 0])
            else:
                offset = np.zeros(3)

            positions[member_id] = leader_pos + offset

        formation = Formation(
            formation_type=formation_type,
            leader_id=leader_id,
            positions=positions,
            spacing=spacing
        )
        self.formations[leader_id] = formation
        return formation

    def update_formation(self, leader_id: str, dt: float):
        if leader_id not in self.formations:
            return

        formation = self.formations[leader_id]
        leader_pos = self.robots[formation.leader_id].position

        for member_id, target_pos in formation.positions.items():
            if member_id == formation.leader_id:
                continue

            if member_id in self.robots:
                current = self.robots[member_id].position
                direction = target_pos - current
                dist = np.linalg.norm(direction)

                if dist > 0.5:
                    direction = direction / dist
                    new_pos = current + direction * min(1.0, dist)
                    self.robots[member_id] = self.robots[member_id]._replace(position=new_pos)

    def create_coalition(
        self,
        coalition_id: str,
        member_ids: List[str],
        tasks: List[str],
        leader_id: Optional[str] = None
    ) -> Coalition:
        for mid in member_ids:
            if mid not in self.robots:
                raise ValueError(f"Robot {mid} not found")

        coalition = Coalition(
            coalition_id=coalition_id,
            members=member_ids,
            tasks=tasks,
            leader=leader_id
        )
        self.coalitions[coalition_id] = coalition
        return coalition

    def _manage_coalitions(self):
        for coalition_id, coalition in list(self.coalitions.items()):
            if not all(mid in self.robots for mid in coalition.members):
                del self.coalitions[coalition_id]

    def get_robot_state(self, robot_id: str) -> Optional[RobotState]:
        return self.robots.get(robot_id)

    def get_task_status(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def get_system_status(self) -> Dict[str, Any]:
        return {
            'robots': {rid: {'position': r.position.tolist(), 'status': r.status, 'battery': r.battery, 'task': r.current_task} for rid, r in self.robots.items()},
            'tasks': {tid: {'status': t.status.value, 'assigned': t.assigned_robot, 'progress': t.progress} for tid, t in self.tasks.items()},
            'formations': len(self.formations),
            'coalitions': len(self.coalitions)
        }


class TaskAllocator:
    def __init__(self, strategy: CoordinationStrategy = CoordinationStrategy.MARKET_BASED):
        self.strategy = strategy

    def allocate(
        self,
        tasks: List[Task],
        robots: Dict[str, RobotState],
        method: str = "greedy"
    ) -> Dict[str, str]:
        if self.strategy == CoordinationStrategy.MARKET_BASED:
            return self._market_based_allocation(tasks, robots)
        elif self.strategy == CoordinationStrategy.CONSENSUS:
            return self._consensus_allocation(tasks, robots)
        elif self.strategy == CoordinationStrategy.HIERARCHICAL:
            return self._hierarchical_allocation(tasks, robots)
        return {}

    def _market_based_allocation(self, tasks: List[Task], robots: Dict[str, RobotState]) -> Dict[str, str]:
        assignments = {}
        available_robots = {rid: r for rid, r in robots.items() if r.current_task is None}

        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)

        for task in sorted_tasks:
            if task.assigned_robot:
                continue

            best_robot = None
            best_bid = float('inf')

            for robot_id, robot in available_robots.items():
                if not all(req in robot.capabilities for req in task.requirements):
                    continue

                dist = np.linalg.norm(robot.position - task.location)
                bid = dist - task.reward * 10

                if bid < best_bid:
                    best_bid = bid
                    best_robot = robot_id

            if best_robot:
                assignments[task.task_id] = best_robot
                available_robots[best_robot].current_task = task.task_id

        return assignments

    def _consensus_allocation(self, tasks: List[Task], robots: Dict[str, RobotState]) -> Dict[str, str]:
        return {}

    def _hierarchical_allocation(self, tasks: List[Task], robots: Dict[str, RobotState]) -> Dict[str, str]:
        return self._market_based_allocation(tasks, robots)


class FormationController:
    def __init__(
        self,
        formation_type: FormationType = FormationType.LINE,
        spacing: float = 2.0
    ):
        self.formation_type = formation_type
        self.spacing = spacing
        self.leader_id: Optional[str] = None

    def set_leader(self, leader_id: str):
        self.leader_id = leader_id

    def compute_formation_positions(
        self,
        leader_state: RobotState,
        num_followers: int
    ) -> List[np.ndarray]:
        positions = []

        if self.formation_type == FormationType.LINE:
            for i in range(num_followers):
                offset = np.array([-(i + 1) * self.spacing, 0, 0])
                rotated = self._rotate_vector(offset, leader_state.orientation)
                positions.append(leader_state.position + rotated)

        elif self.formation_type == FormationType.VEE:
            for i in range(num_followers):
                side = 1 if i % 2 == 0 else -1
                row = (i // 2) + 1
                offset = np.array([-row * self.spacing, side * row * self.spacing, 0])
                rotated = self._rotate_vector(offset, leader_state.orientation)
                positions.append(leader_state.position + rotated)

        elif self.formation_type == FormationType.CIRCLE:
            for i in range(num_followers):
                angle = 2 * math.pi * i / num_followers
                offset = np.array([self.spacing * math.cos(angle), self.spacing * math.sin(angle), 0])
                rotated = self._rotate_vector(offset, leader_state.orientation)
                positions.append(leader_state.position + rotated)

        return positions

    def _rotate_vector(self, vec: np.ndarray, angle: float) -> np.ndarray:
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return np.array([
            vec[0] * cos_a - vec[1] * sin_a,
            vec[0] * sin_a + vec[1] * cos_a,
            vec[2]
        ])


class SwarmBehavior:
    def __init__(
        self,
        separation_weight: float = 1.5,
        alignment_weight: float = 1.0,
        cohesion_weight: float = 1.0,
        separation_radius: float = 3.0,
        neighbor_radius: float = 10.0
    ):
        self.separation_weight = separation_weight
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight
        self.separation_radius = separation_radius
        self.neighbor_radius = neighbor_radius

    def compute_velocity(
        self,
        agent: RobotState,
        neighbors: List[RobotState]
    ) -> np.ndarray:
        separation = self._separation(agent, neighbors)
        alignment = self._alignment(agent, neighbors)
        cohesion = self._cohesion(agent, neighbors)

        return (
            self.separation_weight * separation +
            self.alignment_weight * alignment +
            self.cohesion_weight * cohesion
        )

    def _separation(self, agent: RobotState, neighbors: List[RobotState]) -> np.ndarray:
        steering = np.zeros(3)
        count = 0

        for neighbor in neighbors:
            dist = np.linalg.norm(agent.position - neighbor.position)
            if 0 < dist < self.separation_radius:
                diff = agent.position - neighbor.position
                diff = diff / (dist + 1e-6)
                steering += diff
                count += 1

        if count > 0:
            steering = steering / count
            steering = steering / (np.linalg.norm(steering) + 1e-6)

        return steering

    def _alignment(self, agent: RobotState, neighbors: List[RobotState]) -> np.ndarray:
        steering = np.zeros(3)
        count = 0

        for neighbor in neighbors:
            dist = np.linalg.norm(agent.position - neighbor.position)
            if dist < self.neighbor_radius:
                steering += neighbor.velocity
                count += 1

        if count > 0:
            steering = steering / count
            steering = steering / (np.linalg.norm(steering) + 1e-6)

        return steering

    def _cohesion(self, agent: RobotState, neighbors: List[RobotState]) -> np.ndarray:
        center = np.zeros(3)
        count = 0

        for neighbor in neighbors:
            dist = np.linalg.norm(agent.position - neighbor.position)
            if dist < self.neighbor_radius:
                center += neighbor.position
                count += 1

        if count > 0:
            center = center / count
            steering = center - agent.position
            steering = steering / (np.linalg.norm(steering) + 1e-6)
        else:
            steering = np.zeros(3)

        return steering


class MultiRobotMetrics:
    @staticmethod
    def compute_coverage(robots: Dict[str, RobotState], area_bounds: Tuple[np.ndarray, np.ndarray]) -> float:
        min_bounds, max_bounds = area_bounds
        area = np.prod(max_bounds - min_bounds)

        if area == 0:
            return 0.0

        covered = set()
        for robot in robots.values():
            x = int((robot.position[0] - min_bounds[0]) / 1.0)
            y = int((robot.position[1] - min_bounds[1]) / 1.0)
            covered.add((x, y))

        return len(covered) / (area / 1.0)

    @staticmethod
    def compute_connectivity(robots: Dict[str, RobotState], comm_range: float = 10.0) -> float:
        if len(robots) < 2:
            return 1.0

        robot_ids = list(robots.keys())
        connected = 0
        total = 0

        for i in range(len(robot_ids)):
            for j in range(i + 1, len(robot_ids)):
                total += 1
                dist = np.linalg.norm(
                    robots[robot_ids[i]].position -
                    robots[robot_ids[j]].position
                )
                if dist <= comm_range:
                    connected += 1

        return connected / total if total > 0 else 0.0

    @staticmethod
    def compute_task_completion_rate(tasks: Dict[str, Any]) -> float:
        if not tasks:
            return 1.0

        completed = sum(1 for t in tasks.values() if t.status == TaskStatus.COMPLETED)
        return completed / len(tasks)


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet