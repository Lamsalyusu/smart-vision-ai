from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict


class AgentType(Enum):
    ROBOT = "robot"
    DRONE = "drone"
    VEHICLE = "vehicle"
    HUMAN = "human"
    VIRTUAL = "virtual"
    SWARM = "swarm"


class CommunicationType(Enum):
    DIRECT = "direct"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"
    PUBLISH_SUBSCRIBE = "pub_sub"
    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"


class CoordinationStrategy(Enum):
    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"
    HIERARCHICAL = "hierarchical"
    MARKET_BASED = "market_based"
    CONSENSUS = "consensus"
    SWARM = "swarm"


@dataclass(frozen=True, slots=True)
class AgentState:
    id: str
    agent_type: AgentType
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    battery: float = 1.0
    status: str = "active"
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Task:
    id: str
    type: str
    priority: int
    location: np.ndarray
    requirements: List[str] = field(default_factory=list)
    deadline: Optional[float] = None
    reward: float = 1.0
    assigned_agent: Optional[str] = None
    status: str = "pending"


@dataclass(frozen=True, slots=True)
class Message:
    sender: str
    receiver: str
    content: Any
    msg_type: str
    timestamp: float = field(default_factory=time.time)
    priority: int = 0


class BaseAgent(ABC):
    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        initial_state: AgentState
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.state = initial_state
        self.tasks: Dict[str, Task] = {}
        self.message_queue: List[Message] = []
        self.neighbors: Set[str] = set()
        self.knowledge_base: Dict[str, Any] = {}

    @abstractmethod
    def step(self, dt: float, environment: 'MultiAgentEnvironment'):
        pass

    def send_message(self, receiver: str, content: Any, msg_type: str, priority: int = 0):
        message = Message(
            sender=self.agent_id,
            receiver=receiver,
            content=content,
            msg_type=msg_type,
            priority=priority
        )
        return message

    def receive_message(self, message: Message):
        self.message_queue.append(message)

    def assign_task(self, task: Task):
        self.tasks[task.id] = task

    def complete_task(self, task_id: str):
        if task_id in self.tasks:
            self.tasks[task_id] = Task(
                id=self.tasks[task_id].id,
                type=self.tasks[task_id].type,
                priority=self.tasks[task_id].priority,
                location=self.tasks[task_id].location,
                requirements=self.tasks[task_id].requirements,
                deadline=self.tasks[task_id].deadline,
                reward=self.tasks[task_id].reward,
                assigned_agent=self.tasks[task_id].assigned_agent,
                status="completed"
            )

    def update_state(self, new_state: AgentState):
        self.state = new_state


class RobotAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        initial_state: AgentState,
        max_speed: float = 5.0,
        max_acceleration: float = 2.0,
        sensor_range: float = 10.0
    ):
        super().__init__(agent_id, AgentType.ROBOT, initial_state)
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.sensor_range = sensor_range
        self.path: List[np.ndarray] = []
        self.path_index = 0

    def step(self, dt: float, environment: 'MultiAgentEnvironment'):
        self._process_messages()
        self._execute_path(dt)
        self._avoid_collisions(environment)
        self._update_battery(dt)

    def _process_messages(self):
        for msg in self.message_queue:
            if msg.msg_type == "path":
                self.path = msg.content
                self.path_index = 0
            elif msg.msg_type == "task":
                task = msg.content
                self.assign_task(task)
        self.message_queue.clear()

    def _execute_path(self, dt: float):
        if self.path_index < len(self.path):
            target = self.path[self.path_index]
            direction = target - self.state.position[:2]
            distance = np.linalg.norm(direction)

            if distance < 0.5:
                self.path_index += 1
            elif distance > 0:
                direction = direction / distance
                desired_velocity = direction * min(self.max_speed, distance)
                accel = desired_velocity - self.state.velocity[:2]
                accel_norm = np.linalg.norm(accel)
                if accel_norm > self.max_acceleration:
                    accel = accel / accel_norm * self.max_acceleration

                new_velocity = self.state.velocity[:2] + accel * dt
                new_position = self.state.position[:2] + new_velocity * dt

                self.state = AgentState(
                    id=self.state.id,
                    agent_type=self.state.agent_type,
                    position=np.array([new_position[0], new_position[1], self.state.position[2]]),
                    velocity=np.array([new_velocity[0], new_velocity[1], self.state.velocity[2]]),
                    orientation=self.state.orientation,
                    battery=self.state.battery,
                    status=self.state.status,
                    capabilities=self.state.capabilities,
                    metadata=self.state.metadata
                )

    def _avoid_collisions(self, environment: 'MultiAgentEnvironment'):
        for other_id, other_agent in environment.agents.items():
            if other_id == self.agent_id:
                continue

            dist = np.linalg.norm(self.state.position - other_agent.state.position)
            if dist < 2.0 and dist > 0:
                away = (self.state.position - other_agent.state.position) / dist
                self.state = AgentState(
                    id=self.state.id,
                    agent_type=self.state.agent_type,
                    position=self.state.position + away * 0.5,
                    velocity=self.state.velocity,
                    orientation=self.state.orientation,
                    battery=self.state.battery,
                    status=self.state.status,
                    capabilities=self.state.capabilities,
                    metadata=self.state.metadata
                )

    def _update_battery(self, dt: float):
        self.state = AgentState(
            id=self.state.id,
            agent_type=self.state.agent_type,
            position=self.state.position,
            velocity=self.state.velocity,
            orientation=self.state.orientation,
            battery=max(0, self.state.battery - dt * 0.001),
            status=self.state.status,
            capabilities=self.state.capabilities,
            metadata=self.state.metadata
        )


class DroneAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        initial_state: AgentState,
        max_speed: float = 15.0,
        max_altitude: float = 100.0,
        sensor_range: float = 50.0
    ):
        super().__init__(agent_id, AgentType.DRONE, initial_state)
        self.max_speed = max_speed
        self.max_altitude = max_altitude
        self.sensor_range = sensor_range

    def step(self, dt: float, environment: 'MultiAgentEnvironment'):
        self._process_messages()
        self._execute_3d_path(dt)

    def _process_messages(self):
        for msg in self.message_queue:
            if msg.msg_type == "waypoint":
                self.path = msg.content
                self.path_index = 0
        self.message_queue.clear()

    def _execute_3d_path(self, dt: float):
        if hasattr(self, 'path') and self.path_index < len(self.path):
            target = self.path[self.path_index]
            direction = target - self.state.position
            distance = np.linalg.norm(direction)

            if distance < 1.0:
                self.path_index += 1
            elif distance > 0:
                direction = direction / distance
                new_velocity = direction * min(self.max_speed, distance)
                new_position = self.state.position + new_velocity * dt

                new_position[2] = np.clip(new_position[2], 0, self.max_altitude)

                self.state = AgentState(
                    id=self.state.id,
                    agent_type=self.state.agent_type,
                    position=new_position,
                    velocity=new_velocity,
                    orientation=self.state.orientation,
                    battery=self.state.battery,
                    status=self.state.status,
                    capabilities=self.state.capabilities,
                    metadata=self.state.metadata
                )


class MultiAgentEnvironment:
    def __init__(self, bounds: Tuple[np.ndarray, np.ndarray] = None):
        self.agents: Dict[str, BaseAgent] = {}
        self.tasks: Dict[str, Task] = {}
        self.bounds = bounds or (np.array([-100, -100, 0]), np.array([100, 100, 100]))
        self.message_bus: List[Message] = []
        self.time = 0.0
        self.obstacles: List[Dict[str, Any]] = []

    def add_agent(self, agent: BaseAgent):
        self.agents[agent.agent_id] = agent

    def remove_agent(self, agent_id: str):
        if agent_id in self.agents:
            del self.agents[agent_id]

    def add_task(self, task: Task):
        self.tasks[task.id] = task

    def add_obstacle(self, position: np.ndarray, radius: float):
        self.obstacles.append({'position': position, 'radius': radius})

    def step(self, dt: float):
        self.time += dt

        for agent in self.agents.values():
            agent.step(dt, self)

        self._deliver_messages()
        self._check_task_completion()

    def _deliver_messages(self):
        for message in self.message_bus:
            if message.receiver in self.agents:
                self.agents[message.receiver].receive_message(message)
        self.message_bus.clear()

    def broadcast(self, sender: str, content: Any, msg_type: str):
        for agent_id in self.agents:
            if agent_id != sender:
                self.message_bus.append(Message(sender, agent_id, content, msg_type))

    def send_message(self, message: Message):
        self.message_bus.append(message)

    def _check_task_completion(self):
        for task_id, task in list(self.tasks.items()):
            if task.status != "pending":
                continue

            if task.assigned_agent and task.assigned_agent in self.agents:
                agent = self.agents[task.assigned_agent]
                dist = np.linalg.norm(agent.state.position - task.location)
                if dist < 2.0:
                    agent.complete_task(task_id)
                    self.tasks[task_id] = Task(
                        id=task.id,
                        type=task.type,
                        priority=task.priority,
                        location=task.location,
                        requirements=task.requirements,
                        deadline=task.deadline,
                        reward=task.reward,
                        assigned_agent=task.assigned_agent,
                        status="completed"
                    )


class TaskAllocator:
    def __init__(self, strategy: CoordinationStrategy = CoordinationStrategy.MARKET_BASED):
        self.strategy = strategy

    def allocate_tasks(self, environment: MultiAgentEnvironment) -> Dict[str, str]:
        if self.strategy == CoordinationStrategy.MARKET_BASED:
            return self._market_based_allocation(environment)
        elif self.strategy == CoordinationStrategy.CONSENSUS:
            return self._consensus_allocation(environment)
        elif self.strategy == CoordinationStrategy.CENTRALIZED:
            return self._centralized_allocation(environment)
        elif self.strategy == CoordinationStrategy.SWARM:
            return self._swarm_allocation(environment)
        return {}

    def _market_based_allocation(self, environment: MultiAgentEnvironment) -> Dict[str, str]:
        assignments = {}
        pending_tasks = [t for t in environment.tasks.values() if t.status == "pending"]

        for task in pending_tasks:
            best_agent = None
            best_bid = float('inf')

            for agent_id, agent in environment.agents.items():
                if any(t.status == "assigned" for t in agent.tasks.values()):
                    continue

                if task.requirements and not all(req in agent.state.capabilities for req in task.requirements):
                    continue

                dist = np.linalg.norm(agent.state.position - task.location)
                bid = dist - task.reward * 10

                if bid < best_bid:
                    best_bid = bid
                    best_agent = agent_id

            if best_agent:
                assignments[task.id] = best_agent
                environment.tasks[task.id] = Task(
                    id=task.id,
                    type=task.type,
                    priority=task.priority,
                    location=task.location,
                    requirements=task.requirements,
                    deadline=task.deadline,
                    reward=task.reward,
                    assigned_agent=best_agent,
                    status="assigned"
                )
                environment.agents[best_agent].assign_task(
                    Task(id=task.id, type=task.type, priority=task.priority, location=task.location, status="assigned")
                )

        return assignments

    def _consensus_allocation(self, environment: MultiAgentEnvironment) -> Dict[str, str]:
        return {}

    def _centralized_allocation(self, environment: MultiAgentEnvironment) -> Dict[str, str]:
        return self._market_based_allocation(environment)

    def _swarm_allocation(self, environment: MultiAgentEnvironment) -> Dict[str, str]:
        return {}


class FormationController:
    def __init__(
        self,
        formation_type: str = "line",
        spacing: float = 2.0
    ):
        self.formation_type = formation_type
        self.spacing = spacing
        self.leader_id: Optional[str] = None

    def set_leader(self, leader_id: str):
        self.leader_id = leader_id

    def compute_formation_positions(
        self,
        leader_state: AgentState,
        num_followers: int
    ) -> List[np.ndarray]:
        positions = []

        if self.formation_type == "line":
            for i in range(num_followers):
                offset = np.array([-(i + 1) * self.spacing, 0, 0])
                rotated = self._rotate_vector(offset, leader_state.orientation)
                positions.append(leader_state.position + rotated)

        elif self.formation_type == "vee":
            for i in range(num_followers):
                side = 1 if i % 2 == 0 else -1
                row = (i // 2) + 1
                offset = np.array([-row * self.spacing, side * row * self.spacing, 0])
                rotated = self._rotate_vector(offset, leader_state.orientation)
                positions.append(leader_state.position + rotated)

        elif self.formation_type == "circle":
            for i in range(num_followers):
                angle = 2 * math.pi * i / num_followers
                offset = np.array([
                    self.spacing * math.cos(angle),
                    self.spacing * math.sin(angle),
                    0
                ])
                rotated = self._rotate_vector(offset, leader_state.orientation)
                positions.append(leader_state.position + rotated)

        return positions

    def _rotate_vector(self, vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat
        return np.array([
            (1 - 2*y*y - 2*z*z)*vec[0] + (2*x*y - 2*w*z)*vec[1] + (2*x*z + 2*w*y)*vec[2],
            (2*x*y + 2*w*z)*vec[0] + (1 - 2*x*x - 2*z*z)*vec[1] + (2*y*z - 2*w*x)*vec[2],
            (2*x*z - 2*w*y)*vec[0] + (2*y*z + 2*w*x)*vec[1] + (1 - 2*x*x - 2*y*y)*vec[2]
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
        agent: BaseAgent,
        neighbors: List[BaseAgent]
    ) -> np.ndarray:
        separation = self._separation(agent, neighbors)
        alignment = self._alignment(agent, neighbors)
        cohesion = self._cohesion(agent, neighbors)

        return (
            self.separation_weight * separation +
            self.alignment_weight * alignment +
            self.cohesion_weight * cohesion
        )

    def _separation(self, agent: BaseAgent, neighbors: List[BaseAgent]) -> np.ndarray:
        steering = np.zeros(3)
        count = 0

        for neighbor in neighbors:
            dist = np.linalg.norm(agent.state.position - neighbor.state.position)
            if 0 < dist < self.separation_radius:
                diff = agent.state.position - neighbor.state.position
                diff = diff / (dist + 1e-6)
                steering += diff
                count += 1

        if count > 0:
            steering = steering / count
            steering = steering / (np.linalg.norm(steering) + 1e-6)

        return steering

    def _alignment(self, agent: BaseAgent, neighbors: List[BaseAgent]) -> np.ndarray:
        steering = np.zeros(3)
        count = 0

        for neighbor in neighbors:
            dist = np.linalg.norm(agent.state.position - neighbor.state.position)
            if dist < self.neighbor_radius:
                steering += neighbor.state.velocity
                count += 1

        if count > 0:
            steering = steering / count
            steering = steering / (np.linalg.norm(steering) + 1e-6)

        return steering

    def _cohesion(self, agent: BaseAgent, neighbors: List[BaseAgent]) -> np.ndarray:
        center = np.zeros(3)
        count = 0

        for neighbor in neighbors:
            dist = np.linalg.norm(agent.state.position - neighbor.state.position)
            if dist < self.neighbor_radius:
                center += neighbor.state.position
                count += 1

        if count > 0:
            center = center / count
            steering = center - agent.state.position
            steering = steering / (np.linalg.norm(steering) + 1e-6)
        else:
            steering = np.zeros(3)

        return steering


class MultiAgentMetrics:
    @staticmethod
    def compute_coverage(agents: Dict[str, BaseAgent], area_bounds: Tuple[np.ndarray, np.ndarray]) -> float:
        min_bounds, max_bounds = area_bounds
        area = np.prod(max_bounds - min_bounds)

        if area == 0:
            return 0.0

        covered = set()
        for agent in agents.values():
            x = int((agent.state.position[0] - min_bounds[0]) / 1.0)
            y = int((agent.state.position[1] - min_bounds[1]) / 1.0)
            covered.add((x, y))

        return len(covered) / (area / 1.0)

    @staticmethod
    def compute_connectivity(agents: Dict[str, BaseAgent], comm_range: float = 10.0) -> float:
        if len(agents) < 2:
            return 1.0

        agent_ids = list(agents.keys())
        connected = 0
        total = 0

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                total += 1
                dist = np.linalg.norm(
                    agents[agent_ids[i]].state.position -
                    agents[agent_ids[j]].state.position
                )
                if dist <= comm_range:
                    connected += 1

        return connected / total if total > 0 else 0.0

    @staticmethod
    def compute_task_completion_rate(environment: MultiAgentEnvironment) -> float:
        if not environment.tasks:
            return 1.0

        completed = sum(1 for t in environment.tasks.values() if t.status == "completed")
        return completed / len(environment.tasks)


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet