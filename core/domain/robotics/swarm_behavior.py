from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict, deque
import copy


class SwarmBehaviorType(Enum):
    FLOCKING = "flocking"
    AGGREGATION = "aggregation"
    DISPERSION = "dispersion"
    PATTERN_FORMATION = "pattern_formation"
    COVERAGE = "coverage"
    FORAGING = "foraging"
    HUNTING = "hunting"
    ESCORT = "escort"
    CONSENSUS = "consensus"


class AgentType(Enum):
    DRONE = "drone"
    GROUND_ROBOT = "ground_robot"
    UNDERWATER = "underwater"
    SPACECRAFT = "spacecraft"
    VIRTUAL = "virtual"


@dataclass(frozen=True, slots=True)
class SwarmAgentState:
    agent_id: str
    position: np.ndarray
    velocity: np.ndarray
    orientation: float
    agent_type: AgentType
    capabilities: List[str] = field(default_factory=list)
    battery: float = 1.0
    communication_range: float = 10.0
    sensing_range: float = 5.0
    max_speed: float = 1.0
    max_acceleration: float = 1.0
    role: str = "follower"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SwarmCommand:
    command_id: str
    target_agents: List[str]
    command_type: str
    parameters: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: int = 0


@dataclass(frozen=True, slots=True)
class SwarmState:
    agents: Dict[str, SwarmAgentState]
    timestamp: float
    formation: Optional[str] = None
    mission_status: str = "active"


class SwarmBehavior(ABC):
    @abstractmethod
    def compute_control(
        self,
        agent: SwarmAgentState,
        neighbors: List[SwarmAgentState],
        global_state: Optional[SwarmState] = None
    ) -> np.ndarray:
        pass


class FlockingBehavior(SwarmBehavior):
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

    def compute_control(
        self,
        agent: SwarmAgentState,
        neighbors: List[SwarmAgentState],
        global_state: Optional[SwarmState] = None
    ) -> np.ndarray:
        separation = self._separation(agent, neighbors)
        alignment = self._alignment(agent, neighbors)
        cohesion = self._cohesion(agent, neighbors)

        return (
            self.separation_weight * separation +
            self.alignment_weight * alignment +
            self.cohesion_weight * cohesion
        )

    def _separation(self, agent: SwarmAgentState, neighbors: List[SwarmAgentState]) -> np.ndarray:
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

    def _alignment(self, agent: SwarmAgentState, neighbors: List[SwarmAgentState]) -> np.ndarray:
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

    def _cohesion(self, agent: SwarmAgentState, neighbors: List[SwarmAgentState]) -> np.ndarray:
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


class AggregationBehavior(SwarmBehavior):
    def __init__(
        self,
        attraction_weight: float = 1.0,
        repulsion_weight: float = 1.5,
        attraction_radius: float = 10.0,
        repulsion_radius: float = 2.0
    ):
        self.attraction_weight = attraction_weight
        self.repulsion_weight = repulsion_weight
        self.attraction_radius = attraction_radius
        self.repulsion_radius = repulsion_radius

    def compute_control(
        self,
        agent: SwarmAgentState,
        neighbors: List[SwarmAgentState],
        global_state: Optional[SwarmState] = None
    ) -> np.ndarray:
        attraction = self._attraction(agent, neighbors)
        repulsion = self._repulsion(agent, neighbors)

        return self.attraction_weight * attraction - self.repulsion_weight * repulsion

    def _attraction(self, agent: SwarmAgentState, neighbors: List[SwarmAgentState]) -> np.ndarray:
        steering = np.zeros(3)
        count = 0

        for neighbor in neighbors:
            dist = np.linalg.norm(agent.position - neighbor.position)
            if dist < self.attraction_radius and dist > 0:
                diff = neighbor.position - agent.position
                steering += diff / (dist + 1e-6)
                count += 1

        if count > 0:
            steering = steering / count
            steering = steering / (np.linalg.norm(steering) + 1e-6)

        return steering

    def _repulsion(self, agent: SwarmAgentState, neighbors: List[SwarmAgentState]) -> np.ndarray:
        steering = np.zeros(3)
        count = 0

        for neighbor in neighbors:
            dist = np.linalg.norm(agent.position - neighbor.position)
            if dist < self.repulsion_radius and dist > 0:
                diff = agent.position - neighbor.position
                steering += diff / (dist + 1e-6)
                count += 1

        if count > 0:
            steering = steering / count
            steering = steering / (np.linalg.norm(steering) + 1e-6)

        return steering


class PatternFormationBehavior(SwarmBehavior):
    def __init__(
        self,
        formation_type: str = "line",
        spacing: float = 2.0,
        leader_id: Optional[str] = None
    ):
        self.formation_type = formation_type
        self.spacing = spacing
        self.leader_id = leader_id

    def compute_control(
        self,
        agent: SwarmAgentState,
        neighbors: List[SwarmAgentState],
        global_state: Optional[SwarmState] = None
    ) -> np.ndarray:
        if self.leader_id and global_state:
            leader = global_state.agents.get(self.leader_id)
            if leader:
                target_pos = self._compute_formation_position(agent, leader)
                direction = target_pos - agent.position
                dist = np.linalg.norm(direction)
                if dist > 0.1:
                    return direction / (dist + 1e-6)
        return np.zeros(3)

    def _compute_formation_position(self, agent: SwarmAgentState, leader: SwarmAgentState) -> np.ndarray:
        agent_index = 0
        return agent.position


class CoverageBehavior(SwarmBehavior):
    def __init__(
        self,
        area_bounds: Tuple[np.ndarray, np.ndarray],
        resolution: float = 1.0
    ):
        self.area_bounds = area_bounds
        self.resolution = resolution
        self.visited_cells = set()

    def compute_control(
        self,
        agent: SwarmAgentState,
        neighbors: List[SwarmAgentState],
        global_state: Optional[SwarmState] = None
    ) -> np.ndarray:
        cell = self._position_to_cell(agent.position)
        self.visited_cells.add(cell)

        frontier = self._find_frontier(agent.position)
        if frontier is not None:
            direction = frontier - agent.position
            dist = np.linalg.norm(direction)
            if dist > 0:
                return direction / dist

        return np.zeros(3)

    def _position_to_cell(self, position: np.ndarray) -> Tuple[int, int]:
        min_bounds, max_bounds = self.area_bounds
        x = int((position[0] - self.area_bounds[0][0]) / self.resolution)
        y = int((position[1] - self.area_bounds[1][0]) / self.resolution)
        return (x, y)

    def _find_frontier(self, position: np.ndarray) -> Optional[np.ndarray]:
        min_bounds, max_bounds = self.area_bounds
        best_frontier = None
        best_score = -1

        for x in np.arange(self.area_bounds[0][0], self.area_bounds[1][0], self.resolution):
            for y in np.arange(self.area_bounds[0][1], self.area_bounds[1][1], self.resolution):
                cell = (int((x - self.area_bounds[0][0]) / self.resolution),
                        int((y - self.area_bounds[0][1]) / self.resolution))

                if cell not in self.visited_cells:
                    neighbor_visited = 0
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        if (cell[0]+dx, cell[1]+dy) in self.visited_cells:
                            neighbor_visited += 1

                    if neighbor_visited > 0:
                        score = neighbor_visited / math.sqrt((x - position[0])**2 + (y - position[1])**2 + 1e-6)
                        if score > best_score:
                            best_score = score
                            best_frontier = np.array([x, y, 0])

        return best_frontier


class ForagingBehavior(SwarmBehavior):
    def __init__(
        self,
        food_sources: List[np.ndarray],
        nest_position: np.ndarray,
        carrying_capacity: float = 1.0
    ):
        self.food_sources = food_sources
        self.nest_position = nest_position
        self.carrying_capacity = carrying_capacity

    def compute_control(
        self,
        agent: SwarmAgentState,
        neighbors: List[SwarmAgentState],
        global_state: Optional[SwarmState] = None
    ) -> np.ndarray:
        if agent.metadata.get('carrying', 0) > 0:
            return self._return_to_nest(agent)
        else:
            return self._find_food(agent)

    def _return_to_nest(self, agent: SwarmAgentState) -> np.ndarray:
        direction = self.nest_position - agent.position
        dist = np.linalg.norm(direction)
        if dist > 0.1:
            return direction / dist
        return np.zeros(3)

    def _find_food(self, agent: SwarmAgentState) -> np.ndarray:
        best_source = None
        best_dist = float('inf')

        for source in self.food_sources:
            dist = np.linalg.norm(source - agent.position)
            if dist < best_dist:
                best_dist = dist
                best_source = source

        if best_source is not None:
            direction = best_source - agent.position
            dist = np.linalg.norm(direction)
            if dist > 0:
                return direction / dist

        return np.zeros(3)


class ConsensusBehavior(SwarmBehavior):
    def __init__(
        self,
        consensus_type: str = "average",
        convergence_threshold: float = 1e-3
    ):
        self.consensus_type = consensus_type
        self.convergence_threshold = convergence_threshold

    def compute_control(
        self,
        agent: SwarmAgentState,
        neighbors: List[SwarmAgentState],
        global_state: Optional[SwarmState] = None
    ) -> np.ndarray:
        if not neighbors:
            return np.zeros(3)

        values = [n.metadata.get('value', 0.0) for n in neighbors]
        agent_value = agent.metadata.get('value', 0.0)

        if self.consensus_type == "average":
            avg = np.mean(values) if values else agent_value
        elif self.consensus_type == "max":
            avg = max(values) if values else agent_value
        elif self.consensus_type == "min":
            avg = min(values) if values else agent_value
        else:
            avg = agent_value

        diff = avg - agent_value
        return np.array([diff, 0, 0])


class SwarmCoordinator:
    def __init__(
        self,
        behavior: SwarmBehavior,
        communication_range: float = 10.0
    ):
        self.behavior = behavior
        self.communication_range = communication_range
        self.agents: Dict[str, SwarmAgentState] = {}
        self.global_state = SwarmState(agents={}, timestamp=time.time())
        self.commands: List[SwarmCommand] = []

    def add_agent(self, agent: SwarmAgentState):
        self.agents[agent.agent_id] = agent

    def remove_agent(self, agent_id: str):
        if agent_id in self.agents:
            del self.agents[agent_id]

    def add_command(self, command: SwarmCommand):
        self.commands.append(command)

    def step(self, dt: float) -> SwarmState:
        for cmd in self.commands:
            if cmd.target_agents == ["all"] or all(a in self.agents for a in cmd.target_agents):
                self._execute_command(cmd)

        for agent_id, agent in self.agents.items():
            neighbors = self._get_neighbors(agent)
            control = self.behavior.compute_control(agent, list(neighbors), self.global_state)

            new_velocity = agent.velocity + control * 0.01
            speed = np.linalg.norm(new_velocity)
            if speed > agent.max_speed:
                new_velocity = new_velocity / speed * agent.max_speed

            new_position = agent.position + new_velocity * 0.01

            self.agents[agent.agent_id] = SwarmAgentState(
                agent_id=agent.agent_id,
                position=new_position,
                velocity=new_velocity,
                orientation=agent.orientation,
                agent_type=agent.agent_type,
                capabilities=agent.capabilities,
                battery=max(0, agent.battery - 0.001),
                communication_range=agent.communication_range,
                sensing_range=agent.sensing_range,
                max_speed=agent.max_speed,
                max_acceleration=agent.max_acceleration,
                role=agent.role,
                metadata=agent.metadata
            )

        self.global_state = SwarmState(
            agents=copy.deepcopy(self.agents),
            timestamp=time.time()
        )

        return self.global_state

    def _get_neighbors(self, agent: SwarmAgentState) -> List[SwarmAgentState]:
        neighbors = []
        for other_id, other in self.agents.items():
            if other_id == agent.agent_id:
                continue
            dist = np.linalg.norm(agent.position - other.position)
            if dist <= agent.communication_range:
                neighbors.append(other)
        return neighbors

    def _execute_command(self, command: SwarmCommand):
        pass


class SwarmBehaviorFactory:
    @staticmethod
    def create_behavior(behavior_type: SwarmBehaviorType, **kwargs) -> SwarmBehavior:
        if behavior_type == SwarmBehaviorType.FLOCKING:
            return FlockingBehavior(**kwargs)
        elif behavior_type == SwarmBehaviorType.AGGREGATION:
            return AggregationBehavior(**kwargs)
        elif behavior_type == SwarmBehaviorType.PATTERN_FORMATION:
            return PatternFormationBehavior(**kwargs)
        elif behavior_type == SwarmBehaviorType.COVERAGE:
            return CoverageBehavior(**kwargs)
        elif behavior_type == SwarmBehaviorType.FORAGING:
            return ForagingBehavior(**kwargs)
        elif behavior_type == SwarmBehaviorType.CONSENSUS:
            return ConsensusBehavior(**kwargs)
        else:
            return FlockingBehavior()


class SwarmVisualizer:
    @staticmethod
    def draw_swarm(
        swarm_state: SwarmState,
        image_size: Tuple[int, int] = (800, 600),
        bounds: Tuple[np.ndarray, np.ndarray] = None
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if not swarm_state.agents:
            return vis

        positions = np.array([a.position[:2] for a in swarm_state.agents.values()])
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1

        scale = min(image_size[0], image_size[1]) * 0.8 / max(range_coords)

        for agent in swarm_state.agents.values():
            x = int((agent.position[0] - min_coords[0]) * scale + image_size[0] * 0.1)
            y = int(image_size[1] - (agent.position[1] - min_coords[1]) * scale - image_size[1] * 0.1)

            color = {
                AgentType.DRONE: (0, 255, 0),
                AgentType.GROUND_ROBOT: (255, 0, 0),
                AgentType.UNDERWATER: (0, 0, 255),
                AgentType.SPACECRAFT: (255, 255, 0),
                AgentType.VIRTUAL: (255, 0, 255)
            }.get(agent.agent_type, (255, 255, 255))

            cv2.circle(vis, (x, y), 5, color, -1)

            if agent.velocity is not None:
                vel_end = (int(agent.position[0] + agent.velocity[0] * 10),
                          int(agent.position[1] + agent.velocity[1] * 10))
                cv2.arrowedLine(vis, (x, y), (vel_end[0], vel_end[1]), color, 1)

        return vis

    @staticmethod
    def draw_communication_graph(
        swarm_state: SwarmState,
        image_size: Tuple[int, int] = (800, 600),
        communication_range: float = 10.0
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if not swarm_state.agents:
            return vis

        positions = np.array([a.position[:2] for a in swarm_state.agents.values()])
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1

        scale = min(image_size[0], image_size[1]) * 0.8 / max(range_coords)

        agent_positions = {}
        for agent_id, agent in swarm_state.agents.items():
            x = int((agent.position[0] - min_coords[0]) * scale + image_size[0] * 0.1)
            y = int(image_size[1] - (agent.position[1] - min_coords[1]) * scale - image_size[1] * 0.1)
            agent_positions[agent_id] = (x, y)

        for id1, agent1 in swarm_state.agents.items():
            for id2, agent2 in swarm_state.agents.items():
                if id1 >= id2:
                    continue
                dist = np.linalg.norm(agent1.position - agent2.position)
                if dist <= communication_range:
                    p1 = agent_positions[id1]
                    p2 = agent_positions[id2]
                    cv2.line(vis, p1, p2, (100, 100, 100), 1)

        for agent_id, pos in agent_positions.items():
            cv2.circle(vis, pos, 4, (0, 255, 0), -1)
            cv2.putText(vis, str(agent_id)[:4], (pos[0] + 5, pos[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        return vis


class SwarmMetrics:
    @staticmethod
    def compute_coverage(swarm_state: SwarmState, area_bounds: Tuple[np.ndarray, np.ndarray]) -> float:
        min_bounds, max_bounds = area_bounds
        area = np.prod(max_bounds - min_bounds)

        if area == 0:
            return 0.0

        covered = set()
        for agent in swarm_state.agents.values():
            x = int((agent.position[0] - min_bounds[0]) / 1.0)
            y = int((agent.position[1] - min_bounds[1]) / 1.0)
            covered.add((x, y))

        return len(covered) / (area / 1.0)

    @staticmethod
    def compute_connectivity(swarm_state: SwarmState, comm_range: float = 10.0) -> float:
        if len(swarm_state.agents) < 2:
            return 1.0

        agent_ids = list(swarm_state.agents.keys())
        connected = 0
        total = 0

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                total += 1
                dist = np.linalg.norm(
                    swarm_state.agents[agent_ids[i]].position -
                    swarm_state.agents[agent_ids[j]].position
                )
                if dist <= comm_range:
                    connected += 1

        return connected / total if total > 0 else 0.0

    @staticmethod
    def compute_formation_quality(
        swarm_state: SwarmState,
        desired_formation: Dict[str, np.ndarray]
    ) -> float:
        if not desired_formation:
            return 0.0

        total_error = 0.0
        for agent_id, desired_pos in desired_formation.items():
            if agent_id in swarm_state.agents:
                actual_pos = swarm_state.agents[agent_id].position
                total_error += np.linalg.norm(actual_pos - desired_pos)

        return 1.0 / (1.0 + total_error / len(desired_formation)) if desired_formation else 0.0


class SwarmSimulator:
    def __init__(
        self,
        coordinator: SwarmCoordinator,
        simulation_time: float = 10.0,
        dt: float = 0.01
    ):
        self.coordinator = coordinator
        self.simulation_time = simulation_time
        self.dt = dt
        self.history: List[SwarmState] = []

    def run(self) -> List[SwarmState]:
        num_steps = int(self.simulation_time / self.dt)

        for _ in range(num_steps):
            state = self.coordinator.step(self.dt)
            self.history.append(state)

        return self.history

    def get_trajectories(self) -> Dict[str, List[np.ndarray]]:
        trajectories = defaultdict(list)
        for state in self.history:
            for agent_id, agent in state.agents.items():
                trajectories[agent_id].append(agent.position.copy())
        return dict(trajectories)


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet