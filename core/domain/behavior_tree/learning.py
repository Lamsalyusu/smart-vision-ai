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
import copy


class LearningMethod(Enum):
    REINFORCEMENT = "reinforcement"
    IMITATION = "imitation"
    EVOLUTIONARY = "evolutionary"
    GRADIENT_BASED = "gradient_based"
    BAYESIAN = "bayesian"
    HYBRID = "hybrid"


class BTNodeType(Enum):
    SEQUENCE = "sequence"
    SELECTOR = "selector"
    PARALLEL = "parallel"
    DECORATOR = "decorator"
    ACTION = "action"
    CONDITION = "condition"


@dataclass(frozen=True, slots=True)
class BTGenotype:
    structure: List[Dict[str, Any]]
    parameters: Dict[str, float]
    fitness: float = 0.0
    age: int = 0

    def mutate(self, mutation_rate: float = 0.1) -> 'BTGenotype':
        new_structure = copy.deepcopy(self.structure)
        new_params = self.parameters.copy()

        for node in new_structure:
            if random.random() < mutation_rate:
                if 'params' in node:
                    for k, v in node['params'].items():
                        if isinstance(v, (int, float)):
                            node['params'][k] = v * (1 + random.gauss(0, 0.1))

        for k, v in new_params.items():
            if random.random() < mutation_rate:
                new_params[k] = v * (1 + random.gauss(0, 0.1))

        return BTGenotype(new_structure, new_params, 0.0, self.age + 1)

    def crossover(self, other: 'BTGenotype') -> 'BTGenotype':
        min_len = min(len(self.structure), len(other.structure))
        split = random.randint(1, min_len - 1) if min_len > 1 else 0

        new_structure = self.structure[:split] + other.structure[split:]
        new_params = {**self.parameters, **other.parameters}

        return BTGenotype(new_structure, new_params, 0.0, 0)


@dataclass(frozen=True, slots=True)
class EpisodeData:
    states: List[Any]
    actions: List[int]
    rewards: List[float]
    bt_execution: List[str]
    total_reward: float
    length: int


class BTEnvironment(ABC):
    @abstractmethod
    def reset(self) -> Any:
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        pass

    @abstractmethod
    def get_state(self) -> Any:
        pass

    @abstractmethod
    def get_valid_actions(self) -> List[int]:
        pass


class BTLearner(ABC):
    def __init__(self, method: LearningMethod):
        self.method = method
        self.population: List[BTGenotype] = []
        self.best_genotype: Optional[BTGenotype] = None
        self.generation = 0
        self.history: List[Dict] = []

    @abstractmethod
    def train(self, environment: BTEnvironment, episodes: int) -> BTGenotype:
        pass

    @abstractmethod
    def evaluate(self, genotype: BTGenotype, environment: BTEnvironment, episodes: int) -> float:
        pass


class EvolutionaryBTLearner(BTLearner):
    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elitism: int = 2,
        max_depth: int = 5
    ):
        super().__init__(LearningMethod.EVOLUTIONARY)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.max_depth = max_depth
        self.node_types = list(BTNodeType)

    def initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            genotype = self._random_genotype()
            self.population.append(genotype)

    def _random_genotype(self) -> BTGenotype:
        structure = self._generate_random_tree(self.max_depth)
        parameters = {
            'retry_count': random.randint(1, 5),
            'timeout': random.uniform(1.0, 10.0),
            'cooldown': random.uniform(0.5, 5.0)
        }
        return BTGenotype(structure, parameters, 0.0, 0)

    def _generate_random_tree(self, depth: int) -> List[Dict]:
        if depth <= 0 or random.random() < 0.3:
            return [{
                'type': random.choice([BTNodeType.ACTION, BTNodeType.CONDITION]).value,
                'id': f"node_{random.randint(1000, 9999)}",
                'params': {}
            }]

        node_type = random.choice([BTNodeType.SEQUENCE, BTNodeType.SELECTOR, BTNodeType.PARALLEL, BTNodeType.DECORATOR])
        children = []

        if node_type == BTNodeType.DECORATOR:
            children = [self._generate_random_tree(depth - 1)[0]]
        else:
            num_children = random.randint(2, 4)
            for _ in range(num_children):
                children.extend(self._generate_random_tree(depth - 1))

        return [{
            'type': node_type.value,
            'id': f"node_{random.randint(1000, 9999)}",
            'children': children,
            'params': {}
        }]

    def train(self, environment: BTEnvironment, episodes: int) -> BTGenotype:
        self.initialize_population()

        for generation in range(episodes):
            self.generation = generation

            fitnesses = []
            for genotype in self.population:
                fitness = self.evaluate(genotype, environment, 5)
                genotype.fitness = fitness
                fitnesses.append(fitness)

            self.population.sort(key=lambda x: x.fitness, reverse=True)

            if self.best_genotype is None or self.population[0].fitness > self.best_genotype.fitness:
                self.best_genotype = copy.deepcopy(self.population[0])

            self.history.append({
                'generation': generation,
                'best_fitness': self.population[0].fitness,
                'avg_fitness': np.mean(fitnesses),
                'worst_fitness': self.population[-1].fitness
            })

            new_population = self.population[:self.elitism]

            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    parent1 = self._tournament_selection()
                    parent2 = self._tournament_selection()
                    child = parent1.crossover(parent2)
                else:
                    child = copy.deepcopy(self._tournament_selection())

                child = child.mutate(self.mutation_rate)
                new_population.append(child)

            self.population = new_population

        return self.best_genotype

    def _tournament_selection(self, tournament_size: int = 3) -> BTGenotype:
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)

    def evaluate(self, genotype: BTGenotype, environment: BTEnvironment, episodes: int) -> float:
        total_reward = 0.0

        for _ in range(episodes):
            state = environment.reset()
            episode_reward = 0.0
            done = False

            while not done:
                action = self._execute_genotype(genotype, state, environment)
                state, reward, done, _ = environment.step(action)
                episode_reward += reward

            total_reward += episode_reward

        return total_reward / episodes

    def _execute_genotype(self, genotype: BTGenotype, state: Any, environment: BTEnvironment) -> int:
        if not genotype.structure:
            return 0

        return self._execute_node(genotype.structure[0], state, environment)

    def _execute_node(self, node: Dict, state: Any, environment: BTEnvironment) -> int:
        node_type = BTNodeType(node['type'])

        if node_type == BTNodeType.ACTION:
            return int(node.get('params', {}).get('action_id', 0))

        elif node_type == BTNodeType.CONDITION:
            condition_fn = node.get('params', {}).get('condition_fn')
            if condition_fn and condition_fn(state):
                if 'children' in node and node['children']:
                    return self._execute_node(node['children'][0], state, environment)
            return 0

        elif node_type == BTNodeType.SEQUENCE:
            if 'children' in node:
                for child in node['children']:
                    result = self._execute_node(child, state, environment)
                    if result == -1:
                        return -1
            return 1

        elif node_type == BTNodeType.SELECTOR:
            if 'children' in node:
                for child in node['children']:
                    result = self._execute_node(child, state, environment)
                    if result == 1:
                        return 1
            return -1

        elif node_type == BTNodeType.PARALLEL:
            if 'children' in node:
                for child in node['children']:
                    self._execute_node(child, state, environment)
            return 1

        elif node_type == BTNodeType.DECORATOR:
            if 'children' in node and node['children']:
                return self._execute_node(node['children'][0], state, environment)

        return 0


class ReinforcementBTLearner(BTLearner):
    def __init__(
        self,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        buffer_size: int = 10000,
        batch_size: int = 64
    ):
        super().__init__(LearningMethod.REINFORCEMENT)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer: List[Tuple] = []

    def train(self, environment: BTEnvironment, episodes: int) -> BTGenotype:
        best_genotype = self._create_initial_genotype()

        for episode in range(episodes):
            state = environment.reset()
            total_reward = 0.0
            done = False

            while not done:
                if random.random() < self.epsilon:
                    action = random.choice(environment.get_valid_actions())
                else:
                    action = self._get_best_action(best_genotype, state, environment)

                next_state, reward, done, _ = environment.step(action)
                self.replay_buffer.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                if len(self.replay_buffer) > self.buffer_size:
                    self.replay_buffer.pop(0)

                if len(self.replay_buffer) >= self.batch_size:
                    self._update_network(best_genotype)

            if total_reward > best_genotype.fitness:
                best_genotype.fitness = total_reward

            self.history.append({
                'episode': episode,
                'reward': total_reward,
                'epsilon': self.epsilon
            })

            self.epsilon = max(0.01, self.epsilon * 0.995)

        return best_genotype

    def _create_initial_genotype(self) -> BTGenotype:
        return BTGenotype([], {}, 0.0, 0)

    def _get_best_action(self, genotype: BTGenotype, state: Any, environment: BTEnvironment) -> int:
        return random.choice(environment.get_valid_actions())

    def _update_network(self, genotype: BTGenotype):
        pass

    def evaluate(self, genotype: BTGenotype, environment: BTEnvironment, episodes: int) -> float:
        return 0.0


class ImitationBTLearner(BTLearner):
    def __init__(
        self,
        expert_trajectories: List[EpisodeData],
        learning_rate: float = 0.001
    ):
        super().__init__(LearningMethod.IMITATION)
        self.expert_trajectories = expert_trajectories
        self.learning_rate = learning_rate

    def train(self, environment: BTEnvironment, episodes: int) -> BTGenotype:
        best_genotype = self._create_initial_genotype()

        for episode in range(episodes):
            for trajectory in self.expert_trajectories:
                self._learn_from_trajectory(best_genotype, trajectory)

        return best_genotype

    def _create_initial_genotype(self) -> BTGenotype:
        return BTGenotype([], {}, 0.0, 0)

    def _learn_from_trajectory(self, genotype: BTGenotype, trajectory: EpisodeData):
        pass

    def evaluate(self, genotype: BTGenotype, environment: BTEnvironment, episodes: int) -> float:
        return 0.0


class GradientBasedBTLearner(BTLearner):
    def __init__(
        self,
        learning_rate: float = 0.01,
        num_iterations: int = 1000
    ):
        super().__init__(LearningMethod.GRADIENT_BASED)
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def train(self, environment: BTEnvironment, episodes: int) -> BTGenotype:
        genotype = BTGenotype([], {'param_1': 1.0, 'param_2': 0.5}, 0.0, 0)

        for i in range(self.num_iterations):
            grads = self._compute_gradients(genotype, environment)
            for k, v in genotype.parameters.items():
                if k in grads:
                    genotype.parameters[k] -= self.learning_rate * grads[k]

            fitness = self.evaluate(genotype, environment, 1)
            genotype.fitness = fitness

            self.history.append({
                'iteration': i,
                'fitness': fitness,
                'parameters': genotype.parameters.copy()
            })

        return genotype

    def _compute_gradients(self, genotype: BTGenotype, environment: BTEnvironment) -> Dict[str, float]:
        grads = {}
        eps = 1e-4

        for k in genotype.parameters:
            orig = genotype.parameters[k]
            genotype.parameters[k] = orig + eps
            fitness_plus = self.evaluate(genotype, environment, 1)
            genotype.parameters[k] = orig - eps
            fitness_minus = self.evaluate(genotype, environment, 1)
            genotype.parameters[k] = orig
            grads[k] = (fitness_plus - fitness_minus) / (2 * eps)

        return grads

    def evaluate(self, genotype: BTGenotype, environment: BTEnvironment, episodes: int) -> float:
        return 0.0


class BTLearningPipeline:
    def __init__(self):
        self.learners: Dict[LearningMethod, BTLearner] = {}
        self.results: Dict[str, Any] = {}

    def add_learner(self, method: LearningMethod, learner: BTLearner):
        self.learners[method] = learner

    def train(self, method: LearningMethod, environment: BTEnvironment, episodes: int) -> BTGenotype:
        learner = self.learners.get(method)
        if not learner:
            raise ValueError(f"No learner for method: {method}")

        genotype = learner.train(environment, episodes)
        self.results[method.value] = {
            'genotype': genotype,
            'history': learner.history
        }
        return genotype

    def compare_methods(self, environment: BTEnvironment, episodes: int) -> Dict[str, float]:
        results = {}
        for method, learner in self.learners.items():
            genotype = learner.train(environment, episodes)
            results[method.value] = genotype.fitness
        return results


class BTGenotypeVisualizer:
    @staticmethod
    def to_graphviz(genotype: BTGenotype) -> str:
        lines = ["digraph BT {"]
        lines.append('  node [shape=box, style="filled,rounded"];')

        def add_nodes(nodes, parent_id=None):
            for i, node in enumerate(nodes):
                node_id = node.get('id', f"node_{i}")
                label = f"{node['type']}\\n{node.get('id', '')}"
                color = {
                    'sequence': '#4CAF50',
                    'selector': '#2196F3',
                    'parallel': '#FF9800',
                    'decorator': '#9C27B0',
                    'action': '#F44336',
                    'condition': '#00BCD4'
                }.get(node['type'], '#9E9E9E')

                lines.append(f'  "{node_id}" [label="{label}", fillcolor="{color}"];')

                if parent_id:
                    lines.append(f'  "{parent_id}" -> "{node_id}";')

                if 'children' in node:
                    add_nodes(node['children'], node_id)

        add_nodes(genotype.structure)
        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def draw_fitness_history(history: List[Dict], image_size: Tuple[int, int] = (800, 600)) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if not history:
            return vis

        generations = [h['generation'] for h in history]
        best_fitness = [h['best_fitness'] for h in history]
        avg_fitness = [h['avg_fitness'] for h in history]

        max_gen = max(generations) if generations else 1
        max_fit = max(max(best_fitness) if best_fitness else 1, max(avg_fitness) if avg_fitness else 1)

        for i in range(1, len(generations)):
            x1 = int(generations[i-1] / max_gen * (image_size[0] - 100)) + 50
            y1 = int((1 - best_fitness[i-1] / max_fit) * (image_size[1] - 100)) + 50
            x2 = int(generations[i] / max_gen * (image_size[0] - 100)) + 50
            y2 = int((1 - best_fitness[i] / max_fit) * (image_size[1] - 100)) + 50
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            x1 = int(generations[i-1] / max_gen * (image_size[0] - 100)) + 50
            y1 = int((1 - avg_fitness[i-1] / max_fit) * (image_size[1] - 100)) + 50
            x2 = int(generations[i] / max_gen * (image_size[0] - 100)) + 50
            y2 = int((1 - avg_fitness[i] / max_fit) * (image_size[1] - 100)) + 50
            cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 1)

        return vis


class BTGenotypeSerializer:
    @staticmethod
    def save(genotype: BTGenotype, path: str):
        import json
        data = {
            'structure': genotype.structure,
            'parameters': genotype.parameters,
            'fitness': genotype.fitness,
            'age': genotype.age
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: str) -> BTGenotype:
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        return BTGenotype(
            structure=data['structure'],
            parameters=data['parameters'],
            fitness=data['fitness'],
            age=data['age']
        )


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet