from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
import random
from collections import defaultdict, deque


class StateType(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    HYBRID = "hybrid"


class ActionSpaceType(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MULTI_DISCRETE = "multi_discrete"


@dataclass(frozen=True, slots=True)
class StateSpace:
    shape: Tuple[int, ...]
    low: Optional[np.ndarray] = None
    high: Optional[np.ndarray] = None
    n: Optional[int] = None


@dataclass(frozen=True, slots=True)
class ActionSpace:
    shape: Tuple[int, ...]
    low: Optional[np.ndarray] = None
    high: Optional[np.ndarray] = None
    n: Optional[int] = None


@dataclass(frozen=True, slots=True)
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class BaseEnvironment(ABC):
    def __init__(self):
        self.action_space: Optional[ActionSpace] = None
        self.observation_space: Optional[StateSpace] = None
        self.state: Optional[np.ndarray] = None
        self.step_count = 0
        self.episode_count = 0

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        pass

    @abstractmethod
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def close(self):
        pass

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)


class GridWorldEnv:
    def __init__(
        self,
        size: int = 10,
        num_obstacles: int = 10,
        max_steps: int = 100
    ):
        self.size = size
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps

        self.action_space = type('obj', (object,), {'n': 4})()
        self.observation_space = type('obj', (object,), {'n': size * size})()

        self.obstacles = set()
        self.agent_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        self._generate_obstacles()

    def _generate_obstacles(self):
        while len(self.obstacles) < self.num_obstacles:
            pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if pos != self.agent_pos and pos != self.goal_pos:
                self.obstacles.add(pos)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.agent_pos = (0, 0)
        self.step_count = 0
        return np.array([self.agent_pos[0] * self.size + self.agent_pos[1]], dtype=np.int64), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1

        action = int(action.item()) if hasattr(action, 'item') else int(action)
        x, y = self.agent_pos

        if action == 0:
            y = max(0, y - 1)
        elif action == 1:
            y = min(self.size - 1, y + 1)
        elif action == 2:
            x = max(0, x - 1)
        elif action == 3:
            x = min(self.size - 1, x + 1)

        new_pos = (x, y)

        if new_pos in self.obstacles:
            reward = -1.0
            new_pos = self.agent_pos
        elif new_pos == self.goal_pos:
            reward = 10.0
        else:
            reward = -0.01

        self.agent_pos = new_pos
        done = new_pos == self.goal_pos
        truncated = self.step_count >= self.max_steps

        obs = np.array([self.agent_pos[0] * self.size + self.agent_pos[1]], dtype=np.int64)

        return obs, reward, done, truncated, {'pos': self.agent_pos, 'step': self.step_count}

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        grid = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        for obs in self.obstacles:
            grid[obs[1], obs[0]] = [128, 128, 128]

        grid[self.goal_pos[1], self.goal_pos[0]] = [0, 255, 0]
        grid[self.agent_pos[1], self.agent_pos[0]] = [255, 0, 0]

        if mode == "rgb_array":
            return grid
        elif mode == "human":
            print("\n".join("".join(
                "A" if (x, y) == self.agent_pos else
                "G" if (x, y) == self.goal_pos else
                "#" if (x, y) in self.obstacles else "."
                for x in range(self.size))
                for y in range(self.size)))
        return None

    def close(self):
        pass


class ContinuousControlEnv:
    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        max_steps: int = 200
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps

        self.action_space = type('obj', (object,), {
            'shape': (action_dim,),
            'low': np.array([-1.0] * action_dim, dtype=np.float32),
            'high': np.array([1.0] * action_dim, dtype=np.float32)
        })()
        self.observation_space = type('obj', (object,), {
            'shape': (state_dim,),
            'low': np.array([-np.inf] * state_dim, dtype=np.float32),
            'high': np.array([np.inf] * state_dim, dtype=np.float32)
        })()

        self.state = np.zeros(state_dim, dtype=np.float32)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.state = np.random.uniform(-0.1, 0.1, self.state_dim).astype(np.float32)
        self.step_count = 0
        self.episode_count += 1
        return self.state.copy(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1

        action = np.clip(action, -1.0, 1.0)
        self.state[:2] += action * 0.1
        self.state[2:] += np.random.normal(0, 0.01, 2)

        reward = -np.sum(self.state[:2]**2) - 0.1 * np.sum(action**2)

        done = np.linalg.norm(self.state[:2]) < 0.05
        truncated = self.step_count >= self.max_steps

        return self.state.copy(), reward, done, truncated, {'state': self.state.copy()}

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        if mode == "rgb_array":
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            x = int((self.state[0] + 1) * 100)
            y = int((self.state[1] + 1) * 100)
            if 0 <= x < 200 and 0 <= y < 200:
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            return img
        return None

    def close(self):
        pass


class MultiAgentEnv:
    def __init__(
        self,
        num_agents: int = 2,
        state_dim: int = 4,
        action_dim: int = 2,
        max_steps: int = 100
    ):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps

        self.action_space = type('obj', (object,), {'n': num_agents * action_dim})()
        self.observation_space = type('obj', (object,), {
            'spaces': {
                f'agent_{i}': type('obj', (object,), {
                    'shape': (state_dim,),
                    'low': np.array([-np.inf] * state_dim, dtype=np.float32),
                    'high': np.array([np.inf] * state_dim, dtype=np.float32)
                })() for i in range(num_agents)
            }
        })()

        self.agent_states = [np.zeros(state_dim, dtype=np.float32) for _ in range(num_agents)]

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.agent_states = [
            np.random.uniform(-1, 1, self.state_dim).astype(np.float32)
            for _ in range(self.num_agents)
        ]
        self.step_count = 0
        self.episode_count += 1

        obs = {f'agent_{i}': self.agent_states[i].copy() for i in range(self.num_agents)}
        return obs, {}

    def step(self, action: Dict[str, np.ndarray]) -> Dict[str, Tuple[np.ndarray, float, bool, bool, Dict]]:
        self.step_count += 1

        rewards = {}
        dones = {}
        truncated = self.step_count >= self.max_steps
        infos = {}

        for i in range(self.num_agents):
            agent_key = f'agent_{i}'
            agent_action = action.get(agent_key, np.zeros(self.action_dim))
            self.agent_states[i] += np.array(agent_action) * 0.1

            reward = -np.sum(self.agent_states[i]**2)
            done = np.linalg.norm(self.agent_states[i]) < 0.1

            rewards[agent_key] = reward
            dones[agent_key] = done

        obs = {f'agent_{i}': self.agent_states[i].copy() for i in range(self.num_agents)}

        return {
            agent_key: (obs[agent_key], rewards[agent_key], dones[agent_key], truncated, {})
            for agent_key in obs
        }

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        return None

    def close(self):
        pass


class VectorizedEnv:
    def __init__(self, env_fns: List[Callable[[], 'BaseEnvironment']]):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space

    def reset(self, seed: Optional[int] = None) -> Tuple[List[np.ndarray], List[Dict]]:
        if seed is not None:
            for i, env in enumerate(self.envs):
                env.reset(seed + i)

        observations = []
        infos = []
        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)

        return observations, infos

    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], List[bool], List[bool], List[Dict]]:
        observations = []
        rewards = []
        dones = []
        truncated = []
        infos = []

        for env, action in zip(self.envs, actions):
            result = env.step(action)
            observations.append(result.observation)
            rewards.append(result.reward)
            dones.append(result.done)
            truncated.append(result.truncated)
            infos.append(result.info)

        return observations, rewards, dones, truncated, infos

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        return None

    def close(self):
        for env in self.envs:
            env.close()


class EnvWrapper:
    def __init__(self, env: 'BaseEnvironment'):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        return self.env.reset(seed)

    def step(self, action: np.ndarray):
        return self.env.step(action)

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        return self.env.render(mode)

    def close(self):
        self.env.close()


class NormalizeObservation:
    def __init__(self, env: 'BaseEnvironment', mean: np.ndarray = None, std: np.ndarray = None):
        self.env = env
        self.mean = mean if mean is not None else np.zeros(env.observation_space.shape)
        self.std = std if std is not None else np.ones(env.observation_space.shape)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(seed)
        return (obs - self.mean) / (self.std + 1e-8), info

    def step(self, action: np.ndarray):
        result = self.env.step(action)
        obs = (result.observation - self.mean) / (self.std + 1e-8)
        return EnvStepResult(obs, result.reward, result.done, result.truncated, result.info)


class NormalizeReward:
    def __init__(self, env: 'BaseEnvironment', gamma: float = 0.99, epsilon: float = 1e-8):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.returns = 0.0
        self.ret_sum = 0.0
        self.count = 0

    def step(self, action: np.ndarray):
        result = self.env.step(action)

        self.returns = self.returns * self.gamma + result.reward
        self.ret_sum += self.returns
        self.count += 1

        mean = self.ret_sum / self.count if self.count > 0 else 0
        var = self.epsilon
        normalized_reward = result.reward / math.sqrt(var + 1e-8)

        return EnvStepResult(
            result.observation,
            normalized_reward,
            result.done,
            result.truncated,
            result.info
        )


class FrameStack:
    def __init__(self, env: 'BaseEnvironment', num_stack: int = 4):
        self.env = env
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        old_shape = env.observation_space.shape
        self.observation_space = type('obj', (object,), {
            'shape': (num_stack,) + old_shape,
            'dtype': env.observation_space.dtype
        })()

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(seed)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return np.stack(self.frames, axis=0), info

    def step(self, action: np.ndarray):
        result = self.env.step(action)
        self.frames.append(result.observation)
        obs = np.stack(self.frames, axis=0)
        return EnvStepResult(obs, result.reward, result.done, result.truncated, result.info)


class TimeLimit:
    def __init__(self, env: 'BaseEnvironment', max_episode_steps: int):
        self.env = env
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        self.step_count = 0
        return self.env.reset(seed)

    def step(self, action: np.ndarray):
        self.step_count += 1
        result = self.env.step(action)

        if self.step_count >= self.max_episode_steps:
            return EnvStepResult(
                result.observation,
                result.reward,
                True,
                True,
                result.info
            )

        return result


class Monitor:
    def __init__(self, env: 'BaseEnvironment', log_dir: str = None):
        self.env = env
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = 0.0
        self.current_length = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        self.current_rewards = 0.0
        self.current_length = 0
        return self.env.reset(seed)

    def step(self, action: np.ndarray):
        result = self.env.step(action)
        self.current_rewards += result.reward
        self.current_length += 1

        if result.done or result.truncated:
            self.episode_rewards.append(self.current_rewards)
            self.episode_lengths.append(self.current_length)
            self.current_rewards = 0.0
            self.current_length = 0

        return result


class EvaluationEnv:
    def __init__(self, env_fn: Callable[[], 'BaseEnvironment'], num_episodes: int = 10):
        self.env_fn = env_fn
        self.num_episodes = num_episodes

    def evaluate(self, policy: Callable[[np.ndarray], np.ndarray]) -> Dict[str, float]:
        rewards = []
        lengths = []

        for _ in range(self.num_episodes):
            env = self.env_fn()
            obs, _ = env.reset()
            total_reward = 0.0
            length = 0

            while True:
                action = policy(obs)
                result = env.step(action)
                obs = result.observation
                total_reward += result.reward
                length += 1

                if result.done or result.truncated:
                    break

            rewards.append(total_reward)
            lengths.append(length)
            env.close()

        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths)
        }


import cv2
from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet