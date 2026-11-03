from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import random
import math


class AlgorithmType(Enum):
    Q_LEARNING = "q_learning"
    SARSA = "sarsa"
    DQN = "dqn"
    DOUBLE_DQN = "double_dqn"
    DUELING_DQN = "dueling_dqn"
    POLICY_GRADIENT = "policy_gradient"
    REINFORCE = "reinforce"
    ACTOR_CRITIC = "actor_critic"
    A2C = "a2c"
    A3C = "a3c"
    PPO = "ppo"
    TRPO = "trpo"
    SAC = "sac"
    TD3 = "td3"
    DDPG = "ddpg"
    MADDPG = "maddpg"


@dataclass(frozen=True, slots=True)
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Episode:
    transitions: List[Transition]
    total_reward: float
    length: int
    info: Dict[str, Any] = field(default_factory=dict)


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer: List[Transition] = []
        self.priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def push(self, transition: Transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])

        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return len(self.buffer)


class Environment(ABC):
    @abstractmethod
    def reset(self) -> np.ndarray:
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        pass

    @abstractmethod
    def render(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Tuple:
        pass

    @property
    @abstractmethod
    def action_space(self) -> Tuple:
        pass


class GridWorld(Environment):
    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height
        self.start = (0, 0)
        self.goal = (width - 1, height - 1)
        self.obstacles = set()
        self.current_pos = self.start

    def reset(self) -> np.ndarray:
        self.current_pos = self.start
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        x, y = self.current_pos
        dx, dy = int(action[0]), int(action[1])

        new_x, new_y = x + dx, y + dy

        if 0 <= new_x < self.width and 0 <= new_y < self.height and (new_x, new_y) not in self.obstacles:
            self.current_pos = (new_x, new_y)

        reward = -1.0
        if self.current_pos == self.goal:
            reward = 100.0

        done = self.current_pos == self.goal
        return self._get_obs(), reward, done, {}

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.height, self.width, 3))
        obs[self.current_pos[1], self.current_pos[0]] = [1, 0, 0]
        obs[self.goal[1], self.goal[0]] = [0, 1, 0]
        for ox, oy in self.obstacles:
            obs[oy, ox] = [0, 0, 1]
        return obs

    def render(self) -> np.ndarray:
        return self._get_obs()

    @property
    def observation_space(self) -> Tuple:
        return (self.height, self.width, 3)

    @property
    def action_space(self) -> Tuple:
        return (4,)


class ContinuousControlEnv(Environment):
    def __init__(self, state_dim: int = 4, action_dim: int = 2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state = np.zeros(state_dim)
        self.max_steps = 200
        self.current_step = 0

    def reset(self) -> np.ndarray:
        self.state = np.random.uniform(-0.1, 0.1, self.state_dim)
        self.current_step = 0
        return self.state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        self.current_step += 1

        self.state += action * 0.1 + np.random.normal(0, 0.01, self.state_dim)

        reward = -np.sum(self.state**2) - 0.1 * np.sum(action**2)

        done = self.current_step >= self.max_steps or np.any(np.abs(self.state) > 10)

        return self.state, reward, done, {}

    def render(self) -> np.ndarray:
        return np.array([])

    @property
    def observation_space(self) -> Tuple:
        return (self.state_dim,)

    @property
    def action_space(self) -> Tuple:
        return (self.action_dim,)


class Policy(ABC):
    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        pass

    @abstractmethod
    def update(self, batch: List[Transition]) -> Dict[str, float]:
        pass


class EpsilonGreedyPolicy(Policy):
    def __init__(
        self,
        q_network: Any,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        action_space: int = 4,
    ):
        self.q_network = q_network
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.action_space = action_space

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        if training and random.random() < self.epsilon:
            return np.array([random.randint(0, self.action_space - 1)])

        q_values = self.q_network.predict(state)
        return np.array([np.argmax(q_values)])

    def update(self, batch: List[Transition]) -> Dict[str, float]:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return {'epsilon': self.epsilon}


class GaussianPolicy(Policy):
    def __init__(
        self,
        actor_network: Any,
        action_dim: int,
        std: float = 0.1,
    ):
        self.actor_network = actor_network
        self.action_dim = action_dim
        self.std = std

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        mean = self.actor_network.predict(state)
        if training:
            action = np.random.normal(mean, self.std)
        else:
            action = mean
        return np.clip(action, -1, 1)

    def update(self, batch: List[Transition]) -> Dict[str, float]:
        return {}


class QLearningAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(action_dim))

    def select_action(self, state: np.ndarray) -> int:
        state_key = tuple(state.astype(int))
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return int(np.argmax(self.q_table[state_key]))

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        state_key = tuple(state.astype(int))
        next_key = tuple(next_state.astype(int))

        best_next = np.max(self.q_table[next_key]) if not done else 0
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state_key][action]

        self.q_table[state_key][action] += self.lr * td_error


class SARSAAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(action_dim))

    def select_action(self, state: np.ndarray) -> int:
        state_key = tuple(state.astype(int))
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return int(np.argmax(self.q_table[state_key]))

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, next_action: int, done: bool):
        state_key = tuple(state.astype(int))
        next_key = tuple(next_state.astype(int))

        next_q = self.q_table[next_key][next_action] if not done else 0
        td_target = reward + self.gamma * next_q
        td_error = td_target - self.q_table[state_key][action]

        self.q_table[state_key][action] += self.lr * td_error


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update: int = 1000,
        double_dqn: bool = False,
        dueling: bool = False,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.double_dqn = double_dqn
        self.dueling = dueling

        self.replay_buffer = ReplayBuffer(100000)
        self.steps = 0

        self._build_networks(learning_rate)

    def _build_networks(self, lr: float):
        pass

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        q_values = self._predict(state)
        return int(np.argmax(q_values))

    def _predict(self, state: np.ndarray) -> np.ndarray:
        return np.random.rand(self.action_dim)

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.replay_buffer.push(Transition(
            state=state, action=np.array([action]), reward=reward,
            next_state=next_state, done=done
        ))

    def update(self) -> Dict[str, float]:
        if len(self.replay_buffer) < self.batch_size:
            return {}

        self.steps += 1
        batch = self.replay_buffer.sample(self.batch_size)

        states = np.array([t.state for t in batch])
        actions = np.array([t.action[0] for t in batch])
        rewards = np.array([t.reward for t in batch])
        next_states = np.array([t.next_state for t in batch])
        dones = np.array([t.done for t in batch])

        q_values = self._predict_batch(states)
        next_q_values = self._predict_batch(next_states)

        if self.double_dqn:
            next_actions = np.argmax(self._predict_batch(next_states), axis=1)
            target_q = rewards + self.gamma * (1 - dones) * next_q_values[np.arange(self.batch_size), next_actions]
        else:
            target_q = rewards + self.gamma * (1 - dones) * np.max(next_q_values, axis=1)

        q_values[np.arange(self.batch_size), actions] = target_q

        loss = self._train_step(states, q_values)

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        if self.steps % self.target_update == 0:
            self._update_target_network()

        return {'loss': loss, 'epsilon': self.epsilon}

    def _predict_batch(self, states: np.ndarray) -> np.ndarray:
        return np.random.rand(len(states), self.action_dim)

    def _train_step(self, states: np.ndarray, targets: np.ndarray) -> float:
        return 0.0

    def _update_target_network(self):
        pass


class PolicyGradientAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        baseline: bool = True,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.baseline = baseline
        self.episodes: List[Episode] = []

    def select_action(self, state: np.ndarray) -> int:
        probs = self._get_action_probs(state)
        return np.random.choice(self.action_dim, p=probs)

    def _get_action_probs(self, state: np.ndarray) -> np.ndarray:
        return np.ones(self.action_dim) / self.action_dim

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        pass

    def finish_episode(self) -> Dict[str, float]:
        return {'loss': 0.0}

    def _compute_returns(self, rewards: List[float]) -> List[float]:
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns


class ActorCriticAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef

    def select_action(self, state: np.ndarray) -> int:
        probs = self._get_action_probs(state)
        return np.random.choice(self.action_dim, p=probs)

    def _get_action_probs(self, state: np.ndarray) -> np.ndarray:
        return np.ones(self.action_dim) / self.action_dim

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> Dict[str, float]:
        return {'actor_loss': 0.0, 'critic_loss': 0.0}


class A2CAgent(ActorCriticAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        n_steps: int = 5,
    ):
        super().__init__(state_dim, action_dim, lr_actor, lr_critic, gamma, 0.01)
        self.n_steps = n_steps
        self.buffer = []

    def select_action(self, state: np.ndarray) -> int:
        return super().select_action(state)

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> Dict[str, float]:
        self.buffer.append((state, action, reward, next_state, done))

        if len(self.buffer) >= self.n_steps or done:
            return self._update_buffer()

        return {}

    def _update_buffer(self) -> Dict[str, float]:
        states, actions, rewards, next_states, dones = zip(*self.buffer)

        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)

        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        self.buffer.clear()
        return {'actor_loss': 0.0, 'critic_loss': 0.0}


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        epochs: int = 10,
        batch_size: int = 64,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.buffer = []

    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        probs = self._get_action_probs(state)
        action = np.random.choice(self.action_dim, p=probs)
        log_prob = np.log(probs[action] + 1e-8)
        return action, log_prob

    def _get_action_probs(self, state: np.ndarray) -> np.ndarray:
        return np.ones(self.action_dim) / self.action_dim

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, log_prob: float, value: float):
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': value,
        })

    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.batch_size:
            return {}

        states = np.array([t['state'] for t in self.buffer])
        actions = np.array([t['action'] for t in self.buffer])
        old_log_probs = np.array([t['log_prob'] for t in self.buffer])
        rewards = np.array([t['reward'] for t in self.buffer])
        values = np.array([t['value'] for t in self.buffer])
        dones = np.array([t['done'] for t in self.buffer])

        returns = self._compute_returns(rewards, dones)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            indices = np.random.permutation(len(self.buffer))
            for i in range(0, len(self.buffer), self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                self._update_step(batch_idx, states, actions, old_log_probs, returns, advantages)

        self.buffer.clear()
        return {'actor_loss': 0.0, 'critic_loss': 0.0}

    def _compute_returns(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        returns = np.zeros_like(rewards)
        R = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R * (1 - dones[i])
            returns[i] = R
        return returns

    def _update_step(self, batch_idx: np.ndarray, states: np.ndarray, actions: np.ndarray, old_log_probs: np.ndarray, returns: np.ndarray, advantages: np.ndarray):
        pass


class SACAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        target_entropy: float = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_entropy = target_entropy if target_entropy is not None else -action_dim

        self.replay_buffer = ReplayBuffer(1000000)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return np.random.uniform(-1, 1, self.action_dim)

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.replay_buffer.push(Transition(state, action, reward, next_state, done))

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        if len(self.replay_buffer) < batch_size:
            return {}

        batch = self.replay_buffer.sample(batch_size)
        return {'actor_loss': 0.0, 'critic_loss': 0.0, 'alpha_loss': 0.0}


class TD3Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.steps = 0

        self.replay_buffer = ReplayBuffer(1000000)

    def select_action(self, state: np.ndarray, noise: float = 0.1) -> np.ndarray:
        action = np.random.uniform(-1, 1, self.action_dim)
        return np.clip(action + np.random.normal(0, noise, self.action_dim), -1, 1)

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.replay_buffer.push(Transition(state, action, reward, next_state, done))

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        if len(self.replay_buffer) < batch_size:
            return {}

        self.steps += 1
        batch = self.replay_buffer.sample(batch_size)
        return {'actor_loss': 0.0, 'critic_loss': 0.0}


class DDPGAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.001,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        self.replay_buffer = ReplayBuffer(1000000)

    def select_action(self, state: np.ndarray, noise: float = 0.1) -> np.ndarray:
        return np.random.uniform(-1, 1, self.action_dim)

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.replay_buffer.push(Transition(state, action, reward, next_state, done))

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        if len(self.replay_buffer) < batch_size:
            return {}

        batch = self.replay_buffer.sample(batch_size)
        return {'actor_loss': 0.0, 'critic_loss': 0.0}


class Trainer:
    def __init__(
        self,
        env: Environment,
        agent: Any,
        max_episodes: int = 1000,
        max_steps: int = 1000,
    ):
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps = max_steps

    def train(self) -> List[float]:
        episode_rewards = []

        for episode in range(self.max_episodes):
            state = self.env.reset()
            episode_reward = 0

            for step in range(self.max_steps):
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                if hasattr(self.agent, 'store_transition'):
                    self.agent.store_transition(state, action, reward, next_state, done)
                elif hasattr(self.agent, 'update'):
                    self.agent.update(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward

                if done:
                    break

            if hasattr(self.agent, 'update'):
                self.agent.update()

            episode_rewards.append(episode_reward)

            if episode % 100 == 0:
                print(f"Episode {episode}: Reward = {episode_reward:.2f}")

        return episode_rewards


class MultiAgentTrainer:
    def __init__(
        self,
        env: Environment,
        agents: List[Any],
        max_episodes: int = 1000,
    ):
        self.env = env
        self.agents = agents
        self.max_episodes = max_episodes

    def train(self) -> List[List[float]]:
        all_rewards = [[] for _ in self.agents]

        for episode in range(self.max_episodes):
            state = self.env.reset()
            episode_rewards = [0] * len(self.agents)

            done = False
            while not done:
                actions = []
                for i, agent in enumerate(self.agents):
                    action = agent.select_action(state)
                    actions.append(action)

                next_state, rewards, done, _ = self.env.step(np.array(actions))

                for i, agent in enumerate(self.agents):
                    if hasattr(agent, 'store_transition'):
                        agent.store_transition(state, actions[i], rewards[i], next_state, done)

                state = next_state
                for i in range(len(self.agents)):
                    episode_rewards[i] += rewards[i]

            for agent in self.agents:
                if hasattr(agent, 'update'):
                    agent.update()

            for i in range(len(self.agents)):
                all_rewards[i].append(episode_rewards[i])

        return all_rewards
