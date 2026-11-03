from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math
import time
import random


class ActionType(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class RLState:
    observation: np.ndarray
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass(frozen=True, slots=True)
class RLAction:
    action: np.ndarray
    log_prob: Optional[float] = None
    value: Optional[float] = None


class Policy(ABC):
    @abstractmethod
    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        pass

    @abstractmethod
    def update(self, batch: Dict) -> Dict[str, float]:
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass


class ValueFunction(ABC):
    @abstractmethod
    def predict(self, state: np.ndarray) -> float:
        pass

    @abstractmethod
    def update(self, states: np.ndarray, returns: np.ndarray) -> float:
        pass


class ActorCriticPolicy(Policy, ValueFunction):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.actor_net = self._build_actor()
        self.critic_net = self._build_critic()
        self.actor_optimizer = None
        self.critic_optimizer = None

    def _build_actor(self):
        return None

    def _build_critic(self):
        return None

    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return np.random.uniform(-1, 1, self.action_dim)

    def update(self, batch: Dict) -> Dict[str, float]:
        return {'actor_loss': 0.0, 'critic_loss': 0.0}

    def predict(self, state: np.ndarray) -> float:
        return 0.0

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        epochs: int = 10,
        batch_size: int = 64,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.policy = ActorCriticPolicy(
            state_dim, action_dim,
            lr_actor=lr_actor, lr_critic=lr_critic,
            gamma=gamma, gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            entropy_coef=entropy_coef, value_coef=value_coef,
        )

        self.buffer = []

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        action = self.policy.act(state, deterministic)
        log_prob = 0.0
        value = 0.0
        return action, log_prob, value

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool, log_prob: float, value: float):
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': value,
        })

    def compute_gae(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        gae = 0

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[i] = gae
            returns[i] = advantages[i] + values[i]

        return advantages, returns

    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.batch_size:
            return {}

        states = np.array([t['state'] for t in self.buffer])
        actions = np.array([t['action'] for t in self.buffer])
        rewards = np.array([t['reward'] for t in self.buffer])
        next_states = np.array([t['next_state'] for t in self.buffer])
        dones = np.array([t['done'] for t in self.buffer])
        old_log_probs = np.array([t['log_prob'] for t in self.buffer])
        old_values = np.array([t['value'] for t in self.buffer])

        advantages, returns = self.compute_gae(rewards, np.append(self.values[1:], 0), dones)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            indices = np.random.permutation(len(self.buffer))
            for i in range(0, len(self.buffer), self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                self._update_step(batch_idx, states, actions, old_log_probs, returns, advantages)

        self.buffer.clear()
        return {'actor_loss': 0.0, 'critic_loss': 0.0}

    def _update_step(self, batch_idx, states, actions, old_log_probs, returns, advantages):
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

        self.replay_buffer = deque(maxlen=1000000)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return np.random.uniform(-1, 1, self.action_dim)

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.replay_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
        })

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        if len(self.replay_buffer) < batch_size:
            return {}

        batch = random.sample(self.replay_buffer, batch_size)
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

        self.replay_buffer = deque(maxlen=1000000)

    def select_action(self, state: np.ndarray, noise: float = 0.1) -> np.ndarray:
        action = np.random.uniform(-1, 1, self.action_dim)
        return np.clip(action + np.random.normal(0, noise, self.action_dim), -1, 1)

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        if len(self.replay_buffer) < batch_size:
            return {}

        self.steps += 1
        batch = random.sample(self.replay_buffer, batch_size)
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

        self.replay_buffer = deque(maxlen=1000000)

    def select_action(self, state: np.ndarray, noise: float = 0.1) -> np.ndarray:
        return np.random.uniform(-1, 1, self.action_dim)

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        if len(self.replay_buffer) < batch_size:
            return {}

        batch = random.sample(self.replay_buffer, batch_size)
        return {'actor_loss': 0.0, 'critic_loss': 0.0}


class MADDPGAgent:
    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
    ):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []

        self.replay_buffer = deque(maxlen=1000000)

    def select_actions(self, states: List[np.ndarray], noise: float = 0.1) -> List[np.ndarray]:
        return [np.random.uniform(-1, 1, self.action_dim) for _ in range(self.n_agents)]

    def store_transition(self, states: List[np.ndarray], actions: List[np.ndarray], rewards: List[float], next_states: List[np.ndarray], dones: List[bool]):
        self.replay_buffer.append((states, actions, rewards, next_states, dones))

    def update(self, batch_size: int = 1024) -> Dict[str, float]:
        if len(self.replay_buffer) < batch_size:
            return {}

        batch = random.sample(self.replay_buffer, batch_size)
        return {'actor_loss': 0.0, 'critic_loss': 0.0}


class A2CAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 7e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        n_steps: int = 5,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_steps = n_steps

        self.buffer = []

    def select_action(self, state: np.ndarray) -> int:
        probs = self._get_action_probs(state)
        return np.random.choice(self.action_dim, p=probs)

    def _get_action_probs(self, state: np.ndarray) -> np.ndarray:
        return np.ones(self.action_dim) / self.action_dim

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
        })

    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.n_steps:
            return {}

        states = [t['state'] for t in self.buffer]
        actions = [t['action'] for t in self.buffer]
        rewards = [t['reward'] for t in self.buffer]
        next_states = [t['next_state'] for t in self.buffer]
        dones = [t['done'] for t in self.buffer]

        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)

        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        self.buffer.clear()
        return {'actor_loss': 0.0, 'critic_loss': 0.0}


class REINFORCEAgent:
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

        self.replay_buffer = deque(maxlen=buffer_size)
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
        self.replay_buffer.append(Transition(
            state=state, action=np.array([action]), reward=reward,
            next_state=next_state, done=done
        ))

    def update(self) -> Dict[str, float]:
        if len(self.replay_buffer) < self.batch_size:
            return {}

        self.steps += 1
        batch = random.sample(self.replay_buffer, self.batch_size)

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


class Trainer:
    def __init__(
        self,
        env,
        agent,
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
        env,
        agents: List,
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
