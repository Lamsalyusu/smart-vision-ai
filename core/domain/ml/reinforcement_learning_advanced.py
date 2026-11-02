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


class RLAlgorithm(Enum):
    Q_LEARNING = "q_learning"
    SARSA = "sarsa"
    DQN = "dqn"
    DDQN = "ddqn"
    DUELING_DQN = "dueling_dqn"
    POLICY_GRADIENT = "policy_gradient"
    ACTOR_CRITIC = "actor_critic"
    A2C = "a2c"
    A3C = "a3c"
    PPO = "ppo"
    TRPO = "trpo"
    SAC = "sac"
    TD3 = "td3"
    DDPG = "ddpg"
    RAINBOW = "rainbow"
    IMPALA = "impala"
    R2D2 = "r2d2"
    MUESLI = "muesli"


class EnvironmentType(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MULTI_AGENT = "multi_agent"
    PARTIALLY_OBSERVABLE = "partially_observable"


@dataclass(frozen=True, slots=True)
class RLConfig:
    algorithm: RLAlgorithm = RLAlgorithm.PPO
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 100000
    batch_size: int = 256
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    max_action: float = 1.0
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass(frozen=True, slots=True)
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: Optional[float] = None
    value: Optional[float] = None


@dataclass(frozen=True, slots=True)
class Episode:
    transitions: List[Transition]
    total_reward: float
    length: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Optional[Transition]] = [None] * capacity
        self.position = 0
        self.size = 0

    def push(self, transition: Transition):
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample([t for t in self.buffer if t is not None], batch_size)

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer: List[Optional[Transition]] = [None] * capacity
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.size = 0

    def push(self, transition: Transition, priority: float = None):
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        priority = priority if priority is not None else max_priority

        self.buffer[self.position] = transition
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self) -> int:
        return self.size


class OrnsteinUhlenbeckNoise:
    def __init__(
        self,
        size: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1e-2
    ):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x_prev = np.zeros(size)

    def __call__(self) -> np.ndarray:
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * math.sqrt(self.dt) * np.random.randn(self.size)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = np.zeros(self.size)


class BaseAgent(ABC):
    def __init__(self, config: RLConfig):
        self.config = config
        self.step_count = 0

    @abstractmethod
    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        pass

    @abstractmethod
    def update(self, batch: List[Transition]) -> Dict[str, float]:
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass


class QLearningAgent(BaseAgent):
    def __init__(
        self,
        config: RLConfig,
        state_dim: int,
        action_dim: int,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        super().__init__(config)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = {}

    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        state_key = tuple(state.round(2))

        if training and random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)

        return np.argmax(self.q_table[state_key])

    def update(self, batch: List[Transition]) -> Dict[str, float]:
        for t in batch:
            state_key = tuple(t.state.round(2))
            next_state_key = tuple(t.next_state.round(2))

            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_dim)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.action_dim)

            target = t.reward
            if not t.done:
                target += self.config.gamma * np.max(self.q_table[next_state_key])

            self.q_table[state_key][t.action] = (1 - self.config.learning_rate) * \
                self.q_table[state_key][t.action] + self.config.learning_rate * target

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return {"loss": 0.0}

    def save(self, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, path: str):
        import pickle
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)


class DQNAgent(BaseAgent):
    def __init__(
        self,
        config: RLConfig,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 1000
    ):
        super().__init__(config)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.q_network = self._build_network(state_dim, action_dim, hidden_dims)
        self.target_network = self._build_network(state_dim, action_dim, hidden_dims)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.buffer = ReplayBuffer(config.buffer_size)
        self.steps = 0

    def _build_network(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        return nn.Sequential(*layers)

    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        if training and random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def update(self, batch: List[Transition]) -> Dict[str, float]:
        states = torch.FloatTensor([t.state for t in batch])
        actions = torch.LongTensor([t.action for t in batch])
        rewards = torch.FloatTensor([t.reward for t in batch])
        next_states = torch.FloatTensor([t.next_state for t in batch])
        dones = torch.FloatTensor([float(t.done) for t in batch])

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.config.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return {"loss": loss.item()}


class PPOAgent(BaseAgent):
    def __init__(
        self,
        config: RLConfig,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        clip_ratio: float = 0.2,
        epochs: int = 10,
        gae_lambda: float = 0.95
    ):
        super().__init__(config)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.gae_lambda = gae_lambda

        self.actor = self._build_actor(state_dim, action_dim, hidden_dims)
        self.critic = self._build_critic(state_dim, hidden_dims)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)

        self.buffer: List[Transition] = []

    def _build_actor(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        if action_dim > 1:
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def _build_critic(self, state_dim: int, hidden_dims: List[int]):
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_mean = self.actor(state_tensor)
            action_std = torch.ones_like(action_mean) * 0.1
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)

        if self.config.max_action < 1.0:
            action = torch.clamp(action, -self.config.max_action, self.config.max_action)

        return action.numpy()[0]

    def update(self, batch: List[Transition]) -> Dict[str, float]:
        states = torch.FloatTensor([t.state for t in batch])
        actions = torch.FloatTensor([t.action for t in batch])
        rewards = torch.FloatTensor([t.reward for t in batch])
        next_states = torch.FloatTensor([t.next_state for t in batch])
        dones = torch.FloatTensor([float(t.done) for t in batch])

        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            returns = rewards + self.config.gamma * next_values * (1 - torch.FloatTensor([float(t.done) for t in batch]))
            advantages = returns - values

        for _ in range(self.epochs):
            action_means = self.actor(states)
            action_stds = torch.ones_like(action_means) * 0.1
            dist = torch.distributions.Normal(action_means, action_stds)
            new_log_probs = dist.log_prob(actions).sum(-1)

            old_log_probs = torch.FloatTensor([t.log_prob for t in batch])
            ratio = (new_log_probs - old_log_probs).exp()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.config.entropy_coef * dist.entropy().mean()

            values_pred = self.critic(states).squeeze()
            critic_loss = F.mse_loss(values_pred, returns)

            loss = actor_loss + self.config.value_coef * critic_loss

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}


class SACAgent(BaseAgent):
    def __init__(
        self,
        config: RLConfig,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        alpha: float = 0.2,
        auto_alpha: bool = True
    ):
        super().__init__(config)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.auto_alpha = auto_alpha

        self.actor = self._build_actor(state_dim, action_dim)
        self.critic1 = self._build_critic(state_dim, action_dim)
        self.critic2 = self._build_critic(state_dim, action_dim)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=config.learning_rate)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=config.learning_rate)

        if auto_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)

        self.buffer = ReplayBuffer(config.buffer_size)

    def _build_actor(self, state_dim: int, action_dim: int):
        return nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * 2)
        )

    def _build_critic(self, state_dim: int, action_dim: int):
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            output = self.actor(state_tensor)
            mean, log_std = output.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            action = torch.tanh(action)
            log_prob -= (1 - action.pow(2) + 1e-6).log().sum(-1)

        return action.numpy()[0]

    def update(self, batch: List[Transition]) -> Dict[str, float]:
        states = torch.FloatTensor([t.state for t in batch])
        actions = torch.FloatTensor([t.action for t in batch])
        rewards = torch.FloatTensor([t.reward for t in batch])
        next_states = torch.FloatTensor([t.next_state for t in batch])
        dones = torch.FloatTensor([float(t.done) for t in batch])

        with torch.no_grad():
            next_actions, next_log_probs = self._sample_action(next_states)
            next_q1 = self.target_critic1(torch.cat([next_states, next_actions], dim=-1))
            next_q2 = self.target_critic2(torch.cat([next_states, next_actions], dim=-1))
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + self.config.gamma * (1 - dones) * next_q

        q1 = self.critic1(torch.cat([states, actions], dim=-1))
        q2 = self.critic2(torch.cat([states, actions], dim=-1))
        critic1_loss = F.mse_loss(q1, target_q.detach())
        critic2_loss = F.mse_loss(q2, target_q.detach())

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        new_actions, log_probs = self._sample_action(states)
        q1_new = self.critic1(torch.cat([states, new_actions], dim=-1))
        q2_new = self.critic2(torch.cat([states, new_actions], dim=-1))
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        return {"actor_loss": actor_loss.item(), "critic1_loss": critic1_loss.item(), "critic2_loss": critic2_loss.item()}

    def _sample_action(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.actor(states)
        mean, log_std = output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        action = torch.tanh(action)
        log_prob -= (1 - action.pow(2) + 1e-6).log().sum(-1)
        return action, log_prob


class TD3Agent(BaseAgent):
    def __init__(
        self,
        config: RLConfig,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2
    ):
        super().__init__(config)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        self.actor = self._build_actor(state_dim, action_dim)
        self.critic1 = self._build_critic(state_dim, action_dim)
        self.critic2 = self._build_critic(state_dim, action_dim)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=config.learning_rate)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=config.learning_rate)

        self.buffer = ReplayBuffer(config.buffer_size)
        self.steps = 0

    def _build_actor(self, state_dim: int, action_dim: int):
        return nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def _build_critic(self, state_dim: int, action_dim: int):
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor)

        if training:
            noise = np.random.randn(self.action_dim) * self.config.exploration_noise
            action = action + noise
            action = np.clip(action, -self.config.max_action, self.config.max_action)

        return action.numpy()[0]

    def update(self, batch: List[Transition]) -> Dict[str, float]:
        states = torch.FloatTensor([t.state for t in batch])
        actions = torch.FloatTensor([t.action for t in batch])
        rewards = torch.FloatTensor([t.reward for t in batch])
        next_states = torch.FloatTensor([t.next_state for t in batch])
        dones = torch.FloatTensor([float(t.done) for t in batch])

        with torch.no_grad():
            noise = torch.randn_like(actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.target_actor(next_states) + noise).clamp(-self.config.max_action, self.config.max_action)

            target_q1 = self.target_critic1(torch.cat([next_states, next_actions], dim=-1))
            target_q2 = self.target_critic2(torch.cat([next_states, next_actions], dim=-1))
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.config.gamma * (1 - dones) * target_q

        q1 = self.critic1(torch.cat([states, actions], dim=-1))
        q2 = self.critic2(torch.cat([states, actions], dim=-1))
        critic1_loss = F.mse_loss(q1, target_q.detach())
        critic2_loss = F.mse_loss(q2, target_q.detach())

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        self.steps += 1
        if self.steps % self.policy_delay == 0:
            actor_loss = -self.critic1(torch.cat([states, self.actor(states)], dim=-1)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

            for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

        return {"critic1_loss": critic1_loss.item(), "critic2_loss": critic2_loss.item(), "actor_loss": actor_loss.item() if self.steps % self.policy_delay == 0 else 0.0}


class DDPGAgent(BaseAgent):
    def __init__(
        self,
        config: RLConfig,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        noise: OrnsteinUhlenbeckNoise = None
    ):
        super().__init__(config)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.noise = noise or OrnsteinUhlenbeckNoise(action_dim)

        self.actor = self._build_actor(state_dim, action_dim)
        self.critic = self._build_critic(state_dim, action_dim)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)

        self.buffer = ReplayBuffer(config.buffer_size)

    def _build_actor(self, state_dim: int, action_dim: int):
        return nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def _build_critic(self, state_dim: int, action_dim: int):
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor)

        if training:
            action = action + torch.FloatTensor(self.noise())

        return action.clamp(-self.config.max_action, self.config.max_action).numpy()[0]

    def update(self, batch: List[Transition]) -> Dict[str, float]:
        states = torch.FloatTensor([t.state for t in batch])
        actions = torch.FloatTensor([t.action for t in batch])
        rewards = torch.FloatTensor([t.reward for t in batch])
        next_states = torch.FloatTensor([t.next_state for t in batch])
        dones = torch.FloatTensor([float(t.done) for t in batch])

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(torch.cat([next_states, next_actions], dim=-1))
            target_q = rewards + self.config.gamma * (1 - dones) * target_q

        q = self.critic(torch.cat([states, actions], dim=-1))
        critic_loss = F.mse_loss(q, target_q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(torch.cat([states, self.actor(states)], dim=-1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

        self.noise.reset()

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}


class RLAgentFactory:
    @staticmethod
    def create_agent(
        algorithm: RLAlgorithm,
        config: RLConfig,
        state_dim: int,
        action_dim: int,
        **kwargs
    ) -> BaseAgent:
        if algorithm == RLAlgorithm.Q_LEARNING:
            return QLearningAgent(config, state_dim, action_dim, **kwargs)
        elif algorithm == RLAlgorithm.DQN:
            return DQNAgent(config, state_dim, action_dim, **kwargs)
        elif algorithm == RLAlgorithm.PPO:
            return PPOAgent(config, state_dim, action_dim, **kwargs)
        elif algorithm == RLAlgorithm.SAC:
            return SACAgent(config, state_dim, action_dim, **kwargs)
        elif algorithm == RLAlgorithm.TD3:
            return TD3Agent(config, state_dim, action_dim, **kwargs)
        elif algorithm == RLAlgorithm.DDPG:
            return DDPGAgent(config, state_dim, action_dim, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")


class RLTrainer:
    def __init__(
        self,
        agent: BaseAgent,
        environment: Any,
        max_episodes: int = 1000,
        max_steps: int = 1000,
        eval_freq: int = 100,
        eval_episodes: int = 10
    ):
        self.agent = agent
        self.env = environment
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes

        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

    def train(self) -> Dict[str, List[float]]:
        for episode in range(self.max_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            episode_length = 0

            for step in range(self.max_steps):
                action = self.agent.act(state, training=True)
                next_state, reward, done, _ = self.env.step(action)

                transition = Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
                self.agent.update([transition])

                state = next_state
                episode_reward += reward
                episode_length += 1

                if done:
                    break

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            if episode % self.eval_freq == 0:
                self._evaluate()

        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }

    def _evaluate(self):
        eval_rewards = []
        for _ in range(self.eval_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                action = self.agent.act(state, training=False)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward

            eval_rewards.append(episode_reward)

        print(f"Evaluation: Mean Reward = {np.mean(eval_rewards):.2f}")


import torch
import torch.nn as nn
import torch.nn.functional as F
from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet