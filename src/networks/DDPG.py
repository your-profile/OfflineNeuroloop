# from src.networks import utils_networks as utils
import torch
import random
import torch.nn as nn
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
from torch import from_numpy
import matplotlib.pyplot as plt
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, n_goals, n_hidden1=64, n_hidden2=64, n_hidden3=64, initial_w=3e-3):
        self.n_states = n_states[0]
        self.n_actions = n_actions
        self.n_goals = n_goals
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3
        self.initial_w = initial_w
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(in_features=self.n_states + self.n_goals, out_features=self.n_hidden1)
        self.fc2 = nn.Linear(in_features=self.n_hidden1, out_features=self.n_hidden2)
        self.fc3 = nn.Linear(in_features=self.n_hidden2, out_features=self.n_hidden3)
        self.output = nn.Linear(in_features=self.n_hidden3, out_features=self.n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.output(x))


class Critic(nn.Module):
    def __init__(self, n_states, n_goals, n_hidden1=64, n_hidden2=64, n_hidden3=64, initial_w=3e-3, action_size=4):
        self.n_states = n_states[0]
        self.n_goals = n_goals
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3
        self.initial_w = initial_w
        self.action_size = action_size
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(in_features=self.n_states + self.n_goals + self.action_size, out_features=self.n_hidden1)
        self.fc2 = nn.Linear(in_features=self.n_hidden1, out_features=self.n_hidden2)
        self.fc3 = nn.Linear(in_features=self.n_hidden2, out_features=self.n_hidden3)
        self.output = nn.Linear(in_features=self.n_hidden3, out_features=1)

    def forward(self, x, a):
        x = F.relu(self.fc1(torch.cat([x, a], dim=-1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.output(x)


import threading


class Normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)
        self.lock = threading.Lock()

    def update(self, v):
        v = v.reshape(-1, self.size)
        with self.lock:
            self.local_sum += v.sum(axis=0)
            self.local_sumsq += (np.square(v)).sum(axis=0)
            self.local_count[0] += v.shape[0]

    def sync(self, local_sum, local_sumsq, local_count):
        local_sum[...] = self._mpi_average(local_sum)
        local_sumsq[...] = self._mpi_average(local_sumsq)
        local_count[...] = self._mpi_average(local_count)
        return local_sum, local_sumsq, local_count

    def recompute_stats(self):
        with self.lock:
            local_count = self.local_count.copy()
            local_sum = self.local_sum.copy()
            local_sumsq = self.local_sumsq.copy()
            self.local_count[...] = 0
            self.local_sum[...] = 0
            self.local_sumsq[...] = 0

        self.total_sum += local_sum
        self.total_sumsq += local_sumsq
        self.total_count += local_count
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(
            self.total_sum / self.total_count)))

    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v - self.mean) / self.std, -clip_range, clip_range)


class MemoryBuffer:
    def __init__(self, size, k_future=4, env=None):
        self.buffer = deque(maxlen=size)
        self.len = 0
        self.len_transition = 0
        self.maxSize = size
        self.env = env
        self.future_p = 1 - (1. / (1 + k_future))

    def _collect(self, ep_indices, time_indices):
        """Extract transition arrays for given (episode, timestep) index pairs."""
        state      = np.float32([self.buffer[ep]["state"][t]              for ep, t in zip(ep_indices, time_indices)])
        action     = np.float32([self.buffer[ep]["action"][t]             for ep, t in zip(ep_indices, time_indices)])
        reward     = np.float32([self.buffer[ep]["reward"][t]             for ep, t in zip(ep_indices, time_indices)])
        next_state = np.float32([self.buffer[ep]["next_state"][t]         for ep, t in zip(ep_indices, time_indices)])
        desired_goal      = np.float32([self.buffer[ep]["desired_goal"][t]      for ep, t in zip(ep_indices, time_indices)])
        done              = np.float32([self.buffer[ep]["done"][t]              for ep, t in zip(ep_indices, time_indices)])
        next_achieved_goal = np.float32([self.buffer[ep]["next_achieved_goal"][t] for ep, t in zip(ep_indices, time_indices)])
        return state, action, reward, next_state, desired_goal, done, next_achieved_goal

    def _her_relabel(self, ep_indices, time_indices, desired_goal, next_achieved_goal):
        """Apply HER future-goal relabeling in-place; recompute reward/done if env set."""
        batch_size = len(ep_indices)
        her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)[0]
        future_offset = np.array([
            int(np.random.uniform() * (len(self.buffer[ep]["next_state"]) - time_indices[i] - 1))
            for i, ep in enumerate(ep_indices)
        ], dtype=np.int64)
        future_timesteps = (time_indices + 1 + future_offset)[her_indices]
        desired_goal[her_indices] = np.float32([
            self.buffer[ep_indices[her_indices[i]]]["achieved_goal"][future_timesteps[i]]
            for i in range(len(her_indices))
        ])
        reward = done = None
        if self.env is not None:
            reward = self.env.compute_reward(next_achieved_goal, desired_goal, None).astype(np.float32)
            done   = (reward == 0.0).astype(np.float32)
        return desired_goal, reward, done

    def _random_indices(self, batch_size):
        """Uniform random (episode, timestep) pairs — standard ER behaviour."""
        ep_indices = np.random.randint(0, len(self.buffer), batch_size)
        time_indices = np.array([
            np.random.randint(0, len(self.buffer[ep]["next_state"]) - 1)
            for ep in ep_indices
        ], dtype=np.int64)
        return ep_indices, time_indices

    def sample(self, batch_size):
        ep_indices, time_indices = self._random_indices(batch_size)
        state, action, reward, next_state, desired_goal, done, next_achieved_goal = \
            self._collect(ep_indices, time_indices)
        desired_goal, her_reward, her_done = self._her_relabel(
            ep_indices, time_indices, desired_goal, next_achieved_goal)
        if her_reward is not None:
            reward, done = her_reward, her_done
        # Return flat transition indices so callers can update priorities if needed
        return state, action, reward, next_state, desired_goal, done, list(zip(ep_indices, time_indices))

    def add_episode(self, episode_batch):
        self.buffer.append(episode_batch)
        self.len = min(self.len + 1, self.maxSize)
        self.len_transition += len(episode_batch["state"])


class PrioritizedMemoryBuffer(MemoryBuffer):
    """
    Episode-level HER buffer with transition-level priority sampling.

    Priority logic mirrors the DQN PrioritizedReplayBuffer:
      - Priorities are stored per (episode, timestep) transition.
      - If all priorities are identical (including all 0.0), sampling falls
        back to uniform — identical behaviour to plain MemoryBuffer / vanilla ER.
      - Call update_priorities(indices, td_errors) after each train step to
        keep priorities current.

    Parameters
    ----------
    alpha : float
        Priority exponent. 0 = uniform, 1 = full prioritization.
    eps : float
        Small floor added to |TD error| to prevent zero priority after updates.
    """
    def __init__(self, size, k_future=4, env=None, alpha=0.6, eps=1e-6):
        super().__init__(size, k_future, env)
        self.alpha = alpha
        self.eps   = eps
        # Flat dict: (ep_idx_in_deque, timestep) -> raw priority
        # We track by deque position; positions shift when old episodes evict,
        # so we key by the episode object id instead for correctness.
        self._priorities: dict = {}   # key: (id(episode), t) -> float
        self._default_priority = 1.0  # assigned to brand-new unseen transitions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _priority_for(self, ep_idx, t):
        ep_id = id(self.buffer[ep_idx])
        return self._priorities.get((ep_id, t), self._default_priority)

    def _build_prob_table(self, ep_indices, time_indices):
        """
        Compute sampling probabilities for a candidate set of (ep, t) pairs.
        Returns probs array (or None if uniform fallback should be used).
        """
        raw = np.array([self._priority_for(ep, t)
                        for ep, t in zip(ep_indices, time_indices)], dtype=np.float32)
        scaled = np.abs(raw) ** self.alpha
        total  = scaled.sum()
        # Uniform fallback: all priorities identical (covers all-0.0 case too)
        if total == 0 or np.allclose(scaled, scaled[0]):
            return None
        return scaled / total

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_episode(self, episode_batch):
        # If buffer is full the deque will evict the oldest episode; clean its keys
        if len(self.buffer) == self.maxSize:
            evicted = self.buffer[0]
            evicted_id = id(evicted)
            keys_to_drop = [k for k in self._priorities if k[0] == evicted_id]
            for k in keys_to_drop:
                del self._priorities[k]
        super().add_episode(episode_batch)

    def sample(self, batch_size):
        """
        Sample batch_size transitions.

        Strategy:
          1. Draw a large candidate pool (~10x) of random (ep, t) pairs.
          2. Compute priorities over the pool.
          3. Sample batch_size from the pool with those probabilities
             (or uniformly if all priorities are equal).

        Returns the same tuple as MemoryBuffer.sample() plus transition indices
        for priority updates.
        """
        n_eps = len(self.buffer)
        pool_size = min(batch_size * 10, self.len_transition)
        pool_size = max(pool_size, batch_size)  # guard tiny buffers

        # Build candidate pool
        pool_ep = np.random.randint(0, n_eps, pool_size)
        pool_t  = np.array([
            np.random.randint(0, len(self.buffer[ep]["next_state"]) - 1)
            for ep in pool_ep
        ], dtype=np.int64)

        probs = self._build_prob_table(pool_ep, pool_t)

        if probs is None:
            # Uniform fallback — identical to vanilla ER
            chosen = np.random.choice(pool_size, batch_size, replace=False)
        else:
            chosen = np.random.choice(pool_size, batch_size, replace=False, p=probs)

        ep_indices   = pool_ep[chosen]
        time_indices = pool_t[chosen]

        state, action, reward, next_state, desired_goal, done, next_achieved_goal = \
            self._collect(ep_indices, time_indices)
        desired_goal, her_reward, her_done = self._her_relabel(
            ep_indices, time_indices, desired_goal, next_achieved_goal)
        if her_reward is not None:
            reward, done = her_reward, her_done

        indices = list(zip(ep_indices.tolist(), time_indices.tolist()))
        return state, action, reward, next_state, desired_goal, done, indices

    def update_priorities(self, indices, td_errors):
        """
        Push new priorities back after a train step.

        Parameters
        ----------
        indices   : list of (ep_idx, timestep) tuples returned by sample()
        td_errors : 1-D array-like of scalar TD errors, same length as indices
        """
        td_errors = np.abs(np.asarray(td_errors, dtype=np.float32)) + self.eps
        for (ep_idx, t), td in zip(indices, td_errors):
            if ep_idx < len(self.buffer):           # guard against eviction
                ep_id = id(self.buffer[ep_idx])
                self._priorities[(ep_id, t)] = float(td)


class DDPG:
    def __init__(self, n_states, n_actions, n_goals, action_bounds, capacity, env,
                 k_future,
                 batch_size,
                 ram,
                 action_size=4,
                 tau=0.05,
                 actor_lr=1e-3,
                 critic_lr=1e-3,
                 gamma=0.98,
                 epsilon=0.99,
                 decay_rate=0.995):

        self.n_states = n_states
        self.n_actions = n_actions
        self.n_goals = n_goals
        self.k_future = k_future
        self.action_bounds = action_bounds
        self.action_size = action_size
        self.env = env
        self.ram = ram
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.lr = actor_lr

        self.actor        = Actor(self.n_states, n_actions=self.n_actions, n_goals=self.n_goals).to(device)
        self.critic       = Critic(self.n_states, action_size=self.action_size, n_goals=self.n_goals).to(device)
        self.actor_target  = Actor(self.n_states, n_actions=self.n_actions, n_goals=self.n_goals).to(device)
        self.critic_target = Critic(self.n_states, action_size=self.action_size, n_goals=self.n_goals).to(device)
        self.init_target_networks()
        self.tau   = tau
        self.gamma = gamma

        self.capacity   = capacity
        self.batch_size = batch_size
        self.actor_lr   = actor_lr
        self.critic_lr  = critic_lr
        self.actor_optim  = Adam(self.actor.parameters(),  self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), self.critic_lr)

        self.state_normalizer = Normalizer(self.n_states[0], default_clip_range=5)
        self.goal_normalizer  = Normalizer(self.n_goals,     default_clip_range=5)

    def choose_action(self, state, goal, train_mode=True):
        state = self.state_normalizer.normalize(state)
        goal  = self.goal_normalizer.normalize(goal)
        state = np.expand_dims(state, axis=0)
        goal  = np.expand_dims(goal,  axis=0)

        with torch.no_grad():
            state_goal = np.concatenate([state, goal], axis=1)
            state_goal = from_numpy(state_goal).float().to(device)
            action = self.actor(state_goal)[0].cpu().numpy()

        if train_mode:
            if np.random.rand() < self.epsilon:
                action = np.random.uniform(low=self.action_bounds[0], high=self.action_bounds[1], size=self.n_actions)
            else:
                sigma  = 0.05 * (self.action_bounds[1] - self.action_bounds[0])
                action = action + sigma * np.random.randn(self.n_actions)
                action = np.clip(action, self.action_bounds[0], self.action_bounds[1])

            self.epsilon = max(self.epsilon * self.decay_rate, 0.05)

        return action

    def train(self):
        state, action, reward, state_next, goal, done, indices = self.ram.sample(self.batch_size)

        # Update normalizers with raw (pre-normalization) numpy arrays
        self.state_normalizer.update(state)
        self.state_normalizer.update(state_next)
        self.goal_normalizer.update(goal)
        self.state_normalizer.recompute_stats()
        self.goal_normalizer.recompute_stats()

        # Normalize for network inputs
        state      = self.state_normalizer.normalize(state)
        state_next = self.state_normalizer.normalize(state_next)
        goal       = self.goal_normalizer.normalize(goal)

        reward = from_numpy(reward).float().to(device)
        action = from_numpy(action).float().to(device)
        # FIX: done must be (batch,1) to broadcast correctly against squeezed critic output
        done   = from_numpy(done).float().unsqueeze(1).to(device)

        state_goal      = from_numpy(np.concatenate([state,      goal], axis=1)).float().to(device)
        state_next_goal = from_numpy(np.concatenate([state_next, goal], axis=1)).float().to(device)

        # ---- Train Critic ----
        with torch.no_grad():
            next_q = self.critic_target(state_next_goal, self.actor_target(state_next_goal))
            # FIX: multiply by (1 - done) so we don't bootstrap past terminal states
            q_expected = reward.unsqueeze(1) + self.gamma * (1.0 - done) * next_q
            q_expected = torch.clamp(q_expected, -1 / (1 - self.gamma), 0)

        q_predicted  = self.critic(state_goal, action)
        loss_critic  = F.mse_loss(q_predicted, q_expected)
        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()

        # ---- Train Actor ----
        a = self.actor(state_goal)
        # FIX: removed a.pow(2).mean() regularization — it penalizes large actions and
        #      caps performance on goals that require strong committed actions
        loss_actor = -self.critic(state_goal, a).mean()
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()

        with torch.no_grad():
            td_err = (q_expected - q_predicted).abs().mean().item()
            # Per-sample TD errors for priority update
            td_errors_per_sample = (q_expected - q_predicted).abs().squeeze(1).cpu().numpy()

        # Push updated priorities back if the buffer supports it (PrioritizedMemoryBuffer)
        if hasattr(self.ram, "update_priorities"):
            self.ram.update_priorities(indices, td_errors_per_sample)

        return {
            "actor_loss":  float(loss_actor.item()),
            "critic_loss": float(loss_critic.item()),
            "q_pred_mean": float(q_predicted.mean().item()),
            "q_tgt_mean":  float(q_expected.mean().item()),
            "td_error":    td_err,
            "reward":      float(reward.mean().item()),
        }

    def update_target_networks(self):
        self.soft_update_networks(self.actor,  self.actor_target,  self.tau)
        self.soft_update_networks(self.critic, self.critic_target, self.tau)

    def init_target_networks(self):
        self.hard_update_networks(self.actor,  self.actor_target)
        self.hard_update_networks(self.critic, self.critic_target)

    @staticmethod
    def hard_update_networks(local_model, target_model):
        target_model.load_state_dict(local_model.state_dict())

    @staticmethod
    def soft_update_networks(local_model, target_model, tau):
        for target_param, param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )