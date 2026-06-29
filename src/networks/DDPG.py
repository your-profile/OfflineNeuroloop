import threading
from copy import deepcopy as dc

import numpy as np
import torch
from mpi4py import MPI
from torch import from_numpy, nn
from torch.nn import functional as F
from torch.optim import Adam

from src.rl_loop.utils_rl import torch_load_checkpoint

MIN_PRIORITY = 1e-8

class DDPG:
    def __init__(self, n_states, n_actions, n_goals, action_bounds, capacity, env,
                 k_future,
                 batch_size,
                 action_size=1,
                 tau=0.05,
                 actor_lr=1e-3,
                 critic_lr=1e-3,
                 gamma=0.98,
                 seed=None,
                 verbose = False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_goals = n_goals
        self.k_future = k_future
        self.action_bounds = action_bounds
        self.action_size = action_size
        self.env = env

        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        self.actor = Actor(self.n_states, n_actions=self.n_actions, n_goals=self.n_goals).to(self.device)
        self.critic = Critic(self.n_states, action_size=self.action_size, n_goals=self.n_goals).to(self.device)
        self.sync_networks(self.actor)
        self.sync_networks(self.critic)
        self.actor_target = Actor(self.n_states, n_actions=self.n_actions, n_goals=self.n_goals).to(self.device)
        self.critic_target = Critic(self.n_states, action_size=self.action_size, n_goals=self.n_goals).to(self.device)
        self.init_target_networks()
        self.tau = tau
        self.gamma = gamma

        self.capacity = capacity
        self.memory = Memory(self.capacity, self.k_future, self.env)

        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optim = Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), self.critic_lr)

        self.state_normalizer = Normalizer(self.n_states[0], default_clip_range=5)
        self.goal_normalizer = Normalizer(self.n_goals, default_clip_range=5)
        self.verbose = verbose

        if self.verbose:
            print(f"Initialized DDPG_PER Agent with buffer type: Prioritized Experience Replay")
            print(f"Initialized DDPG_PER Agent with batch size: {batch_size}")
            print(f"Initialized DDPG_PER Agent with actor learning rate: {actor_lr}")
            print(f"Initialized DDPG_PER Agent with critic learning rate: {critic_lr}")
            print(f"Initialized DDPG_PER Agent with gamma: {gamma}")
            print(f"Initialized DDPG_PER Agent with tau: {tau}")
            print(f"Initialized DDPG_PER Agent with k_future: {k_future}")
            print("\n\n")
            
        print("GPU Available: ", torch.cuda.is_available())

    def choose_action(self, state, goal, train_mode=True):
        state = self.state_normalizer.normalize(state)
        goal = self.goal_normalizer.normalize(goal)
        state = np.expand_dims(state, axis=0)
        goal = np.expand_dims(goal, axis=0)

        with torch.no_grad():
            x = np.concatenate([state, goal], axis=1)
            x = from_numpy(x).float().to(self.device)
            action = self.actor(x)[0].cpu().data.numpy()

        if train_mode:
            action += 0.2 * np.random.randn(self.n_actions)
            action = np.clip(action, self.action_bounds[0], self.action_bounds[1])

            random_actions = np.random.uniform(low=self.action_bounds[0], high=self.action_bounds[1],
                                               size=self.n_actions)
            action += np.random.binomial(1, 0.3, 1)[0] * (random_actions - action)

        return action

    def store(self, mini_batch):
        for episode in mini_batch:
            self.fill_missing_priorities(episode)
            self.memory.add(episode)
        self.update_normalizer(mini_batch)

    def fill_missing_priorities(self, episode):
        """Use explicit step priorities when provided; otherwise use TD error."""
        step_priorities = episode.get("transition_priority")
        if not step_priorities:
            return

        num_steps = len(episode["state"])
        resolved = []
        for step in range(num_steps):
            priority = step_priorities[step] if step < len(step_priorities) else None
            if priority is None:
                priority = self.td_error(episode, step)
            resolved.append(max(abs(float(priority)), MIN_PRIORITY))
        episode["transition_priority"] = resolved

    def td_error(self, episode, step):
        state = self.state_normalizer.normalize(episode["state"][step])
        next_state = self.state_normalizer.normalize(episode["next_state"][step])
        goal = self.goal_normalizer.normalize(episode["desired_goal"][step])
        action = np.asarray(episode["action"][step], dtype=np.float32)
        reward = float(episode["reward"][step])
        q_augmentation = float(episode["q_augmentation"][step])

        state_goal = np.concatenate([state, goal])
        next_state_goal = np.concatenate([next_state, goal])

        with torch.no_grad():
            inputs = torch.tensor(state_goal, dtype=torch.float32, device=self.device).unsqueeze(0)
            next_inputs = torch.tensor(next_state_goal, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)

            q_value = self.critic(inputs, action_tensor).squeeze()
            next_action = self.actor_target(next_inputs)
            target_q = self.critic_target(next_inputs, next_action).squeeze()
            target = reward + self.gamma * target_q + q_augmentation
            target = torch.clamp(target, -1 / (1 - self.gamma), 0)
            return float(torch.abs(target - q_value).item())

    def init_target_networks(self):
        self.hard_update_networks(self.actor, self.actor_target)
        self.hard_update_networks(self.critic, self.critic_target)

    @staticmethod
    def hard_update_networks(local_model, target_model):
        target_model.load_state_dict(local_model.state_dict())

    @staticmethod
    def soft_update_networks(local_model, target_model, tau=0.05):
        for t_params, e_params in zip(target_model.parameters(), local_model.parameters()):
            t_params.data.copy_(tau * e_params.data + (1 - tau) * t_params.data)

    def train(self):
        states, actions, rewards, next_states, goals, q_augmentation = self.memory.sample(self.batch_size)

        states = self.state_normalizer.normalize(states)
        next_states = self.state_normalizer.normalize(next_states)
        goals = self.goal_normalizer.normalize(goals)
        inputs = np.concatenate([states, goals], axis=1)
        next_inputs = np.concatenate([next_states, goals], axis=1)

        inputs = torch.Tensor(inputs).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        q_augmentation = torch.Tensor(q_augmentation).to(self.device)
        next_inputs = torch.Tensor(next_inputs).to(self.device)
        actions = torch.Tensor(actions).to(self.device)

        with torch.no_grad():
            target_q = self.critic_target(next_inputs, self.actor_target(next_inputs))
            target_returns = rewards + self.gamma * target_q.detach() + q_augmentation
            target_returns = torch.clamp(target_returns, -1 / (1 - self.gamma), 0)

        q_eval = self.critic(inputs, actions)
        critic_loss = (target_returns - q_eval).pow(2).mean()

        a = self.actor(inputs)
        actor_loss = -self.critic(inputs, a).mean()
        actor_loss += a.pow(2).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.sync_grads(self.actor)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.sync_grads(self.critic)
        self.critic_optim.step()

        return actor_loss.item(), critic_loss.item()

    def save_weights(self):
        torch.save({"actor_state_dict": self.actor.state_dict(),
                    "state_normalizer_mean": self.state_normalizer.mean,
                    "state_normalizer_std": self.state_normalizer.std,
                    "goal_normalizer_mean": self.goal_normalizer.mean,
                    "goal_normalizer_std": self.goal_normalizer.std}, "FetchPickAndPlace.pth")

    def load_weights(self):

        checkpoint = torch_load_checkpoint("FetchPickAndPlace.pth")
        actor_state_dict = checkpoint["actor_state_dict"]
        self.actor.load_state_dict(actor_state_dict)
        state_normalizer_mean = checkpoint["state_normalizer_mean"]
        self.state_normalizer.mean = state_normalizer_mean
        state_normalizer_std = checkpoint["state_normalizer_std"]
        self.state_normalizer.std = state_normalizer_std
        goal_normalizer_mean = checkpoint["goal_normalizer_mean"]
        self.goal_normalizer.mean = goal_normalizer_mean
        goal_normalizer_std = checkpoint["goal_normalizer_std"]
        self.goal_normalizer.std = goal_normalizer_std

    def set_to_eval_mode(self):
        self.actor.eval()
        # self.critic.eval()

    def update_networks(self):
        self.soft_update_networks(self.actor, self.actor_target, self.tau)
        self.soft_update_networks(self.critic, self.critic_target, self.tau)

    def update_normalizer(self, mini_batch):
        states, goals = self.memory.sample_for_normalization(mini_batch)

        self.state_normalizer.update(states)
        self.goal_normalizer.update(goals)
        self.state_normalizer.recompute_stats()
        self.goal_normalizer.recompute_stats()

    @staticmethod
    def sync_networks(network):
        comm = MPI.COMM_WORLD
        flat_params = get_flat_params_or_grads(network, mode='params')
        comm.Bcast(flat_params, root=0)
        set_flat_params_or_grads(network, flat_params, mode='params')

    @staticmethod
    def sync_grads(network):
        flat_grads = get_flat_params_or_grads(network, mode='grads')
        comm = MPI.COMM_WORLD
        global_grads = np.zeros_like(flat_grads)
        comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
        set_flat_params_or_grads(network, global_grads, mode='grads')

    def load_model(self, filename):
        checkpoint = torch_load_checkpoint(filename)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optim.load_state_dict(checkpoint["actor_optim"])
        self.critic_optim.load_state_dict(checkpoint["critic_optim"])
        return self


def get_flat_params_or_grads(network, mode='params'):
    attr = 'data' if mode == 'params' else 'grad'
    return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])


def set_flat_params_or_grads(network, flat_params, mode='params'):
    attr = 'data' if mode == 'params' else 'grad'
    pointer = 0
    for param in network.parameters():
        getattr(param, attr).copy_(
            torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
        pointer += param.data.numel()

class Normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        # some local information
        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)
        # get the total sum sumsq and sum count
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)
        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)
        # thread locker
        self.lock = threading.Lock()

    # update the parameters of the normalizer
    def update(self, v):
        v = v.reshape(-1, self.size)
        # do the computing
        with self.lock:
            self.local_sum += v.sum(axis=0)
            self.local_sumsq += (np.square(v)).sum(axis=0)
            self.local_count[0] += v.shape[0]

    # sync the parameters across the cpus
    def sync(self, local_sum, local_sumsq, local_count):
        local_sum[...] = self.mpi_average(local_sum)
        local_sumsq[...] = self.mpi_average(local_sumsq)
        local_count[...] = self.mpi_average(local_count)
        return local_sum, local_sumsq, local_count

    def recompute_stats(self):
        with self.lock:
            local_count = self.local_count.copy()
            local_sum = self.local_sum.copy()
            local_sumsq = self.local_sumsq.copy()
            # reset
            self.local_count[...] = 0
            self.local_sum[...] = 0
            self.local_sumsq[...] = 0
        # sync the stats
        sync_sum, sync_sumsq, sync_count = self.sync(local_sum, local_sumsq, local_count)
        # update the total stuff
        self.total_sum += sync_sum
        self.total_sumsq += sync_sumsq
        self.total_count += sync_count
        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(
            self.total_sum / self.total_count)))

    # average across the cpu's data
    def mpi_average(self, x):
        buf = np.zeros_like(x)
        MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
        buf /= MPI.COMM_WORLD.Get_size()
        return buf

    # normalize the observation
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v - self.mean) / self.std, -clip_range, clip_range)


class Memory:
    """Episode replay buffer with HER and optional prioritized episode sampling."""

    def __init__(self, capacity, k_future, env):
        self.capacity = capacity
        self.episodes = []
        self.env = env.unwrapped
        self.her_probability = 1 - (1.0 / (1 + k_future))

    @staticmethod
    def episode_sampling_weight(episode):
        """Episode weight = max step priority, or uniform when no priorities are stored."""
        step_priorities = episode.get("transition_priority")
        if not step_priorities:
            return 1.0
        return float(max(max(abs(float(p)), MIN_PRIORITY) for p in step_priorities))

    def episode_sampling_probs(self):
        if not self.episodes:
            raise ValueError("Cannot sample from empty memory.")
        weights = np.array([self.episode_sampling_weight(ep) for ep in self.episodes], dtype=np.float64)
        weights = np.maximum(weights, MIN_PRIORITY)
        return weights / weights.sum()

    @staticmethod
    def random_timesteps(episodes, episode_indices):
        return np.array(
            [
                np.random.randint(0, len(episodes[ep]["next_state"]))
                if len(episodes[ep]["next_state"]) > 0
                else 0
                for ep in episode_indices
            ],
            dtype=int,
        )

    def relabel_goals_with_her(self, episodes, episode_indices, timestep_indices, sample_count):
        relabel_mask = np.random.uniform(size=sample_count) < self.her_probability
        relabel_indices = np.flatnonzero(relabel_mask)
        if relabel_indices.size == 0:
            return relabel_indices, None

        episode_lengths = np.array([len(episodes[ep]["next_state"]) for ep in episode_indices], dtype=int)
        steps_until_end = np.maximum(episode_lengths - timestep_indices, 0)
        future_offset = (np.random.uniform(size=sample_count) * steps_until_end).astype(int)
        future_timesteps = timestep_indices + 1 + future_offset

        her_episode_indices = episode_indices[relabel_indices]
        future_timesteps = future_timesteps[relabel_indices]
        max_timesteps = np.array(
            [max(len(episodes[ep]["achieved_goal"]) - 1, 0) for ep in her_episode_indices],
            dtype=int,
        )
        future_timesteps = np.clip(future_timesteps, 0, max_timesteps)

        future_goals = []
        for episode, goal_timestep in zip(her_episode_indices, future_timesteps):
            future_goals.append(dc(episodes[episode]["achieved_goal"][goal_timestep]))
        return relabel_indices, np.vstack(future_goals)

    def sample(self, batch_size):
        num_episodes = len(self.episodes)
        episode_probs = self.episode_sampling_probs()
        sampled_episode_indices = np.random.choice(num_episodes, size=batch_size, replace=True, p=episode_probs)
        sampled_timestep_indices = self.random_timesteps(self.episodes, sampled_episode_indices)

        states, actions, desired_goals = [], [], []
        next_states, next_achieved_goals, q_augmentation = [], [], []

        for episode, timestep in zip(sampled_episode_indices, sampled_timestep_indices):
            episode_data = self.episodes[episode]
            states.append(dc(episode_data["state"][timestep]))
            actions.append(dc(episode_data["action"][timestep]))
            desired_goals.append(dc(episode_data["desired_goal"][timestep]))
            next_achieved_goals.append(dc(episode_data["next_achieved_goal"][timestep]))
            next_states.append(dc(episode_data["next_state"][timestep]))
            q_augmentation.append(dc(episode_data["q_augmentation"][timestep]))

        states = np.vstack(states)
        actions = np.vstack(actions)
        desired_goals = np.vstack(desired_goals)
        next_achieved_goals = np.vstack(next_achieved_goals)
        next_states = np.vstack(next_states)
        q_augmentation = np.vstack(q_augmentation)

        relabel_indices, future_goals = self.relabel_goals_with_her(
            self.episodes, sampled_episode_indices, sampled_timestep_indices, batch_size
        )
        if future_goals is not None:
            desired_goals[relabel_indices] = future_goals

        rewards = np.expand_dims(
            self.env.compute_reward(next_achieved_goals, desired_goals, None), 1
        )

        return (
            self.clip_obs(states),
            actions,
            rewards,
            self.clip_obs(next_states),
            self.clip_obs(desired_goals),
            q_augmentation,
        )

    def add(self, episode):
        self.episodes.append(episode)
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)
        assert len(self.episodes) <= self.capacity

    def _len__(self):
        return len(self.episodes)

    @staticmethod
    def clip_obs(x):
        return np.clip(x, -200, 200)

    def sample_for_normalization(self, batch):
        num_episodes = len(batch)
        if num_episodes == 0:
            raise ValueError("Empty minibatch for normalization.")

        sample_count = max((len(batch[i]["next_state"]) for i in range(num_episodes)), default=1)
        weights = np.array([self.episode_sampling_weight(batch[i]) for i in range(num_episodes)], dtype=np.float64)
        weights = np.maximum(weights, MIN_PRIORITY)
        episode_probs = weights / weights.sum()
        sampled_episode_indices = np.random.choice(num_episodes, size=sample_count, replace=True, p=episode_probs)
        sampled_timestep_indices = self.random_timesteps(batch, sampled_episode_indices)

        states, desired_goals = [], []
        for episode, timestep in zip(sampled_episode_indices, sampled_timestep_indices):
            states.append(dc(batch[episode]["state"][timestep]))
            desired_goals.append(dc(batch[episode]["desired_goal"][timestep]))

        states = np.vstack(states)
        desired_goals = np.vstack(desired_goals)

        relabel_indices, future_goals = self.relabel_goals_with_her(
            batch, sampled_episode_indices, sampled_timestep_indices, sample_count
        )
        if future_goals is not None:
            desired_goals[relabel_indices] = future_goals

        return self.clip_obs(states), self.clip_obs(desired_goals)


def init_weights_biases(size):
    v = 1.0 / np.sqrt(size[0])
    return torch.FloatTensor(size).uniform_(-v, v)


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, n_goals, n_hidden1=256, n_hidden2=256, n_hidden3=256, initial_w=3e-3):
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
        output = torch.tanh(self.output(x))  # TODO add scale of the action

        return output


class Critic(nn.Module):
    def __init__(self, n_states, n_goals, n_hidden1=512, n_hidden2=512, n_hidden3=512, initial_w=3e-3, action_size=1):
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
        output = self.output(x)

        return output
