import torch
from torch import from_numpy, device
import numpy as np
from src.rl_loop.utils_rl import torch_load_checkpoint
from torch.optim import Adam
from mpi4py import MPI
import threading
from torch import nn
import torch
from torch.nn import functional as F
import numpy as np

class DDPG:
    def __init__(self, n_states, n_actions, n_goals, action_bounds, capacity, env,
                 k_future,
                 batch_size,
                 action_size=1,
                 tau=0.05,
                 actor_lr=1e-3,
                 critic_lr=1e-3,
                 gamma=0.98,
                 verbose = False):

        self.device = device("cpu")
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_goals = n_goals
        self.k_future = k_future
        self.action_bounds = action_bounds
        self.action_size = action_size
        self.env = env

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
        self.randomprob = 0.3
        self.verbose = verbose

        if self.verbose:
            print(f"Initialized DDPG Agent with buffer type: Experience Replay")
            print(f"Initialized DDPG Agent with batch size: {batch_size}")
            print(f"Initialized DDPG Agent with actor learning rate: {actor_lr}")
            print(f"Initialized DDPG Agent with critic learning rate: {critic_lr}")
            print(f"Initialized DDPG Agent with gamma: {gamma}")
            print(f"Initialized DDPG Agent with tau: {tau}")
            print(f"Initialized DDPG Agent with k_future: {k_future}")
            print("\n\n")

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
            action += np.random.binomial(1, self.randomprob, 1)[0] * (random_actions - action)

        return action

    def store(self, mini_batch):
        for batch in mini_batch:
            self.memory.add(batch)
        self._update_normalizer(mini_batch)

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
        states, actions, rewards, next_states, goals = self.memory.sample(self.batch_size)

        states = self.state_normalizer.normalize(states)
        next_states = self.state_normalizer.normalize(next_states)
        goals = self.goal_normalizer.normalize(goals)
        inputs = np.concatenate([states, goals], axis=1)
        next_inputs = np.concatenate([next_states, goals], axis=1)

        inputs = torch.Tensor(inputs).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        next_inputs = torch.Tensor(next_inputs).to(self.device)
        actions = torch.Tensor(actions).to(self.device)

        with torch.no_grad():
            target_q = self.critic_target(next_inputs, self.actor_target(next_inputs))
            target_returns = rewards + self.gamma * target_q.detach()
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

    def _update_normalizer(self, mini_batch):
        states, goals = self.memory.sample_for_normalization(mini_batch)

        self.state_normalizer.update(states)
        self.goal_normalizer.update(goals)
        self.state_normalizer.recompute_stats()
        self.goal_normalizer.recompute_stats()

    @staticmethod
    def sync_networks(network):
        comm = MPI.COMM_WORLD
        flat_params = _get_flat_params_or_grads(network, mode='params')
        comm.Bcast(flat_params, root=0)
        _set_flat_params_or_grads(network, flat_params, mode='params')

    @staticmethod
    def sync_grads(network):
        flat_grads = _get_flat_params_or_grads(network, mode='grads')
        comm = MPI.COMM_WORLD
        global_grads = np.zeros_like(flat_grads)
        comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
        _set_flat_params_or_grads(network, global_grads, mode='grads')
    
    def load_model(self, filename):
        checkpoint = torch_load_checkpoint(filename)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optim.load_state_dict(checkpoint["actor_optim"])
        self.critic_optim.load_state_dict(checkpoint["critic_optim"])


def _get_flat_params_or_grads(network, mode='params'):
    attr = 'data' if mode == 'params' else 'grad'
    return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])


def _set_flat_params_or_grads(network, flat_params, mode='params'):
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
        local_sum[...] = self._mpi_average(local_sum)
        local_sumsq[...] = self._mpi_average(local_sumsq)
        local_count[...] = self._mpi_average(local_count)
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
    def _mpi_average(self, x):
        buf = np.zeros_like(x)
        MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
        buf /= MPI.COMM_WORLD.Get_size()
        return buf

    # normalize the observation
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v - self.mean) / self.std, -clip_range, clip_range)


import numpy as np
from copy import deepcopy as dc
import random


class Memory:
    def __init__(self, capacity, k_future, env):
        self.capacity = capacity
        self.memory = []
        self.memory_counter = 0
        self.memory_length = 0
        self.env = env.unwrapped

        self.future_p = 1 - (1. / (1 + k_future))

    def sample(self, batch_size):

        ep_indices = np.random.randint(0, len(self.memory), batch_size)
        time_indices = np.random.randint(0, len(self.memory[0]["next_state"]), batch_size)
        
        states = []
        actions = []
        desired_goals = []
        next_states = []
        next_achieved_goals = []

        for episode, timestep in zip(ep_indices, time_indices):
            try:
                # print(episode, timestep, len(self.memory[episode]["state"]))
                timestep = min(timestep, len(self.memory[episode]["state"]) - 1)
                states.append(dc(self.memory[episode]["state"][timestep]))
                actions.append(dc(self.memory[episode]["action"][timestep]))
                desired_goals.append(dc(self.memory[episode]["desired_goal"][timestep]))
                next_achieved_goals.append(dc(self.memory[episode]["next_achieved_goal"][timestep]))
                next_states.append(dc(self.memory[episode]["next_state"][timestep]))
            except:
                timestep = min(timestep, random.randint(0, len(self.memory[episode]["state"])-1))
                states.append(dc(self.memory[episode]["state"][timestep]))
                actions.append(dc(self.memory[episode]["action"][timestep]))
                desired_goals.append(dc(self.memory[episode]["desired_goal"][timestep]))
                next_achieved_goals.append(dc(self.memory[episode]["next_achieved_goal"][timestep]))
                next_states.append(dc(self.memory[episode]["next_state"][timestep]))

        states = np.vstack(states)
        actions = np.vstack(actions)
        desired_goals = np.vstack(desired_goals)
        next_achieved_goals = np.vstack(next_achieved_goals)
        next_states = np.vstack(next_states)

        her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (len(self.memory[0]["next_state"]) - time_indices)
        future_offset = future_offset.astype(int)
        future_t = (time_indices + 1 + future_offset)[her_indices]

        future_ag = []
        for episode, f_offset in zip(ep_indices[her_indices], future_t):
            try:
                future_ag.append(dc(self.memory[episode]["achieved_goal"][f_offset]))
            except:
                f_offset = min(f_offset, random.randint(0, len(self.memory[episode]["next_achieved_goal"])-1))
                future_ag.append(dc(self.memory[episode]["next_achieved_goal"][f_offset]))

        future_ag = np.vstack(future_ag)

        desired_goals[her_indices] = future_ag
        rewards = np.expand_dims(self.env.compute_reward(next_achieved_goals, desired_goals, None), 1)

        return self.clip_obs(states), actions, rewards, self.clip_obs(next_states), self.clip_obs(desired_goals)

    def add(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        assert len(self.memory) <= self.capacity

    def __len__(self):
        return len(self.memory)

    @staticmethod
    def clip_obs(x):
        return np.clip(x, -200, 200)

    def sample_for_normalization(self, batch):
        size = len(batch[0]["next_state"])
        ep_indices = np.random.randint(0, len(batch), size)
        time_indices = np.random.randint(0, len(batch[0]["next_state"]), size)
        states = []
        desired_goals = []

        for episode, timestep in zip(ep_indices, time_indices):
            try:
                states.append(dc(batch[episode]["state"][timestep]))
                desired_goals.append(dc(batch[episode]["desired_goal"][timestep]))
            except:
                timestep = min(timestep, random.randint(0, len(batch[episode]["state"])-1))
                states.append(dc(batch[episode]["state"][timestep]))
                desired_goals.append(dc(batch[episode]["desired_goal"][timestep]))

        states = np.vstack(states)
        desired_goals = np.vstack(desired_goals)

        her_indices = np.where(np.random.uniform(size=size) < self.future_p)
        future_offset = np.random.uniform(size=size) * (len(batch[0]["next_state"]) - time_indices)
        future_offset = future_offset.astype(int)
        future_t = (time_indices + 1 + future_offset)[her_indices]

        future_ag = []
        for episode, f_offset in zip(ep_indices[her_indices], future_t):
            try:
                future_ag.append(dc(batch[episode]["achieved_goal"][f_offset]))
            except:
                f_offset = min(f_offset, random.randint(0, len(batch[episode]["next_achieved_goal"])-1))
                future_ag.append(dc(batch[episode]["next_achieved_goal"][f_offset]))

        future_ag = np.vstack(future_ag)

        desired_goals[her_indices] = future_ag

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
    def __init__(self, n_states, n_goals, n_hidden1=256, n_hidden2=256, n_hidden3=256, initial_w=3e-3, action_size=1):
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
