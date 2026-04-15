import torch
import random
import torch.nn as nn
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
from torch import from_numpy
import matplotlib.pyplot as plt
from collections import deque

# Epsilon = 0.1 # Epsilon-greedy exploration parameter

# transfer to mps if it's a Mac
device = torch.device("cuda:0"  if torch.cuda.is_available() else 
                       "cpu")

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
        output = torch.tanh(self.output(x))

        return output
    
# actor = Actor((10,), 1, 1)
# # print(list(actor.parameters()))
# x = torch.tensor(np.zeros(11), dtype=torch.float32)
# print(x)
# print(actor.forward(x))

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
        output = self.output(x) # Critic output is a single value representing the Q-value for the given state-action pair

        return output

import threading


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
        self.lock = threading.Lock()

    # update the parameters of the normalizer
    def update(self, v):
        v = v.reshape(-1, self.size)
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

        self.total_sum += local_sum
        self.total_sumsq += local_sumsq
        self.total_count += local_count
        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(
            self.total_sum / self.total_count)))

    # normalize the observation
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

    def sample(self, batch_size):
        ep_indices = np.random.randint(0, len(self.buffer), batch_size)
        time_indices = np.array([
            np.random.randint(0, len(self.buffer[ep]["next_state"]) - 1)
            for ep in ep_indices
        ], dtype=np.int64)

        state = np.float32([self.buffer[episode]["state"][timestep] for episode, timestep in zip(ep_indices, time_indices)])
        action = np.float32([self.buffer[episode]["action"][timestep] for episode, timestep in zip(ep_indices, time_indices)])
        reward = np.float32([self.buffer[episode]["reward"][timestep] for episode, timestep in zip(ep_indices, time_indices)])
        next_state = np.float32([self.buffer[episode]["next_state"][timestep] for episode, timestep in zip(ep_indices, time_indices)])
        desired_goal = np.float32([self.buffer[episode]["desired_goal"][timestep] for episode, timestep in zip(ep_indices, time_indices)])
        done = np.float32([self.buffer[episode]["done"][timestep] for episode, timestep in zip(ep_indices, time_indices)])
        next_achieved_goal = np.float32([self.buffer[episode]["next_achieved_goal"][timestep] for episode, timestep in zip(ep_indices, time_indices)])

        # Transition-level HER relabeling: replace desired goal by achieved goal.
        her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)[0]

        future_offset = np.array([
            int(np.random.uniform() * (len(self.buffer[ep]["next_state"]) - time_indices[i] - 1))
            for i, ep in enumerate(ep_indices)
        ], dtype=np.int64)
        future_timesteps = (time_indices + 1 + future_offset)[her_indices]
        desired_goal[her_indices] = np.float32([self.buffer[ep_indices[her_indices[i]]]["achieved_goal"][future_timesteps[i]] for i in range(len(her_indices))])
        if self.env is not None:
            # Recompute reward and done for the HER relabeled transitions
            reward = self.env.compute_reward(next_achieved_goal, desired_goal, None
                ).astype(np.float32)
            done = (reward == 0.0).astype(np.float32)

        return state, action, reward, next_state, desired_goal, done

    def add_episode(self, episode_batch):
        self.buffer.append(episode_batch)
        self.len += 1
        self.len_transition += len(episode_batch["state"])
        if self.len > self.maxSize:
           self.len = self.maxSize


class Agent:
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
                 decay_rate=0.995
                 ):
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_goals = n_goals
        self.k_future = k_future
        self.action_bounds = action_bounds
        self.action_size = action_size
        self.env = env # I don't know where this argument is used
        self.ram = ram # Add a replay buffer using samples in it to train neural network
        self.epsilon = epsilon
        self.randomprob = 0.2
        self.decay_rate = decay_rate

        self.actor = Actor(self.n_states, n_actions=self.n_actions, n_goals=self.n_goals).to(device)
        self.critic = Critic(self.n_states, action_size=self.action_size, n_goals=self.n_goals).to(device)
        self.actor_target = Actor(self.n_states, n_actions=self.n_actions, n_goals=self.n_goals).to(device)
        self.critic_target = Critic(self.n_states, action_size=self.action_size, n_goals=self.n_goals).to(device)
        self.init_target_networks() # Initialize target networks with the same weights as the main networks
        self.tau = tau
        self.gamma = gamma

        self.capacity = capacity
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optim = Adam(self.actor.parameters(), self.actor_lr) 
        self.critic_optim = Adam(self.critic.parameters(), self.critic_lr)

        self.state_normalizer = Normalizer(self.n_states[0], default_clip_range=5)
        self.goal_normalizer = Normalizer(self.n_goals, default_clip_range=5)

    def choose_action(self, state, goal, train_mode=True):
        # Normalize state and goal, and add batch dimension
        state = self.state_normalizer.normalize(state)
        goal = self.goal_normalizer.normalize(goal)
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        goal = np.expand_dims(goal, axis=0)

        # Compute action using the actor network
        with torch.no_grad():
            state_goal = np.concatenate([state, goal], axis=1)
            state_goal = from_numpy(state_goal).float().to(device)
            action = self.actor(state_goal)[0].cpu().numpy()  # Get action and convert to numpy

        # Add exploration noise during training
        # if train_mode:
        #     action += 0.2 * np.random.randn(self.n_actions)  # Exploration noise
        #     action = np.clip(action, self.action_bounds[0], self.action_bounds[1]) # Clip to action bounds
        #     random_actions = np.random.uniform(low=self.action_bounds[0], high=self.action_bounds[1],
        #                                        size=self.n_actions)  # Random actions for epsilon-greedy exploration
        #     action += np.random.binomial(1, self.epsilon, 1)[0] * (random_actions - action)
        
        if train_mode:
            if np.random.rand() < self.randomprob:
                action = np.random.uniform(low=self.action_bounds[0], high=self.action_bounds[1], size=self.n_actions)
            else:
                sigma = 0.05 * (self.action_bounds[1] - self.action_bounds[0])  # 5% of action range
                action = action + sigma * np.random.randn(self.n_actions)
                action = np.clip(action, self.action_bounds[0], self.action_bounds[1])  # Ensure action is within bounds

            self.randomprob *= self.decay_rate  # Decay random probability
            self.randomprob = max(self.randomprob, 0.05)  # Clip random probability

        return action
    
    # train part using adam optium
    def train(self, goal):
        state, action, reward, state_next, goal, done = self.ram.sample(self.batch_size)

        # update normalizer
        self.state_normalizer.update(state)
        self.state_normalizer.update(state_next)
        self.goal_normalizer.update(goal)
        self.state_normalizer.recompute_stats()
        self.goal_normalizer.recompute_stats()

        # use normalized batch for training
        state = self.state_normalizer.normalize(state)
        state_next = self.state_normalizer.normalize(state_next)
        goal = self.goal_normalizer.normalize(goal)

        # transfer reward and action to tensor
        reward = from_numpy(reward).float().to(device)
        action = from_numpy(action).float().to(device)
        done = from_numpy(done).float().to(device)

        # concatenate state and goal and transfer to tensor
        state_goal = np.concatenate([state, goal], axis=1)
        state_goal = from_numpy(state_goal).float().to(device)
        state_next_goal = np.concatenate([state_next, goal], axis=1)
        state_next_goal = from_numpy(state_next_goal).float().to(device)

        # -------Train Critic-------
        with torch.no_grad():
            q_expected = reward + self.gamma * torch.squeeze(
                self.critic_target(state_next_goal, self.actor_target(state_next_goal))
            )
            # Clip the target Q-values to prevent them from becoming too large, which can destabilize training
            q_expected = torch.clamp(q_expected, -1 / (1 - self.gamma), 0)

        q_predicted = torch.squeeze(self.critic(state_goal, action))
        loss_critic = F.mse_loss(q_predicted, q_expected.detach()) 
        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()

        # -------Train Actor-------
        a = self.actor(state_goal)
        loss_actor = -self.critic(state_goal, a).mean() + a.pow(2).mean()
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()

        with torch.no_grad():
            td_err = (q_expected - q_predicted).abs().mean().item()

        metrics = {
            "actor_loss": float(loss_actor.item()),
            "critic_loss": float(loss_critic.item()),
            "q_pred_mean": float(q_predicted.mean().item()),
            "q_tgt_mean": float(q_expected.mean().item()),
            "td_error": td_err,
            "reward": float(reward.mean().item()),
        }
        return metrics

    def update_target_networks(self):
        self.soft_update_networks(self.actor, self.actor_target, self.tau)
        self.soft_update_networks(self.critic, self.critic_target, self.tau)
 
    # Initialize the target networks by copying the weights from the main networks
    def init_target_networks(self):
        self.hard_update_networks(self.actor, self.actor_target)
        self.hard_update_networks(self.critic, self.critic_target)
        # self.actor.load_state_dict(self.actor.state_dict()) 

    @staticmethod
    # Copy the weights from the local model to the target model
    def hard_update_networks(local_model, target_model):
        target_model.load_state_dict(local_model.state_dict())

    # Copy the weights from the current networks to target networks with a soft update (using tau)
    @staticmethod
    def soft_update_networks(local_model, target_model, tau):
        for target_param, param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
		)