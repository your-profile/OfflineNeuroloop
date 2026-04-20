
import torch
import torch.nn as nn
from src.rl_loop.utils_rl import torch_load_checkpoint
import torch.optim as optim
import numpy as np
import random
from tqdm import trange
from collections import namedtuple, deque
import heapq
from src.networks import utils_networks as utils


device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Q NETWORK ARCHITECTURE
class DeepQNetwork(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_layer_size=128):
        super(DeepQNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(n_observations, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, n_actions)
            )

    def forward(self, x):
        return self.fc(x)

#DQN ALGORITHM
class DQN():
    def __init__(self, n_observations, n_actions, batch_size=64, lr=1e-4, gamma=0.99, mem_size=int(1e5), learn_step=5, tau=1e-3, hidden_layer_size=128, buffer_type = "ER", verbose = False):
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_step = learn_step
        self.tau = tau
        self.lr = lr
        self.hidden_layer_size = hidden_layer_size
        self.buffer_type = buffer_type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.policy_net = DeepQNetwork(n_observations, n_actions, hidden_layer_size).to(self.device)
        self.target_net = DeepQNetwork(n_observations, n_actions, hidden_layer_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        if buffer_type == "PER":
            self.memory = PrioritizedReplayBuffer(n_actions, mem_size, batch_size)
        else:
            self.memory = ReplayBuffer(n_actions, mem_size, batch_size)

        self.algorithm = "DQN"
        self.verbose = verbose

        if self.verbose:
            print(f"Initialized DQN Agent with buffer type: {buffer_type}")
            print(f"Initialized DQN Agent with hidden layer size: {hidden_layer_size}")
            print(f"Initialized DQN Agent with batch size: {batch_size}")
            print(f"Initialized DQN Agent with learning rate: {lr}")
            print(f"Initialized DQN Agent with gamma: {gamma}")
            print(f"Initialized DQN Agent with tau: {tau}")
            print("\n\n")


        self.counter = 0

    def chooseAction(self, state, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()

        # epsilon-greedy
        if random.random() < epsilon:
            action = random.choice(np.arange(self.n_actions))
        else:
            action = np.argmax(action_values.cpu().data.numpy())

        return action, utils.Data.from_tensor(x=action_values)[action]

    def remember(self, state, action, reward, next_state, done, q_augmentation = 0.0, priority = 0.0):
        if priority is None:
            #priority is td error
            priority = reward + self.gamma * self.target_net((torch.from_numpy(next_state).float().unsqueeze(0).to(self.device))).squeeze()[action]  - self.policy_net(torch.from_numpy(state).float().unsqueeze(0).to(self.device)).squeeze()[action] 
        
        action = int(action)

        self.counter += 1
        if self.counter % self.learn_step == 0:
            if len(self.memory) >= self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

        self.memory.add(state, action, reward, next_state, done, priority, q_augmentation)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones, q_augmentation = experiences

        q_target = self.target_net(next_states).detach().max(axis=1)[0].unsqueeze(1)
        y_j = rewards + self.gamma * q_target * (1 - dones)  
        y_j += q_augmentation
        q_eval = self.policy_net(states).gather(1, actions)

        #backpropogation
        loss = self.criterion(q_eval, y_j)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #Update Target Policy
        self.softUpdate()

    def set_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr



    def softUpdate(self):
        for eval_param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau*eval_param.data + (1.0-self.tau)*target_param.data)
        
    def load_model(self, filename):
        ckpt = torch_load_checkpoint(filename)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
        self.policy_net.load_state_dict(state)
        self.target_net.load_state_dict(state)

        return self


_Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "q_augmentation"])


class ReplayBuffer:
    """Uniform experience replay: FIFO storage and uniform random minibatches."""

    def __init__(self, n_actions, memory_size, batch_size):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.capacity = int(memory_size)
        self.memory = deque(maxlen=self.capacity)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("Initialized Replay Buffer")

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done, priority=None, q_augmentation=0.0):
        self.memory.append(_Experience(state, action, reward, next_state, done, q_augmentation))

    def sample(self):
        current_size = len(self.memory)
        k = min(self.batch_size, current_size)
        experiences = random.sample(self.memory, k=k)

        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)
        q_augmentation = torch.from_numpy(np.vstack([e.q_augmentation for e in experiences])).float().to(self.device)
        return (states, actions, rewards, next_states, dones, q_augmentation)


class PrioritizedReplayBuffer():
    def __init__(self, n_actions, memory_size, batch_size, alpha=0.6, eps=1e-6):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.capacity = memory_size
        self.alpha = alpha
        self.eps = eps  

        self.memory = []
        self.priorities = np.zeros((memory_size,), dtype=np.float32)
        self.pos = 0
        self.experience = _Experience
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("Initialized Prioritized Replay Buffer")

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done, priority=None, q_augmentation=0.0):
        e = self.experience(state, action, reward, next_state, done, q_augmentation)

        if len(self.memory) < self.capacity:
            self.memory.append(e)
        else:
            self.memory[self.pos] = e

        raw = abs(float(priority)) if priority is not None else 1.0
        self.priorities[self.pos] = max(raw, self.eps)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self):
        current_size = len(self.memory)
        k = min(self.batch_size, current_size)
        priorities = self.priorities[:current_size]

        scaled = np.maximum(priorities, self.eps) ** self.alpha
        total = float(scaled.sum())

        if total <= 0 or current_size == 0:
            indices = np.random.choice(current_size, k, replace=False)
        elif np.allclose(scaled, scaled[0]):
            indices = np.random.choice(current_size, k, replace=False)
        else:
            probs = scaled / total
            nonzero = int(np.count_nonzero(probs))
            use_replace = k > nonzero or k > current_size
            if use_replace:
                indices = np.random.choice(current_size, k, replace=True, p=probs)
            else:
                indices = np.random.choice(current_size, k, replace=False, p=probs)

        experiences = [self.memory[i] for i in indices]

        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)
        q_augmentation = torch.from_numpy(np.vstack([e.q_augmentation for e in experiences])).float().to(self.device)

        return (states, actions, rewards, next_states, dones, q_augmentation)
