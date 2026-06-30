
import torch
import torch.nn as nn
from src.rl_loop.utils_rl import torch_load_checkpoint, td_priority
import torch.optim as optim
import numpy as np
import random
from tqdm import trange
from collections import namedtuple, deque
import heapq
from src.networks import utils_networks as utils


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Q NETWORK ARCHITECTURE
class DeepQNetwork(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_layer_size=128, random_seed=None):
        super(DeepQNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(n_observations, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, n_actions)
        )
        # Initialize all weights and biases uniformly between 0 and 1 using the provided seed
        torch.manual_seed(int(random_seed))
        for m in self.fc:
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(0.0, 1.0)
                if m.bias is not None:
                    m.bias.data.uniform_(0.0, 1.0)
 

    def forward(self, x):
        return self.fc(x)

#DQN ALGORITHM
class DQN():
    def __init__(self, n_observations, n_actions, batch_size=64, lr=1e-4, gamma=0.99, mem_size=int(1e5), learn_step=5, tau=1e-3, hidden_layer_size=128, buffer_type = "ER", seed=None, verbose=False):
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_step = learn_step
        self.tau = tau
        self.lr = lr
        self.hidden_layer_size = hidden_layer_size
        self.buffer_type = buffer_type

        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        self.policy_net = DeepQNetwork(n_observations, n_actions, hidden_layer_size, random_seed=seed).to(device)
        self.target_net = DeepQNetwork(n_observations, n_actions, hidden_layer_size, random_seed=seed).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        if buffer_type == "PER":
            self.memory = PrioritizedReplayBuffer(n_actions, mem_size, batch_size)
        else:
            self.memory = ReplayBuffer(n_actions, mem_size, batch_size)

        self.algorithm = "DQN"
        self.verbose = verbose

        if verbose:
            print(f"Initialized DQN agent with buffer type: {buffer_type}")
        

        self.counter = 0

    def chooseAction(self, state, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

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

    def remember(self, state, action, reward, next_state, done, q_augmentation = 0.0, priority = None):
        action = int(action)

        if priority is None:
            priority = td_priority(self, "DQN", reward, action, state, next_state, done=done, buffer_type=self.buffer_type)

        if self.verbose:
            print(f"Added priority: {priority} and q_augmentation: {q_augmentation} to memory")

        self.memory.add(state, action, reward, next_state, done, priority, q_augmentation)

        self.counter += 1
        if self.counter % self.learn_step == 0:
            if len(self.memory) >= self.batch_size:
                self.learn(self.memory.sample())

    def learn(self, experiences):
        if len(experiences) == 8:
            (
                states,
                actions,
                rewards,
                next_states,
                dones,
                q_augs,
                indices,
                is_weights,
            ) = experiences
            indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        elif len(experiences) == 6:
            states, actions, rewards, next_states, dones, q_augs = experiences
            indices = None
            is_weights = torch.ones_like(rewards, dtype=torch.float32, device=device)
        else:
            raise ValueError(f"Unexpected sample bundle length: {len(experiences)}")

        q_target = self.target_net(next_states).detach().max(dim=1, keepdim=True)[0]
        y_j = rewards + self.gamma * q_target * (1 - dones) + q_augs
        q_eval = self.policy_net(states).gather(1, actions)

        if indices is not None:
            td_abs = torch.abs(y_j.detach() - q_eval.detach()).squeeze(1).cpu().numpy()
            self.memory.update_priorities(indices, td_abs)

        td = q_eval - y_j
        loss = (is_weights * (td ** 2)).sum() / (is_weights.sum() + 1e-8)

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
            self.policy_net.load_state_dict(ckpt["model_state_dict"])
            
            if "target_model_state_dict" in ckpt:
                self.target_net.load_state_dict(ckpt["target_model_state_dict"])
            else:
                self.target_net.load_state_dict(ckpt["model_state_dict"])
            
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                
            if "epsilon" in ckpt:
                self.epsilon = ckpt["epsilon"]
                
        else:
            self.policy_net.load_state_dict(ckpt)
            self.target_net.load_state_dict(ckpt)
    
        return self

_Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "next_state", "done", "q_augmentation"],
)


class ReplayBuffer:
    """Uniform experience replay: FIFO storage and uniform random minibatches."""

    def __init__(self, n_actions, memory_size, batch_size):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.capacity = int(memory_size)
        self.memory = deque(maxlen=self.capacity)

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done, priority=None, q_augmentation = None):
        qa = float(q_augmentation) if q_augmentation is not None else 0.0
        self.memory.append(_Experience(state, action, reward, next_state, done, qa))

    def sample(self):
        current_size = len(self.memory)
        k = min(self.batch_size, current_size)
        experiences = random.sample(self.memory, k=k)

        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        q_augs = torch.tensor(
            [e.q_augmentation for e in experiences], dtype=torch.float32, device=device
        ).unsqueeze(1)

        return (states, actions, rewards, next_states, dones, q_augs)


class PrioritizedReplayBuffer():
    def __init__(self, n_actions, memory_size, batch_size, alpha=0.6, beta=0.4, eps=1e-6):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.capacity = memory_size
        self.alpha = alpha
        self.beta = beta
        self.eps = eps  

        self.memory = []
        self.priorities = np.zeros((memory_size,), dtype=np.float32)
        self.pos = 0
        self.experience = _Experience

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done, priority=None, q_augmentation = None):
        qa = float(q_augmentation) if q_augmentation is not None else 0.0
        e = self.experience(state, action, reward, next_state, done, qa)

        if len(self.memory) < self.capacity:
            self.memory.append(e)
        else:
            self.memory[self.pos] = e

        raw = abs(float(priority)) if priority is not None else 1.0
        self.priorities[self.pos] = max(raw, self.eps)
        self.pos = (self.pos + 1) % self.capacity

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        err = np.asarray(td_errors, dtype=np.float64).reshape(-1)
        n = len(self.memory)
        for i, td in zip(idx, err):
            if 0 <= int(i) < n:
                self.priorities[int(i)] = max(float(abs(td)), self.eps)

    def sample(self):
        current_size = len(self.memory)
        k = min(self.batch_size, current_size)
        priorities = self.priorities[:current_size]

        scaled = np.maximum(priorities, self.eps) ** self.alpha
        total = float(scaled.sum())
        uniform_p = 1.0 / float(current_size)

        if total <= 0 or current_size == 0:
            indices = np.random.choice(current_size, k, replace=False)
            prob_i = np.full(k, uniform_p, dtype=np.float64)
        elif np.allclose(scaled, scaled[0]):
            indices = np.random.choice(current_size, k, replace=False)
            prob_i = np.full(k, uniform_p, dtype=np.float64)
        else:
            probs = scaled / total
            nonzero = int(np.count_nonzero(probs))
            use_replace = k > nonzero or k > current_size
            if use_replace:
                indices = np.random.choice(current_size, k, replace=True, p=probs)
            else:
                indices = np.random.choice(current_size, k, replace=False, p=probs)
            prob_i = probs[indices].astype(np.float64)

        experiences = [self.memory[i] for i in indices]

        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        q_augs = torch.tensor(
            [e.q_augmentation for e in experiences], dtype=torch.float32, device=device
        ).unsqueeze(1)

        # Importance-sampling weights (Schaul et al.): (N * P(i))^(-beta), normalized by max.
        raw_w = (current_size * prob_i) ** (-self.beta)
        raw_w = raw_w / (np.max(raw_w) + 1e-8)
        is_weights = torch.from_numpy(raw_w.astype(np.float32)).to(device).unsqueeze(1)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            q_augs,
            np.asarray(indices, dtype=np.int64),
            is_weights,
        )