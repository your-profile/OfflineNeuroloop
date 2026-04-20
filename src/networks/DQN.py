
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


device =  "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        self.policy_net = DeepQNetwork(n_observations, n_actions, hidden_layer_size).to(device)
        self.target_net = DeepQNetwork(n_observations, n_actions, hidden_layer_size).to(device)
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
            td_error = reward + self.gamma * self.target_net((torch.from_numpy(next_state).float().unsqueeze(0).to(device))).squeeze()[action]  - self.policy_net(torch.from_numpy(state).float().unsqueeze(0).to(device)).squeeze()[action] 
            priority = td_error.item()

        self.counter += 1
        if self.counter % self.learn_step == 0:
            if len(self.memory) >= self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, q_augmentation=q_augmentation)

        self.memory.add(state, action, reward, next_state, done, priority)

    def learn(self, experiences, q_augmentation: float = 0.0):
        states, actions, rewards, next_states, dones = experiences

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


_Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """Uniform experience replay: FIFO storage and uniform random minibatches."""

    def __init__(self, n_actions, memory_size, batch_size):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.capacity = int(memory_size)
        self.memory = deque(maxlen=self.capacity)

        print("Initialized Replay Buffer")

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done, priority=None):
        self.memory.append(_Experience(state, action, reward, next_state, done))

    def sample(self):
        current_size = len(self.memory)
        k = min(self.batch_size, current_size)
        experiences = random.sample(self.memory, k=k)

        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)


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

        print("Initialized Prioritized Replay Buffer")

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done, priority=None):
        e = self.experience(state, action, reward, next_state, done)

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

        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)


# #PRIORITIZED EXPERIENCE REPLAY (IN PROGRESS)  
# class PrioritizedReplayBuffer():
#     def __init__(self, n_actions, memory_size, batch_size):
#         self.n_actions = n_actions
#         self.batch_size = batch_size
#         self.memory = []
#         self.capacity = memory_size
#         self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
#         self.counter = 0

#     def __len__(self):
#         return len(self.memory)

#     def add(self, state, action, reward, next_state, done, priority):
#         if priority is None:
#             priority = 0
#         e = self.experience(state, action, reward, next_state, done)
#         exp = (priority, self.counter, e)
#         self.counter += 1

#         if self.__len__() >= self.capacity-1:
#             try:
#                 heapq.heappop(self.memory)
#             except:
#                 print(self.memory[0], exp)
#                 heapq.heapify(self.memory)
#                 self.memory.pop(0)

#         self.memory.append(exp)

#     def sample(self):
#         experiences = random.sample(self.memory, k=self.batch_size)
#         _, _, experiences = zip(*experiences)

#         states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
#         actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
#         rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
#         next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
#         dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

#         return (states, actions, rewards, next_states, dones)

# class PrioritiedExperienceReplay:
#     def __init__(self, n_actions, capacity, batch_size,
#                  alpha=0.6, beta=0.4, beta_increment=1e-4, eps=1e-6):

#         self.n_actions = n_actions
#         self.capacity = capacity
#         self.batch_size = batch_size

#         self.alpha = alpha              
#         self.beta = beta                
#         self.beta_increment = beta_increment
#         self.eps = eps                 

#         self.memory = []
#         self.priorities = np.zeros((capacity,), dtype=np.float32)

#         self.pos = 0
#         self.experience = namedtuple(
#             "Experience",
#             field_names=["state", "action", "reward", "next_state", "done"]
#         )

#     def __len__(self):
#         return len(self.memory)

#     def add(self, state, action, reward, next_state, done):
#         max_priority = self.priorities.max() if self.memory else 1.0

#         e = self.experience(state, action, reward, next_state, done)

#         if len(self.memory) < self.capacity:
#             self.memory.append(e)
#         else:
#             self.memory[self.pos] = e

#         self.priorities[self.pos] = max_priority
#         self.pos = (self.pos + 1) % self.capacity

#     def sample(self, device):
#         if len(self.memory) < self.batch_size:
#             return None

#         # Select relevant priorities
#         if len(self.memory) == self.capacity:
#             priorities = self.priorities
#         else:
#             priorities = self.priorities[:self.pos]

#         # Compute probabilities
#         scaled_priorities = priorities ** self.alpha
#         probs = scaled_priorities / scaled_priorities.sum()

#         indices = np.random.choice(len(self.memory),
#                                    self.batch_size,
#                                    p=probs)

#         experiences = [self.memory[i] for i in indices]

#         # Importance sampling weights
#         total = len(self.memory)
#         weights = (total * probs[indices]) ** (-self.beta)
#         weights /= weights.max()

#         self.beta = min(1.0, self.beta + self.beta_increment)

#         # Convert to tensors
#         states = torch.from_numpy(
#             np.vstack([e.state for e in experiences])
#         ).float().to(device)

#         actions = torch.from_numpy(
#             np.vstack([e.action for e in experiences])
#         ).long().to(device)

#         rewards = torch.from_numpy(
#             np.vstack([e.reward for e in experiences])
#         ).float().to(device)

#         next_states = torch.from_numpy(
#             np.vstack([e.next_state for e in experiences])
#         ).float().to(device)

#         dones = torch.from_numpy(
#             np.vstack([e.done for e in experiences]).astype(np.uint8)
#         ).float().to(device)

#         weights = torch.from_numpy(weights).float().unsqueeze(1).to(device)

#         return states, actions, rewards, next_states, dones, weights, indices

#     def update_priorities(self, indices, td_errors):
#         td_errors = np.abs(td_errors) + self.eps
#         for idx, td_error in zip(indices, td_errors):
#             self.priorities[idx] = td_error


# BATCH_SIZE = 128
# LR = 1e-3
# EPISODES = 5000
# TARGET_SCORE = 140    # early training stop at avg score of last 100 episodes
# GAMMA = 0.99            # discount factor
# MEMORY_SIZE = 100000     # max memory buffer size
# LEARN_STEP = 5          # how often to learn
# TAU = 0.005             # for soft update of target parameters
# SAVE_CHKPT = True      # save trained network .pth file


# def evaluatePolicy(env, agent, loop=3):
#     wins=0
#     for i in range(loop):
#         state  = env.reset()
#         for idx_step in range(500):
#             action = agent.chooseAction(state, epsilon=0)
#             state, reward, done, win = env.step(action)
#             if done:
#                 if win:
#                     wins += 1
#                 break
#     env.close()

#     return wins/loop

# def train():
#     from lunar_lander import LunarLander

#     eval = 0.0
#     eval_hist = []
#     score_hist = []
#     eval_hist = []
#     epsilon = 1.0
#     n_episodes = 2000

#     env = LunarLander()
#     max_steps = 1000
#     agent = DQN(
#             n_observations = env.observation_space.shape[0],
#             n_actions = 4,
#             batch_size = BATCH_SIZE,
#             lr = LR,
#             gamma = GAMMA,
#             mem_size = MEMORY_SIZE,
#             learn_step = LEARN_STEP,
#             tau = TAU,
#             ) 
#     bar_format = '{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]'
#     pbar = trange(n_episodes, unit="ep", bar_format=bar_format, ascii=True)
    
#     for idx_epi in pbar:
#         state  = env.reset()
#         score = 0
#         for idx_step in range(max_steps):
#             action = agent.chooseAction(state, epsilon)
#             next_state, reward, done, _ = env.step(action)
#             agent.remember(state, action, reward, next_state, done, priority = 10)
#             state = next_state
#             score += reward

#             if done:
#                 break

#         score_hist.append(score)
#         score_avg = np.mean(score_hist[-100:])
#         epsilon = max(0.01, epsilon*0.995)

#         pbar.set_postfix_str(f"Score: {score: 7.2f}, 100 score avg: {score_avg: 7.2f}, Eval: {eval}")
#         pbar.update(0)

#         if idx_epi%20 == 0:
#             eval = evaluatePolicy(env=LunarLander(), agent=agent, loop=50)
#             eval_hist.append(eval)
#             eval_avg = np.mean(eval_hist[-5:])

#             if eval > 0.40:
#                 torch.save({
#                 'epoch': idx_epi,
#                 'model_state_dict': agent.policy_net.state_dict(),
#                 'optimizer_state_dict': agent.optimizer.state_dict(),
            
#                 }, "LLPolicy" + str(int(eval*100)))
                
#             eval_hist.append(eval)
#         if len(score_hist) >= 100:
#             if score_avg >= TARGET_SCORE:
#                 break

#     if (idx_epi+1) < n_episodes:
#         print("\nTarget Reached!")
#     else:
#         print("\nDone!")

#     return score_hist, eval_hist

# if __name__ == "__main__":
#     train()