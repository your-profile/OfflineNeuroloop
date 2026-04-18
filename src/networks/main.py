import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.utils import make_fetch_env
from DDPG import DDPG as Agent
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from copy import deepcopy as dc
import os
import torch

np.random.seed(np.random.randint(0, 20000))

Train = True
Play_FLAG = False
seed = 42
MAX_EPOCHS = 200
MAX_CYCLES = 50
num_updates = 40
MAX_EPISODES = 16
memory_size = 7e+5
batch_size = 256
actor_lr = 1e-3
critic_lr = 1e-3
gamma = 0.98
tau = 0.05
k_future = 4

test_env = make_fetch_env(max_episode_steps=50, mujoco_version=4)
state_shape = test_env.observation_space.spaces["observation"].shape
n_actions = test_env.action_space.shape[0]
n_goals = test_env.observation_space.spaces["desired_goal"].shape[0]
action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]

def eval_agent(env_, agent_, steps=50, episodes = 20):
    total_success_rate = []
    running_r = []
    for ep in range(10):
        per_success_rate = []
        env_dictionary, _ = env_.reset()
        s = env_dictionary["observation"]
        ag = env_dictionary["achieved_goal"]
        g = env_dictionary["desired_goal"]
        while np.linalg.norm(ag - g) <= 0.05:
            env_dictionary, _ = env_.reset()
            s = env_dictionary["observation"]
            ag = env_dictionary["achieved_goal"]
            g = env_dictionary["desired_goal"]
        ep_r = 0
        for t in range(steps):
            with torch.no_grad():
                a = agent_.choose_action(s, g, train_mode=False)
            observation_new, r, done, info_, info = env_.step(a)
            s = observation_new['observation']
            g = observation_new['desired_goal']
            succ = 0
            if info["is_success"] > 0.0:
                succ = 1
                print("win")
            per_success_rate.append(succ)
            ep_r += r
        total_success_rate.append(per_success_rate)
        if ep == 0:
            running_r.append(ep_r)
        else:
            running_r.append(running_r[-1] * 0.99 + 0.01 * ep_r)
    total_success_rate = np.array(total_success_rate)
    local_success_rate = np.mean(total_success_rate)
    return local_success_rate, running_r, ep_r

env = make_fetch_env(max_episode_steps=50, mujoco_version=4)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
agent = Agent(n_states=state_shape,
              n_actions=n_actions,
              n_goals=n_goals,
              action_bounds=action_bounds,
              capacity=memory_size,
              action_size=n_actions,
              batch_size=batch_size,
              actor_lr=actor_lr,
              critic_lr=critic_lr,
              gamma=gamma,
              tau=tau,
              k_future=k_future,
              env=dc(env))

def run():

    total_success_rate = []
    total_ac_loss = []
    total_cr_loss = []
    rewards = []

    for epoch in range(MAX_EPOCHS):
        print("Epoch: ", epoch)
        start_time = time.time()
        epoch_actor_loss = 0
        epoch_critic_loss = 0

        for cycle in range(0, MAX_CYCLES):
            print("Cycle: ", cycle)
            minibatch = []
            cycle_actor_loss = 0
            cycle_critic_loss = 0
            for episode_num in range(MAX_EPISODES):
                episode_dict = {
                    "state": [],
                    "action": [],
                    "reward": [],
                    "info": [],
                    "achieved_goal": [],
                    "desired_goal": [],
                    "next_state": [],
                    "next_achieved_goal": []}
                env_dict, _ = env.reset()
                state = env_dict["observation"]
                achieved_goal = env_dict["achieved_goal"]
                desired_goal = env_dict["desired_goal"]
                while np.linalg.norm(achieved_goal - desired_goal) <= 0.05:
                    env_dict, _ = env.reset()
                    state = env_dict["observation"]
                    achieved_goal = env_dict["achieved_goal"]
                    desired_goal = env_dict["desired_goal"]
                for t in range(50):
                    action = agent.choose_action(state, desired_goal)
                    next_env_dict, reward, done, info, _ = env.step(action)

                    next_state = next_env_dict["observation"]
                    next_achieved_goal = next_env_dict["achieved_goal"]
                    next_desired_goal = next_env_dict["desired_goal"]

                    episode_dict["state"].append(state.copy())
                    episode_dict["action"].append(action.copy())
                    episode_dict["reward"].append(reward)
                    episode_dict["achieved_goal"].append(achieved_goal.copy())
                    episode_dict["desired_goal"].append(desired_goal.copy())

                    state = next_state.copy()
                    achieved_goal = next_achieved_goal.copy()
                    desired_goal = next_desired_goal.copy()

                episode_dict["state"].append(state.copy())
                episode_dict["reward"].append(reward)
                episode_dict["achieved_goal"].append(achieved_goal.copy())
                episode_dict["desired_goal"].append(desired_goal.copy())
                episode_dict["next_state"] = episode_dict["state"][1:]
                episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
                minibatch.append(dc(episode_dict))

            agent.store(minibatch)
            for _ in range(num_updates):
                actor_loss, critic_loss = agent.train()
                cycle_actor_loss += actor_loss
                cycle_critic_loss += critic_loss

            epoch_actor_loss += cycle_actor_loss / num_updates
            epoch_critic_loss += cycle_critic_loss /num_updates
            agent.update_networks()

        success_rate, running_reward, episode_reward = eval_agent(env, agent)
        total_ac_loss.append(epoch_actor_loss)
        total_cr_loss.append(epoch_critic_loss)
        total_success_rate.append(success_rate)

        print(f"Epoch:{epoch}| "
                f"Running_reward:{running_reward[-1]:.3f}| "
                f"EP_reward:{episode_reward:.3f}| "
                f"Memory_length:{len(agent.memory)}| "
                f"Duration:{time.time() - start_time:.3f}| "
                f"Actor_Loss:{actor_loss:.3f}| "
                f"Critic_Loss:{critic_loss:.3f}| "
                f"Success rate:{success_rate:.3f}| ")
        agent.save_weights()
            
    plt.plot(total_success_rate)


def run2():

    total_success_rate = []
    total_ac_loss = []
    total_cr_loss = []
    rewards = []

    start_time = time.time()
    epoch_actor_loss, epoch_critic_loss = 0, 0
    minibatch = []
    cycle_actor_loss = 0
    cycle_critic_loss = 0
    for episode_num in range(MAX_EPISODES):
        episode_dict = {
            "state": [],
            "action": [],
            "reward": [],
            "info": [],
            "achieved_goal": [],
            "desired_goal": [],
            "next_state": [],
            "next_achieved_goal": []}
        env_dict, _ = env.reset()
        state = env_dict["observation"]
        achieved_goal = env_dict["achieved_goal"]
        desired_goal = env_dict["desired_goal"]
        while np.linalg.norm(achieved_goal - desired_goal) <= 0.05:
            env_dict, _ = env.reset()
            state = env_dict["observation"]
            achieved_goal = env_dict["achieved_goal"]
            desired_goal = env_dict["desired_goal"]
        for t in range(50):
            action = agent.choose_action(state, desired_goal)
            next_env_dict, reward, done, info, _ = env.step(action)

            next_state = next_env_dict["observation"]
            next_achieved_goal = next_env_dict["achieved_goal"]
            next_desired_goal = next_env_dict["desired_goal"]

            episode_dict["state"].append(state.copy())
            episode_dict["action"].append(action.copy())
            episode_dict["reward"].append(reward)
            episode_dict["achieved_goal"].append(achieved_goal.copy())
            episode_dict["desired_goal"].append(desired_goal.copy())

            state = next_state.copy()
            achieved_goal = next_achieved_goal.copy()
            desired_goal = next_desired_goal.copy()

        episode_dict["state"].append(state.copy())
        episode_dict["reward"].append(reward)
        episode_dict["achieved_goal"].append(achieved_goal.copy())
        episode_dict["desired_goal"].append(desired_goal.copy())
        episode_dict["next_state"] = episode_dict["state"][1:]
        episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
        minibatch.append(dc(episode_dict))

        if episode_num % MAX_CYCLES == 0:
            agent.store(minibatch)
            for _ in range(num_updates):
                actor_loss, critic_loss = agent.train()
                cycle_actor_loss += actor_loss
                cycle_critic_loss += critic_loss

            epoch_actor_loss += cycle_actor_loss / num_updates
            epoch_critic_loss += cycle_critic_loss /num_updates
            agent.update_networks()
        if episode_num % MAX_EPOCHS == 0:
            success_rate, running_reward, episode_reward = eval_agent(env, agent)
            total_ac_loss.append(epoch_actor_loss)
            total_cr_loss.append(epoch_critic_loss)
            total_success_rate.append(success_rate)

            print(f"Epoch:{episode_num}| "
                    f"Running_reward:{running_reward[-1]:.3f}| "
                    f"EP_reward:{episode_reward:.3f}| "
                    f"Memory_length:{len(agent.memory)}| "
                    f"Duration:{time.time() - start_time:.3f}| "
                    f"Actor_Loss:{actor_loss:.3f}| "
                    f"Critic_Loss:{critic_loss:.3f}| "
                    f"Success rate:{success_rate:.3f}| ")
            agent.save_weights()
            
    plt.plot(total_success_rate)

run()
