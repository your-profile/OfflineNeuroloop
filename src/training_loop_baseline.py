import numpy as np
import pandas as pd
from typing import List, Optional
from src.networks.DDPG import DDPG

from src.models.model_training import ModelTrainer
from src.models.model_neural_predictor import FnirsFeaturePredictor
from src.networks.DQN import DQN
from src.envs.lunar_lander import LunarLander
from src.envs.flappy_bird import FlappyBirdEnv as FlappyBird
import gymnasium
import time
from tqdm import trange
from copy import deepcopy as dc
import numpy as np
import src.utils as utils
import torch
from src.neural.buffer import fNIRSBuffer
from src.rl_loop import utils_rl
from src.neural.preprocessing import DatasetProcessor

def train(env:gymnasium.Env, 
          task_df:pd.DataFrame, 
          agent: DQN, 
          clf, 
          processor: DatasetProcessor,
          ml: ModelTrainer,
          flags: list, 
          fnirs_channel_names: List[str], 
          episodes_num: int, 
          steps: int, 
          window_duration_s: float, 
          granularity: str,
          means: tuple,
          fnirs_rate_hz: float = 5.2, 
          shift: float = 0.0, 
          noise = 0.0,
          smoothing_window_size: int = 0,
          eval_update: int = 200, 
          buffer_type: str = 'ER', 
          seed: int = 42,
          beta: float = 1.0,
          epsilon: float = 0.1,
          save_results: bool = False, 
          save_to_csv: bool = False,
          verbose: bool = False,
          finetune_threshold = 0.0,
          success_save_threshold = 1.0,
          save_agent = False,
    ):

    start_time = time.time()

     # domain key
    domain_key = task_df["condition"].iloc[0][0]

    # Calculate total number of participant episodes
    total_participant_episodes = task_df.drop_duplicates(subset=["participantKey", "episode"]).shape[0]
    print("Total Participant Episodes: ", total_participant_episodes, "Flags: ", flags)

    # rewards, timesteps, success rate, optimality predictions
    all_average_rewards, all_total_rewards, all_episode_steps, all_episode_success, classes_truth, classes_pred = [],[],[],[],[],[]
    eval_success, combined_steps, combined_episodes = (0.0, 0, 0)

    # training progress bar
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt} | {postfix}]'
    pbar = trange(episodes_num, unit="ep", bar_format=bar_format, ascii=True)


    # ONLINE POST-TRAINING LOOP
    for online_episode in range(episodes_num):
        # set seed
        if domain_key == "F":
            state, _ = env.reset(seed=seed) #seed for flappy bird
        else:
            state = env.reset(seed=seed) #seed for lunar lander

        seed += 1 #increment seed

        # reset total reward and state action value
        total_reward = 0

        for online_step in range(steps):
            # choose action
            action, _ = agent.chooseAction(state, epsilon)
            
            # take action in env
            if domain_key == "F":
                next_state, reward, done, terminated, _ = env.step(action) #flappy bird
            else:
                terminated = False
                next_state, reward, done, _ = env.step(action) #lunar lander

            agent.remember(state=state, action=action, reward=reward, next_state=next_state, done=done, priority=None, q_augmentation=0.0)
            
            state = next_state
            total_reward += reward

            # evaluate agent
            if combined_steps % eval_update == 0:
                if domain_key == "F": #flappy bird
                    eval_reward, eval_success = utils_rl.evaluate(env=FlappyBird(score_limit=100), agent=agent, episodes=25, steps=steps, domain_key=domain_key)
                else: #lunar lander
                    eval_reward, eval_success = utils_rl.evaluate(env=LunarLander(), agent=agent, episodes=25, steps=steps, domain_key=domain_key)
                
                # store success rate
                all_episode_success.append(eval_success)
                all_total_rewards.extend(eval_reward)
                all_episode_steps.append(online_step)
                score_avg = np.mean(all_total_rewards[-200:])
           
            combined_steps += 1

            if done or terminated:                    
                break

        if eval_success >= success_save_threshold and save_agent:
            # save agent if above success save threshold
            torch.save({
                'episode': online_episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'target_model_state_dict': agent.target_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': epsilon,  
            }, f"{str(domain_key).upper()}Policy{str(int(eval_success*100))}")

        # bar update
        pbar.set_postfix(
            {"Score": f"{score_avg:7.2f}",
                "Eval": f"{eval_success:.3f}",
            }, refresh=True
        )          
        pbar.update(1)
        combined_episodes += 1

    # close environment
    pbar.close()
    env.close()

    if 0 not in flags:
        offline_model_report = ml.get_report(np.array(classes_truth), np.array(classes_pred), (granularity[0] != "c"))
        print("OFFLINE:\n", offline_model_report)

    if save_results:
        results = utils_rl.Results.save_results(experiment_list = flags, 
                                   episodes = total_participant_episodes, 
                                   total_rewards = all_total_rewards, 
                                   success_rate = all_episode_success,
                                   steps = all_episode_steps,
                                   save_to_csv = save_to_csv)

    print(f"Episode {online_episode}, Reward: {total_reward:.2f}, Success: {eval_success:.2f}")

    print("Summation of participant episodes seen: ", total_participant_episodes)
    print("Elapsed time in hours: ", (time.time() - start_time) / 3600)

    return results


def vectorize_action(x, dtype=np.float32):
    return np.asarray(x, dtype=dtype).ravel()

def train_robot(env:gymnasium.Env, 
          task_df:pd.DataFrame, 
          agent: DQN, 
          clf, 
          processor: DatasetProcessor,
          ml: ModelTrainer,
          flags: list, 
          fnirs_channel_names: List[str], 
          episodes_num: int, 
          steps: int, 
          window_duration_s: float, 
          granularity: str,
          means: tuple,
          fnirs_rate_hz: float = 5.2, 
          shift: float = 0.0, 
          noise = 0.0,
          smoothing_window_size: int = 0,
          eval_update: int = 200, 
          buffer_type: str = 'ER', 
          seed: int = 42,
          beta: float = 1.0,
          save_results: bool = False, 
          save_to_csv: bool = False,
          verbose: bool = False,
          finetune_threshold = 0.0,
          success_save_threshold = 1.0,
          save_agent = False,
    ):
    """
    Offline neuro + online Fetch (DDPG + HER) with the same experiment_list flags as ``train``
    """

    blank_episode_dict = {
            "state": [],
            "action": [],
            "reward": [],
            "next_state": [],
            "achieved_goal": [],
            "next_achieved_goal": [],
            "desired_goal": [],
            "q_augmentation": [],
            "transition_priority": [],
            "done": [],
        }

    start_time = time.time()

    # get total participant episodes
    total_participant_episodes = int(task_df["episode"].nunique())

    total_participant_episodes = task_df.drop_duplicates(subset=["participantKey", "episode"]).shape[0]
    print("Total Participant Episodes: ", total_participant_episodes)

    # initialize rewards, success, and classes
    all_total_rewards, all_episode_steps, all_episode_success, classes_truth, classes_pred = ([],[],[],[],[])
    eval_success, combined_steps, last_participant_episode, combined_episodes = 0.0, 0, 0, 0
    learning_rate = float(agent.actor_lr)

    # training progress bar
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt} | {postfix}]'
    pbar = trange(episodes_num, unit="ep", bar_format=bar_format, ascii=True)
    minibatch = []

    # ONLINE POST-TRAINING LOOP
    for online_episode in range(total_participant_episodes, episodes_num):
        state_dict, _ = env.reset(seed=seed)
        seed += 1
        total_reward = 0.0
        online_ep = dc(blank_episode_dict)

        for online_step in range(steps):
            state = state_dict["observation"].astype(np.float32).ravel()
            desired_goal = state_dict["desired_goal"].astype(np.float32).ravel()
            achieved_goal = state_dict["achieved_goal"].astype(np.float32).ravel()

            action = agent.choose_action(state, desired_goal, train_mode=True)
            next_state_dict, reward, terminated, truncated, info = env.step(action)
            
            done = float(terminated or truncated)
            next_state = next_state_dict["observation"].astype(np.float32).ravel()
            next_achieved_goal = next_state_dict["achieved_goal"].astype(np.float32).ravel()

            if buffer_type == "PER":
                priority = ddpg_priority(reward, action, desired_goal, next_achieved_goal)

            online_ep["state"].append(state)
            online_ep["action"].append(action.astype(np.float32))
            online_ep["reward"].append(float(reward))
            online_ep["next_state"].append(next_state)
            online_ep["achieved_goal"].append(achieved_goal)
            online_ep["next_achieved_goal"].append(next_achieved_goal)
            online_ep["desired_goal"].append(desired_goal)
            online_ep["done"].append(done)
            online_ep["q_augmentation"].append(float(0.0))
            online_ep["transition_priority"].append(priority)

            state_dict = next_state_dict
            total_reward += float(reward)

            if combined_steps % eval_update == 0 and save_agent:
                eval_success, eval_reward = utils_rl.evaluate_fetch(env, agent, steps=steps, episodes=10)
                all_episode_success.append(eval_success)
                all_total_rewards.append(eval_reward)
                all_episode_steps.append(online_step)
                score_avg = np.mean(all_total_rewards[-200:])

                if eval_success >= success_save_threshold:
                    agent.save_weights()
                    torch.save(
                        {
                            "episode": online_episode,
                            "actor": agent.actor.state_dict(),
                            "critic": agent.critic.state_dict(),
                            "actor_target": agent.actor_target.state_dict(),
                            "critic_target": agent.critic_target.state_dict(),
                            "actor_optim": agent.actor_optim.state_dict(),
                            "critic_optim": agent.critic_optim.state_dict(),
                        },
                        "FetchPolicy" + str(int(eval_success * 100)) + ".pth",
                    )

            if terminated or truncated:
                break

        minibatch.append(dc(online_ep))

        if len(minibatch) == 20:
            agent.store(minibatch)
            
            for _ in range(40): actor_loss, critic_loss = agent.train()

            agent.update_networks()
            minibatch = []

        # bar update
        pbar.set_postfix(
            {"Score": f"{score_avg:7.2f}",
                "Eval": f"{eval_success:.3f}",
            }, refresh=True
        )    

        pbar.update(1)
        combined_episodes += 1
 
    env.close()

    results = None

    if save_results:
        results = utils_rl.Results.save_results(experiment_list = flags, 
                episodes = total_participant_episodes, 
                total_rewards = all_total_rewards, 
                success_rate = all_episode_success,
                steps = all_episode_steps,
                save_to_csv = save_to_csv)

    print(f"Robot episode {last_participant_episode}, Reward: {total_reward:.2f}, Success: {eval_success:.2f}")
    print("Summation of participant episodes seen: ", total_participant_episodes)
    print("Elapsed time in hours: ", (time.time() - start_time) / 3600)

    return results


def dqn_priority(reward, action: int, action_dist, next_action_dist) -> float:
    """Discrete actions: same as reward + P(a|s_t) - P(a|s_{t+1}) over stored optimal distributions."""
    prev_action_value = next_action_dist[action] if action < len(next_action_dist) else 0.0
    curr_action_value = action_dist[action] if action < len(action_dist) else 0.0
    return float(reward + curr_action_value - prev_action_value)


def ddpg_priority(reward, action, action_dist, next_action_dist) -> float:
    """Continuous actions (e.g. Fetch): reward + score(a, opt_t) - score(a, opt_{t+1}) with score = -||a - opt||^2."""
    a = np.asarray(action, dtype=np.float64).ravel()
    opt_curr = np.asarray(action_dist, dtype=np.float64).ravel()
    opt_next = np.asarray(next_action_dist, dtype=np.float64).ravel()
    m = int(min(a.size, opt_curr.size, opt_next.size))
    if m == 0:
        return float(reward)
    curr_action_value = -np.sum((a[:m] - opt_curr[:m]) ** 2)
    prev_action_value = -np.sum((a[:m] - opt_next[:m]) ** 2)
    return float(reward + curr_action_value - prev_action_value)
