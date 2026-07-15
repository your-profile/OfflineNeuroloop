import numpy as np
import pandas as pd
from typing import List, Optional
from src.networks.DDPG import DDPG

from src.models.model_training import ModelTrainer
# from src.models.model_neural_predictor import FnirsFeaturePredictor
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
from src.seed_utils import begin_rl_training, set_global_seed

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
          success_save_threshold = 0.0,
          save_agent = False,
    ):

    set_global_seed(seed)

    start_time = time.time()

    # calculate window size + initialize buffer
    sample_period_s = 1.0 / fnirs_rate_hz
    buffer = fNIRSBuffer(window_duration_s=window_duration_s, sample_period_s=sample_period_s)
    grouped = task_df.groupby(['participantKey', 'episode'])
    
    # Calculate total number of participant episodes
    total_participant_episodes = task_df.drop_duplicates(subset=["participantKey", "episode"]).shape[0]
    print("Total Participant Episodes: ", total_participant_episodes, "Flags: ", flags)

    # domain key
    domain_key = task_df["condition"].iloc[0][0]
    end_tag_episodes = 500 #episodes to follow neural injection

    # granularity index
    if granularity[0] == "b": gr = 0
    if granularity[0] == "t": gr = 1
    if granularity[0] == "c": gr = 2

    # rewards, timesteps, success rate, optimality predictions
    all_total_rewards, all_episode_steps, all_episode_success, classes_truth, classes_pred = [],[],[],[],[]
    eval_success, combined_steps, combined_episodes = (0.0, 0, 0)

    # training progress bar
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt} | {postfix}]'
    pbar = trange(episodes_num+total_participant_episodes+end_tag_episodes, unit="ep", bar_format=bar_format, ascii=True)

    last_seed, score_avg = 0, 0.0
    online_seed = begin_rl_training(seed)

    # ONLINE PRE-TRAINING LOOP
    for online_episode in range(0, episodes_num):
        if domain_key == "F":
            threshold = score_avg / 50
        else:
            threshold = eval_success

        if threshold >= finetune_threshold:
            print(f"Online episode {online_episode} reached training threshold {finetune_threshold}")
            break

        # set seed
        if domain_key == "F":
            state, _ = env.reset(seed=online_seed) #seed for flappy bird
        else:
            state = env.reset(seed=online_seed) #seed for lunar lander

        online_seed += 1 #increment seed
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

            priority = utils_rl.td_priority(agent, "DQN", reward, action, state, next_state, done=done, buffer_type=buffer_type)

            agent.remember(state=state, action=action, reward=reward, next_state=next_state, done=done, priority=priority, q_augmentation=0.0)
            
            state = next_state
            total_reward += reward

            # evaluate agent
            if combined_steps % eval_update == 0:
                if domain_key == "F": #flappy bird
                    eval_reward, eval_success = utils_rl.evaluate(env=FlappyBird(score_limit=100), agent=agent, episodes=25, steps=steps, domain_key=domain_key, random_seed=seed)
                else: #lunar lander
                    eval_reward, eval_success = utils_rl.evaluate(env=LunarLander(), agent=agent, episodes=25, steps=steps, domain_key=domain_key, random_seed=seed)
                
                # store success rate
                all_episode_success.append(eval_success)
                all_total_rewards.extend(eval_reward)
                all_episode_steps.append(combined_steps)
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
                "Epsilon": f"{epsilon:.3f}",
            }, refresh=True
        )          
        pbar.update(1)
        combined_episodes += 1
    last_seed = online_seed
    starting_neural_injection = online_episode

    # OFFLINE DATASET PRE-TRAINING LOOP
    print(f"Offline dataset pre-training loop started")
    for (participant, episode), episode_df in grouped:  
        total_reward = 0.0
        done = False

        # get episode data
        rows = episode_df.reset_index(drop=True)
        n = len(rows)

        off_seed = int(rows["seed"].iloc[0])

        if domain_key == "F":
            state, _ = env.reset(seed=off_seed) #seed for flappy bird
        else:
            state = env.reset(seed=off_seed) #seed for lunar lander

        for offline_step in range(1, n):
            action = rows["actions"].iloc[offline_step]
            action_dist = rows["optimal_actions"].iloc[offline_step]
            reward_dataset = rows["rewards"].iloc[offline_step]
            state_dataset = rows["states"].iloc[offline_step]
            final_step = rows["steps"].iloc[offline_step]

            q_augmentation = 0.0

            # if action is Nan, skip
            if action != action or action_dist is None:
                action = 0
                action_dist = 0
            else:
                action = int(action)

            # take action in env
            if domain_key == "F":
                next_state, reward, done, terminated, _ = env.step(action) #flappy bird
            else:
                terminated = False
                next_state, reward, done, _ = env.step(action) #lunar lander

            fs = int(final_step) if pd.notna(final_step) else n
            if offline_step < fs - 1 and offline_step + 1 < n:
                next_action_dist = rows["optimal_actions"].iloc[offline_step + 1]
            else:
                next_action_dist = action_dist

            priority = utils_rl.td_priority(agent, "DQN", reward, action, state, next_state, done=done, buffer_type=buffer_type)

            if 0 not in flags:
                # get rl task statistic tuple (state, action, reward) timestamp
                rl_timestamp = rows["time"].iloc[offline_step]

                # get associated fNIRS sample given timestep
                neural_features = buffer.get_features()
                neural_signal, clf_probs = utils_rl.get_neural_signal(features = neural_features, clf = clf)

                if gr == 2:
                    clf_probs = "regression"               

                # update neural buffer
                fnirs_sample = processor.get_fnirs_sample(timestamp = rl_timestamp, temporal_shift = -shift, fnirs_channels = fnirs_channel_names)
                buffer.add_sample(timestamp = rl_timestamp, x = fnirs_sample, classification=neural_signal)
                
                # get + adjust neural classification
                new_neural_signal = buffer.get_neural_credit(granularity = granularity, X = smoothing_window_size)

                # add noise to neural classification
                if noise > 0.0:
                    new_neural_signal = ml.noisy_output(clf,  new_neural_signal, granularity, flip_rate = noise)

                # get true sample label
                class_truth = processor.get_label_sample(timestamp = rl_timestamp, temporal_shift = -shift)
                
                # Reward Augmentation Experiment
                if 1 in flags:
                    if verbose:
                        print(f"Experiment Condition 1: Reward Augmentation -- Episode {episode} -- Participant: {participant}")
                        print("Original Reward: ", reward, "| Neural Signal: ", new_neural_signal)
                    reward = utils_rl.adjust_signal(reward, new_neural_signal, clf_probs = clf_probs, means = means, beta = beta)
                
                # Priorirization experiment
                if 2 in flags:
                    if verbose:
                        print(f"Experiment Condition 2: Prioritization -- Episode {episode} -- Participant: {participant}")
                        print("Original Priority: ", abs(priority), "| Neural Signal: ", new_neural_signal)
                    priority = abs(priority)
                    priority = utils_rl.adjust_signal(priority, new_neural_signal, clf_probs = clf_probs, beta = beta)

                # Q Augmentation Experiment
                if 3 in flags:
                    if verbose:
                        print(f"Experiment Condition 3: Q-Augmentation -- Episode {episode} -- Participant: {participant}")
                    q_augmentation = utils_rl.adjust_signal(0.0, new_neural_signal, clf_probs = clf_probs, beta = beta)

                 # store sample optimality prediction and truth
                if smoothing_window_size > 1 or noise > 0.0:
                    classes_pred.append(new_neural_signal) #predictions that have been altered
                else:
                    classes_pred.append(neural_signal) #raw predictions
                
                classes_truth.append(class_truth.to_list()[gr]) #truth
                
            agent.remember(state, action, reward, next_state, done, priority = priority, q_augmentation = q_augmentation)
            state = next_state
            # evaluate agent
            if combined_steps % eval_update == 0:
                if domain_key == "F": #flappy bird
                    eval_reward, eval_success = utils_rl.evaluate(env=FlappyBird(score_limit=100), agent=agent, episodes=25, steps=steps, domain_key=domain_key, random_seed=seed)
                else: #lunar lander
                    eval_reward, eval_success = utils_rl.evaluate(env=LunarLander(), agent=agent, episodes=25, steps=steps, domain_key=domain_key, random_seed=seed)
                
                # store success rate
                all_episode_success.append(eval_success)
                all_total_rewards.extend(eval_reward)
                all_episode_steps.append(combined_steps)
                score_avg = np.mean(all_total_rewards[-200:])

            combined_steps += 1

        # bar update
        pbar.set_postfix(
            {"Score": f"{score_avg:7.2f}",
                "Eval": f"{eval_success:.3f}",
                "Epsilon": f"{epsilon:.3f}",
            }, refresh=True
        )          
        pbar.update(1)

    print(f"Offline dataset pre-training loop completed")
    print(f"Online post-training loop started")

    last_online_episode = online_episode

    # epsilon = 0.1

    # ONLINE POST-TRAINING LOOP
    for online_episode in range(last_online_episode, episodes_num+end_tag_episodes):

        # set seed
        if domain_key == "F":
            state, _ = env.reset(seed=last_seed) #seed for flappy bird
        else:
            state = env.reset(seed=last_seed) #seed for lunar lander

        last_seed += 1 #increment seed

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
                # if online_step == steps-1:
                #     reward -= 100

            priority = utils_rl.td_priority(agent, "DQN", reward, action, state, next_state, done=done, buffer_type=buffer_type)

            agent.remember(state=state, action=action, reward=reward, next_state=next_state, done=done, priority=priority, q_augmentation=0.0)
            
            state = next_state
            total_reward += reward
            # evaluate agent
            if combined_steps % eval_update == 0:
                if domain_key == "F": #flappy bird
                    eval_reward, eval_success = utils_rl.evaluate(env=FlappyBird(score_limit=100), agent=agent, episodes=25, steps=steps, domain_key=domain_key, random_seed=seed)
                else: #lunar lander
                    eval_reward, eval_success = utils_rl.evaluate(env=LunarLander(), agent=agent, episodes=25, steps=steps, domain_key=domain_key, random_seed=seed)
                
                # store success rate
                all_episode_success.append(eval_success)
                all_total_rewards.extend(eval_reward)
                all_episode_steps.append(combined_steps)
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
                "Epsilon": f"{epsilon:.3f}",
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
                                   index_of_interest = starting_neural_injection,
                                   save_to_csv = save_to_csv)

    print(f"Episode {episode}, Reward: {total_reward:.2f}, Success: {eval_success:.2f}")

    print("Summation of participant episodes seen: ", total_participant_episodes)
    print("Elapsed time in hours: ", (time.time() - start_time) / 3600)

    return results


def train_robot(env:gymnasium.Env, 
          task_df:pd.DataFrame, 
          agent: DDPG, 
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
          #epsilon
          save_results: bool = False, 
          save_to_csv: bool = False,
          verbose: bool = False,
          finetune_threshold = 0.0,
          success_save_threshold = 0.0,
          save_agent = False,
    ):
    """
    Offline neuro + online Fetch (DDPG + HER) with the same experiment_list flags as ``train``
    """

    set_global_seed(seed)

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
    robot_df = task_df[task_df["desired_goal"].notna()].copy()
    
    # calculate window size + initialize buffer
    sample_period_s = 1.0 / fnirs_rate_hz
    buffer = fNIRSBuffer(window_duration_s=window_duration_s, sample_period_s=sample_period_s)
    grouped = robot_df.groupby(["participantKey", "episode"])

    # Calculate total number of participant episodes
    total_participant_episodes = robot_df.drop_duplicates(subset=["participantKey", "episode"]).shape[0]
    print("Total Participant Episodes: ", total_participant_episodes, "Flags: ", flags)

    # check if robot dataframe is empty
    # if robot_df.empty:
        # rw_df = task_df[task_df["participantKey"].astype(str).str.contains("RW", na=False)].copy()
    if robot_df.empty:
        raise ValueError("No robot rows in task_df (need desired_goal or RW in participantKey).")

    end_tag_episodes = 5000 #episodes to follow neural injection

    # granularity index
    if granularity[0] == "b": gr = 0
    if granularity[0] == "t": gr = 1
    if granularity[0] == "c": gr = 2


    # rewards, timesteps, success rate, optimality predictions
    all_total_rewards, all_episode_steps, all_episode_success, classes_truth, classes_pred = [],[],[],[],[]
    eval_success, combined_steps, combined_episodes = (0.0, 0, 0)

    # training progress bar
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt} | {postfix}]'
    pbar = trange(episodes_num+total_participant_episodes+end_tag_episodes, unit="ep", bar_format=bar_format, ascii=True)
    
    last_seed, score_avg = 0, 0.0
    online_seed = begin_rl_training(seed)
    minibatch = []

    # ONLINE PRE-TRAINING LOOP
    for online_episode in range(0, episodes_num):
        
        if eval_success >= finetune_threshold:
            print(f"Online episode {online_episode} reached training threshold {finetune_threshold}")
            break

        state_dict, _ = env.reset(seed=online_seed)

        online_seed += 1
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

            priority = utils_rl.td_priority(agent, "DDPG", float(reward), action, state, next_state, goal=desired_goal, buffer_type=buffer_type)

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
            total_reward += float(reward)

            if combined_steps % eval_update == 0:
                eval_success, eval_reward = utils_rl.evaluate_fetch(utils.make_fetch_env(), agent, steps=steps, episodes=25, random_seed=seed)
                
                # store success rate
                all_episode_success.append(eval_success)
                all_total_rewards.extend(eval_reward)
                all_episode_steps.append(combined_steps)
                score_avg = np.mean(all_total_rewards[-200:])
            state_dict = next_state_dict
            combined_steps += 1

            if terminated or truncated:
                break

        if eval_success >= success_save_threshold and save_agent:
            agent.save_weights()
            torch.save(
                {
                    "episode": episode,
                    "actor": agent.actor.state_dict(),
                    "critic": agent.critic.state_dict(),
                    "actor_target": agent.actor_target.state_dict(),
                    "critic_target": agent.critic_target.state_dict(),
                    "actor_optim": agent.actor_optim.state_dict(),
                    "critic_optim": agent.critic_optim.state_dict(),
                },
                "FetchPolicy" + str(int(eval_success * 100)) + ".pth",
            )

        minibatch.append(dc(online_ep))

        if len(minibatch) == 20:
            agent.store(minibatch)
            
            for _ in range(10): actor_loss, critic_loss = agent.train()

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

    last_seed = online_seed
    index_of_interest = online_episode

    # OFFLINE DATASET PRE-TRAINING LOOP
    for (participant, episode), episode_df in grouped:
        total_reward = 0.0
        done = False

        # get episode data
        episode_dict = dc(blank_episode_dict)
        rows = episode_df.reset_index(drop=True)
        n = len(rows)

        # get seed
        offline_seed = int(rows["seed"].iloc[0])
        dataset_state = rows["states"].iloc[0]

        # reset environment
        state_dict, _ = env.reset(seed=offline_seed)
        state = state_dict["observation"]
        achieved_goal = state_dict["achieved_goal"]
        desired_goal = state_dict["desired_goal"]

        # OFFLINE DATASET PRE-TRAINING LOOP
        for offline_step in range(0, n):
            state = state_dict["observation"]
            achieved_goal = state_dict["achieved_goal"]
            desired_goal = state_dict["desired_goal"]

            action = vectorize_action(rows["actions"].iloc[offline_step])
            action_dist = rows["optimal_actions"].iloc[offline_step]
            reward_dataset = rows["rewards"].iloc[offline_step]
            state_dataset = rows["states"].iloc[offline_step]
            final_step = rows["steps"].iloc[offline_step]

            # initialize q augmentation
            q_augmentation = 0.0

            if action.size == 0 or not np.isfinite(action).all():
                continue

            next_state_dict, reward, done, terminated, info = env.step(action)
            next_state = next_state_dict["observation"]
            next_achieved_goal = next_state_dict["achieved_goal"]
            next_desired_goal = next_state_dict["desired_goal"]

            # get next state unless episode ends
            if offline_step + 1 < n:
                next_state_dataset = rows["states"].iloc[offline_step + 1]
                next_action_dist = rows["optimal_actions"].iloc[offline_step + 1]
            else:
                next_state_dataset = rows["states"].iloc[offline_step]
                next_action_dist = action_dist

            priority = utils_rl.td_priority(agent, "DDPG", float(reward), action, state, next_state, goal=desired_goal, buffer_type=buffer_type)

            if 0 not in flags:
                # get associated fNIRS sample given timestep
                rl_timestamp = rows["time"].iloc[offline_step]
                neural_features = buffer.get_features()
                neural_signal, clf_probs = utils_rl.get_neural_signal(features = neural_features, clf = clf)

                if gr == 2:
                    clf_probs = "regression"               

                # update neural buffer
                fnirs_sample = processor.get_fnirs_sample(timestamp = rl_timestamp, temporal_shift = -shift, fnirs_channels = fnirs_channel_names)
                buffer.add_sample(timestamp = rl_timestamp, x = fnirs_sample, classification=neural_signal)
                
                # get + adjust neural classification
                new_neural_signal = buffer.get_neural_credit(granularity = granularity, X = smoothing_window_size)

                # add noise to neural classification
                if noise > 0.0:
                    new_neural_signal = ml.noisy_output(clf,  new_neural_signal, granularity, flip_rate = noise)

                # Reward Augmentation Experiment
                if 1 in flags:
                    if verbose: 
                        print(f"Reward Augmentation — ep {episode} participant {participant}")
                        print("Original Reward: ", reward, "| Neural Signal: ", new_neural_signal, "| Adjusted Reward: ")
                    reward = utils_rl.adjust_signal(reward, new_neural_signal, clf_probs = clf_probs, means = means, beta = beta)

                # Priorirization experiment
                if 2 in flags:
                    if verbose:
                        print(f"Prioritization — ep {episode} participant {participant}")
                        print("Original Priority: ", abs(priority), "| Neural Signal: ", new_neural_signal, "| Adjusted Priority: ")
                    priority = abs(priority)
                    priority = utils_rl.adjust_signal(priority, new_neural_signal, clf_probs = clf_probs, beta = beta)

                # Q Augmentation Experiment
                if 3 in flags:
                    if verbose:
                        print(f"Q-aug analogue — ep {episode} participant {participant}")
                        print("Neural Signal: ", new_neural_signal, "| Q-Value: ", reward)
                    q_augmentation = utils_rl.adjust_signal(0.0, new_neural_signal, clf_probs = clf_probs, beta = beta)

                if smoothing_window_size > 1 or noise > 0.0:
                    classes_pred.append(new_neural_signal)
                else:
                    classes_pred.append(neural_signal)

            episode_dict["state"].append(state)
            episode_dict["action"].append(action.astype(np.float32))
            episode_dict["reward"].append(reward)
            episode_dict["next_state"].append(next_state.astype(np.float32))
            episode_dict["achieved_goal"].append(achieved_goal.astype(np.float32))
            episode_dict["next_achieved_goal"].append(next_achieved_goal.astype(np.float32))
            episode_dict["desired_goal"].append(desired_goal.astype(np.float32))
            episode_dict["transition_priority"].append(priority)
            episode_dict["q_augmentation"].append(float(q_augmentation))
            episode_dict["done"].append(float(done))

            if combined_steps % eval_update == 0:
                eval_success, eval_reward = utils_rl.evaluate_fetch(utils.make_fetch_env(), agent, steps=steps, episodes=25, random_seed=seed)
                all_episode_success.append(eval_success)
                all_total_rewards.extend(eval_reward)
                all_episode_steps.append(combined_steps)
                score_avg = np.mean(all_total_rewards[-200:])

                if eval_success >= success_save_threshold:
                    agent.save_weights()
                    torch.save(
                        {
                            "episode": episode,
                            "actor": agent.actor.state_dict(),
                            "critic": agent.critic.state_dict(),
                            "actor_target": agent.actor_target.state_dict(),
                            "critic_target": agent.critic_target.state_dict(),
                            "actor_optim": agent.actor_optim.state_dict(),
                            "critic_optim": agent.critic_optim.state_dict(),
                        },
                        "FetchPolicy" + str(int(eval_success * 100)) + ".pth",
                    )
            state_dict = next_state_dict
            combined_steps += 1
        
        minibatch.append(dc(episode_dict))

        if len(minibatch) == 20:
            agent.store(minibatch)
            for _ in range(10):
                actor_loss, critic_loss = agent.train()
            
            agent.update_networks()
            minibatch = []

        # bar update
        pbar.set_postfix(
            {"Score": f"{score_avg:7.2f}",
                "Eval": f"{eval_success:.3f}",
            }, refresh=True
        )    

        pbar.update(1)
    
    last_online_episode = online_episode
    
    # ONLINE POST-TRAINING LOOP
    for online_episode in range(last_online_episode, episodes_num+end_tag_episodes):
        state_dict, _ = env.reset(seed=online_seed)

        online_seed += 1
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

            priority = utils_rl.td_priority(agent, "DDPG", float(reward), action, state, next_state, goal=desired_goal, buffer_type=buffer_type)

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
            total_reward += float(reward)

            if combined_steps % eval_update == 0:
                eval_success, eval_reward = utils_rl.evaluate_fetch(utils.make_fetch_env(), agent, steps=steps, episodes=25, random_seed=seed)
                
                # store success rate
                all_episode_success.append(eval_success)
                all_total_rewards.extend(eval_reward)
                all_episode_steps.append(combined_steps)
                score_avg = np.mean(all_total_rewards[-200:])
            state_dict = next_state_dict
            combined_steps += 1
            if terminated or truncated:
                break

        if eval_success >= success_save_threshold and save_agent:
            agent.save_weights()
            torch.save(
                {
                    "episode": episode,
                    "actor": agent.actor.state_dict(),
                    "critic": agent.critic.state_dict(),
                    "actor_target": agent.actor_target.state_dict(),
                    "critic_target": agent.critic_target.state_dict(),
                    "actor_optim": agent.actor_optim.state_dict(),
                    "critic_optim": agent.critic_optim.state_dict(),
                },
                "FetchPolicy" + str(int(eval_success * 100)) + ".pth",
            )

        minibatch.append(dc(online_ep))

        if len(minibatch) == 20:
            agent.store(minibatch)
            
            for _ in range(10): actor_loss, critic_loss = agent.train()

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
                index_of_interest = index_of_interest,
                save_to_csv = save_to_csv)

    print(f"Robot episode {online_episode}, Reward: {total_reward:.2f}, Success: {eval_success:.2f}")
    print("Summation of participant episodes seen: ", total_participant_episodes)
    print("Elapsed time in hours: ", (time.time() - start_time) / 3600)

    return results


def vectorize_action(x, dtype=np.float32):
    return np.asarray(x, dtype=dtype).ravel()
