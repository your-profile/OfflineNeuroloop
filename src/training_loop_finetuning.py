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
          fnirs_rate_hz: float = 5.2, 
          shift: float = 0.0, 
          noise = 0.0,
          smoothing_window_size: int = 0,
          target_update: int = 200, 
          buffer_type: str = 'ER', 
          seed: int = 42,
          beta: float = 1.0,
          save_results: bool = False, 
          save_to_csv: bool = False,
          verbose: bool = False,
          fnirs_predictor = None,
    ):

    start_time = time.time()
    epsilon = 0.1

    # calculate window size + initialize buffer
    sample_period_s = 1.0 / fnirs_rate_hz
    buffer = fNIRSBuffer(window_duration_s=window_duration_s, sample_period_s=sample_period_s)
    grouped = task_df.groupby(['participantKey', 'episode'])
    
    # Calculate total number of participant episodes
    total_participant_episodes = task_df.drop_duplicates(subset=["participantKey", "episode"]).shape[0]
    print("Total Participant Episodes: ", total_participant_episodes, "Flags: ", flags)

    # domain key
    domain_key = task_df["condition"].iloc[0][0]

    # granularity index
    if granularity[0] == "b": gr = 0
    if granularity[0] == "t": gr = 1
    if granularity[0] == "c": gr = 2

    if domain_key.lower() == "l":
        means = (1.4, -0.95, -2.6)
    if domain_key.lower() == "f":
        means = (0.75, -0.1, -0.75)

    # rewards, timesteps, success rate, optimality predictions
    all_average_rewards, all_total_rewards, all_episode_steps, all_episode_success, classes_truth, classes_pred = [],[],[],[],[],[]
    success, combined_steps= (0.0, 0)

    # training progress bar
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt} | {postfix}]'
    pbar = trange(episodes_num, unit="ep", bar_format=bar_format, ascii=True)

    # episode intervals for online training
    online_episode_num = int(episodes_num - total_participant_episodes)

    # OFFLINE DATASET TRAINING LOOP
    for (participant, episode), episode_df in grouped:  
        total_reward = 0
        done = False

        # get episode data
        rows = episode_df.reset_index(drop=True)
        n = len(rows)

        for offline_step in range(n):
            action = rows["actions"].iloc[offline_step]
            action_dist = rows["optimal_actions"].iloc[offline_step]
            reward = rows["rewards"].iloc[offline_step]
            state = rows["states"].iloc[offline_step]
            final_step = rows["steps"].iloc[offline_step]
            priority = None
            q_augmentation = 0.0

            # if action is Nan, skip
            if action != action or action_dist is None:
                continue
            else:
                action = int(action)

            # get next state unless episode ends
            if offline_step + 1 < n:
                done = False
                next_state = rows["states"].iloc[offline_step + 1]
            else:
                done = True
                next_state = state

            if 0 not in flags:
                # get rl task statistic tuple (state, action, reward) timestamp
                rl_timestamp = rows["time"].iloc[offline_step]

                # get associated fNIRS sample given timestep
                neural_features = buffer.get_features()
                neural_signal, clf_probs = utils_rl.get_neural_signal(features = neural_features, clf = clf)

                # update neural buffer
                fnirs_sample = processor.get_fnirs_sample(timestamp = rl_timestamp, temporal_shift = -shift, fnirs_channels = fnirs_channel_names)
                buffer.add_sample(timestamp = rl_timestamp, x = fnirs_sample, classification=neural_signal)
                
                # get + adjust neural classification
                new_neural_signal = buffer.get_neural_credit(granularity = granularity, X = smoothing_window_size)

                # add noise to neural classification
                if noise > 0.0:
                    new_neural_signal = ml.noisy_output(clf,  new_neural_signal, granularity, flip_rate = noise)

                # adjust neural classification - scales with beta and makes 0 class and 1 class negative
                adjusted_neural_signal = utils_rl.adjust_neural_classification(new_neural_signal, beta=beta)

                # get true sample label
                class_truth = processor.get_label_sample(timestamp = rl_timestamp, temporal_shift = -shift)
                
                # get next action distribution, unless episode ends
                fs = int(final_step) if pd.notna(final_step) else n

                if offline_step < fs - 1 and offline_step + 1 < n:
                    next_action_dist = rows["optimal_actions"].iloc[offline_step + 1]
                else:
                    next_action_dist = action_dist

                priority = dqn_priority(reward, action, action_dist, next_action_dist)

                # Reward Augmentation Experiment
                if 1 in flags:
                    if verbose:
                        print(f"Experiment Condition 1: Reward Augmentation -- Episode {episode} -- Participant: {participant}")
                        print("Original Reward: ", reward, "| Neural Signal: ", new_neural_signal, "| Adjusted Reward: ", reward + adjusted_neural_signal)
                    reward = utils_rl.adjust_reward(reward, new_neural_signal, clf_probs = clf_probs, means = means)
                
                # Priorirization experiment
                if 2 in flags:
                    if verbose:
                        print(f"Experiment Condition 2: Prioritization -- Episode {episode} -- Participant: {participant}")
                        print("Original Priority: ", abs(priority), "| Neural Signal: ", new_neural_signal, "| Adjusted Priority: ", abs(priority) + adjusted_neural_signal)
                    priority = abs(priority)
                    priority += adjusted_neural_signal

                # Q Augmentation Experiment
                if 3 in flags:
                    if verbose:
                        print(f"Experiment Condition 3: Q-Augmentation -- Episode {episode} -- Participant: {participant}")
                    q_augmentation = utils_rl.adjust_reward(0.0, new_neural_signal, clf_probs = clf_probs)

                 # store sample optimality prediction and truth
                if smoothing_window_size > 1 or noise > 0.0:
                    classes_pred.append(new_neural_signal) #predictions that have been altered
                else:
                    classes_pred.append(neural_signal) #raw predictions
                
                classes_truth.append(class_truth.to_list()[gr]) #truth
                
            # set priority to 0 for ER buffer functionality
            if buffer_type == "ER":
                priority = 0.0

            # remember transition
            agent.remember(state, action, reward, next_state, done, priority = priority, q_augmentation = q_augmentation)
            # evaluate agent
            if combined_steps % target_update == 0:
                if domain_key == "F": #flappy bird
                    eval_reward, eval_success = utils_rl.evaluate(env=FlappyBird(score_limit=20), agent=agent, episodes=20, steps=steps, domain_key=domain_key)
                else: #lunar lander
                    eval_reward, eval_success = utils_rl.evaluate(env=LunarLander(), agent=agent, episodes=20, steps=steps, domain_key=domain_key)
                
                # store success rate
                all_episode_success.append(eval_success)
                all_total_rewards.extend(eval_reward)
                all_episode_steps.append(offline_step)
                score_avg = np.mean(all_total_rewards[-200:])

            combined_steps += 1

        # bar update
        pbar.set_postfix(
            {"Score": f"{score_avg:7.2f}",
                "Eval": f"{eval_success:.3f}",
            }, refresh=True
        )          
        pbar.update(1)



    # ONLINE TRAINING LOOP
    for online_step in range(0, online_episode_num):
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
            if combined_steps % target_update == 0:
                if domain_key == "F": #flappy bird
                    eval_reward, eval_success = utils_rl.evaluate(env=FlappyBird(score_limit=20), agent=agent, episodes=20, steps=steps, domain_key=domain_key)
                else: #lunar lander
                    eval_reward, eval_success = utils_rl.evaluate(env=LunarLander(), agent=agent, episodes=20, steps=steps, domain_key=domain_key)
                
                # store success rate
                all_episode_success.append(eval_success)
                all_total_rewards.extend(eval_reward)
                all_episode_steps.append(online_step)
                score_avg = np.mean(all_total_rewards[-200:])
           
            combined_steps += 1

            if done or terminated:                    
                    break

        if eval_success >= 0.01:
            # save agent if above 60% success rate
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'target_model_state_dict': agent.target_net.state_dict(),  # critical
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,  # important
            }, f"{str(domain_key).upper()}Policy{str(int(success*100))}")

        # bar update
        pbar.set_postfix(
            {"Score": f"{score_avg:7.2f}",
                "Eval": f"{eval_success:.3f}",
            }, refresh=True
        )          
        pbar.update(1)

    # close environment
    env.close()

    if 0 not in flags:
        offline_model_report = ml.get_report(np.array(classes_truth), np.array(classes_pred), (granularity[0] != "c"))
        print("OFFLINE:\n", offline_model_report)

    if save_results:
        results = utils_rl.Results.save_results(experiment_list = flags, 
                                   episodes = episodes_num, 
                                   avg_rewards = all_average_rewards, 
                                   total_rewards = all_total_rewards, 
                                   success_rate = all_episode_success,
                                   steps = all_episode_steps,
                                   save_to_csv = save_to_csv)

    print(f"Episode {episode}, Reward: {total_reward:.2f}, Success: {success:.2f}")

    print("Summation of participant episodes seen: ", total_participant_episodes)
    print("Elapsed time in hours: ", (time.time() - start_time) / 3600)

    return results


def vectorize_action(x, dtype=np.float32):
    return np.asarray(x, dtype=dtype).ravel()


def train_robot(env: gymnasium.Env,
                task_df: pd.DataFrame,
                agent,
                clf,
                processor: DatasetProcessor,
                ml: ModelTrainer,
                experiment_list: list,
                fnirs_channel_names: List[str],
                episodes_num: int,
                steps: int,
                window_duration_s: float,
                granularity: str,
                fnirs_rate_hz: float = 5.2,
                shift: float = 0.0,
                noise: float = 0.0,
                smoothing_window_size: int = 0,
                target_update: int = 20,
                buffer_type: str = "ER",
                beta: float = 1.0,
                seed = 42,
                save_results: bool = False,
                save_to_csv: bool = False,
                verbose: bool = False):
    """
    Offline neuro + online Fetch (DDPG + HER) with the same experiment_list flags as ``train``
    """
    start_time = time.time()
    
    # get experiment flags
    flags = list(experiment_list)
    sample_period_s = 1.0 / fnirs_rate_hz

    # initialize buffer
    buffer = fNIRSBuffer(window_duration_s=window_duration_s, sample_period_s=sample_period_s)

    # get robot dataframe
    rw_df = task_df[task_df["desired_goal"].notna()].copy()

    # get total participant episodes
    total_participant_episodes = int(task_df["episode"].nunique())

    # check if robot dataframe is empty
    if rw_df.empty:
        rw_df = task_df[task_df["participantKey"].astype(str).str.contains("RW", na=False)].copy()
    if rw_df.empty:
        raise ValueError("No robot rows in task_df (need desired_goal or RW in participantKey).")

    grouped = rw_df.groupby(["participantKey", "episode"])
    total_participant_episodes = task_df.drop_duplicates(subset=["participantKey", "episode"]).shape[0]
    if verbose: print("Total Participant Episodes: ", total_participant_episodes)

    # get granularity index
    if granularity[0] == "b": gr = 0
    elif granularity[0] == "t": gr = 1
    else: gr = 2

    # initialize rewards, success, and classes
    all_average_rewards, all_total_rewards, all_episode_steps, all_episode_success, classes_truth, classes_pred = ([],[],[],[],[],[])
    success, last_success, last_participant_episode, combined_episodes = 0.0, 0.0, 0, 0
    learning_rate = float(agent.actor_lr)

    # training progress bar
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt} | {postfix}]'
    pbar = trange(episodes_num, unit="ep", bar_format=bar_format, ascii=True)

    # offline training loop
    for (participant, episode), episode_df in grouped:
        minibatch = []
        last_participant_episode = int(episode)
        rows = episode_df.reset_index(drop=True)

        episode_dict = {
            "state": [],
            "action": [],
            "reward": [],
            "next_state": [],
            "achieved_goal": [],
            "next_achieved_goal": [],
            "desired_goal": [],
            "q_augmentation": [],
            "done": [],
        }

        # initialize transition priority for PER buffer
        transition_priority = [] if buffer_type == "PER" else None

        # initialize total reward, neural signal, and new neural signal
        total_reward, neural_signal, new_neural_signal = 0.0, 0.0, 0.0

        # initialize class truth
        class_truth = None

        n = len(rows)

        # for loop for episode data
        for offline_step in range(n):
            action = vectorize_action(rows["actions"].iloc[offline_step])
            action_dist = rows["optimal_actions"].iloc[offline_step]
            reward = float(rows["rewards"].iloc[offline_step])
            state = vectorize_action(rows["states"].iloc[offline_step])
            achieved_goal = vectorize_action(rows["achieved_goal"].iloc[offline_step])
            desired_goal = vectorize_action(rows["desired_goal"].iloc[offline_step])
            rl_timestamp = rows["time"].iloc[offline_step]

            # if action is Nan, skip
            if action.size == 0 or not np.isfinite(action).all():
                continue

            # get next state unless episode ends
            if offline_step + 1 < n:
                next_state = vectorize_action(rows["states"].iloc[offline_step + 1])
                next_achieved_goal = vectorize_action(rows["achieved_goal"].iloc[offline_step + 1])
                done = 0.0
                nopt = rows["optimal_actions"].iloc[offline_step + 1]
            else:
                next_state = state
                next_achieved_goal = vectorize_action(rows["achieved_goal"].iloc[offline_step])
                done = 1.0
                nopt = action_dist

            # get neural features, signal, and sample
            neural_features = buffer.get_features()
            neural_signal = utils_rl.get_neural_signal(clf, neural_features)
            fnirs_sample = processor.get_fnirs_sample(timestamp=rl_timestamp, temporal_shift=-shift, fnirs_channels=fnirs_channel_names)
            buffer.add_sample(timestamp=rl_timestamp, x=fnirs_sample, classification=neural_signal)
            new_neural_signal = buffer.get_neural_credit(granularity=granularity, X=smoothing_window_size)

            # add noise to neural classification if noise ablation is enabled
            if noise > 0.0: new_neural_signal = ml.noisy_output(clf, new_neural_signal, granularity, flip_rate=noise)

            adjusted_neural_signal = utils_rl.adjust_neural_classification(new_neural_signal, beta=beta)

            # get class truth
            class_truth = processor.get_label_sample(timestamp=rl_timestamp, temporal_shift=-shift)

            priority = ddpg_priority(reward, action, action_dist, nopt)
            
            if buffer_type == "ER": priority = 0.0

            # Reward Augmentation Experiment
            if 1 in flags:
                if verbose: 
                    print(f"Reward Augmentation — ep {episode} participant {participant}")
                    print("Original Reward: ", reward, "| Neural Signal: ", new_neural_signal, "| Adjusted Reward: ", reward + adjusted_neural_signal)
                reward = float(reward + adjusted_neural_signal)

            # Priorirization experiment
            if 2 in flags:
                if verbose:
                    print(f"Prioritization — ep {episode} participant {participant}")
                    print("Original Priority: ", abs(priority), "| Neural Signal: ", new_neural_signal, "| Adjusted Priority: ", abs(priority) + adjusted_neural_signal)
                priority = abs(priority)
                priority = float(priority + adjusted_neural_signal)
            else:
                priority = None

            # Q Augmentation Experiment
            if 3 in flags:
                if verbose:
                    print(f"Q-aug analogue — ep {episode} participant {participant}")
                    print("Neural Signal: ", new_neural_signal, "| Q-Value: ", reward + adjusted_neural_signal)
                q_aug = float(adjusted_neural_signal)
            else:
                q_aug = 0.0

            episode_dict["state"].append(state)
            episode_dict["action"].append(action.astype(np.float32))
            episode_dict["reward"].append(reward)
            episode_dict["next_state"].append(next_state.astype(np.float32))
            episode_dict["achieved_goal"].append(achieved_goal.astype(np.float32))
            episode_dict["next_achieved_goal"].append(next_achieved_goal.astype(np.float32))
            episode_dict["desired_goal"].append(desired_goal.astype(np.float32))
            episode_dict["done"].append(float(done))
            episode_dict["q_augmentation"].append(float(q_aug))

            if transition_priority is not None:
                transition_priority.append(priority)
        
        minibatch.append(dc(episode_dict))

        if len(minibatch) == 20:
            agent.store(minibatch)
            for _ in range(40):
                actor_loss, critic_loss = agent.train()
            
            agent.update_networks()
            minibatch = []


        if len(episode_dict["state"]) == 0:
            continue

        if transition_priority is not None:
            episode_dict["transition_priority"] = transition_priority

        if smoothing_window_size > 1 or noise > 0.0:
            classes_pred.append(new_neural_signal)
        else:
            classes_pred.append(neural_signal)

        if class_truth.to_list()[gr] is None or class_truth.to_list()[gr] != class_truth.to_list()[gr]:
            classes_truth.append(0.0)
        else:
            classes_truth.append(class_truth.to_list()[gr])

        if combined_episodes%800 == 0:
            agent.save_weights()

        step_count = len(episode_dict["state"]) - 1
        step_count = max(step_count, 1)
        new_episode_num = max(0, episodes_num // max(total_participant_episodes, 1))
            
        for _ in range(new_episode_num):
            obs, _ = env.reset(seed=seed)
            seed += 1
            total_reward = 0.0
            online_ep = {
                "state": [],
                "action": [],
                "reward": [],
                "next_state": [],
                "achieved_goal": [],
                "next_achieved_goal": [],
                "desired_goal": [],
                "q_augmentation": [],
                "done": [],
            }
            prios = [] if buffer_type == "PER" else None

            for _step in range(steps):
                state = obs["observation"].astype(np.float32).ravel()
                desired_goal = obs["desired_goal"].astype(np.float32).ravel()
                achieved_goal = obs["achieved_goal"].astype(np.float32).ravel()

                action = agent.choose_action(state, desired_goal, train_mode=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                done = float(terminated or truncated)
                next_state = next_obs["observation"].astype(np.float32).ravel()
                next_achieved_goal = next_obs["achieved_goal"].astype(np.float32).ravel()

                td_error = 0.0
                if buffer_type == "PER":
                    td_error = ddpg_priority(reward, action, desired_goal, next_achieved_goal)

                online_ep["state"].append(state)
                online_ep["action"].append(action.astype(np.float32))
                online_ep["reward"].append(float(reward))
                online_ep["next_state"].append(next_state)
                online_ep["achieved_goal"].append(achieved_goal)
                online_ep["next_achieved_goal"].append(next_achieved_goal)
                online_ep["desired_goal"].append(desired_goal)
                online_ep["done"].append(done)
                online_ep["q_augmentation"].append(float(0.0 + 1e-4))

                if prios is not None:
                    prios.append(td_error)

                obs = next_obs
                total_reward += float(reward)

                if terminated or truncated:
                    break

            minibatch.append(dc(online_ep))

            if len(minibatch) == 20:
                agent.store(minibatch)

                for _ in range(40):
                    actor_loss, critic_loss = agent.train()

                agent.update_networks()
                minibatch = []

            if len(online_ep["state"]) > 0:
                if prios is not None:
                    online_ep["transition_priority"] = prios
        
            st = max(len(online_ep["state"]) - 1, 1)
            all_average_rewards.append(round(total_reward / st, 2))
            all_total_rewards.append(round(total_reward, 2))
            all_episode_steps.append(len(online_ep["state"]) - 1)
            score_avg = np.mean(all_total_rewards[-50:])

            if combined_episodes%800 == 0:
                agent.save_weights()
                success = utils_rl.evaluate_fetch(env, agent, steps=steps, episodes=50)
                all_episode_success.append(success)
                last_success = success

                if success >= 0.99:
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
                        "FetchPolicy" + str(int(success * 100)) + ".pth",
                    )
            else:
                all_episode_success.append(success)

            pbar.set_postfix(
                {"Score": f"{total_reward:7.2f}",
                    "Avg50": f"{score_avg:7.2f}",
                    "Eval": f"{last_success:.3f}",
                    "Success": f"{success:.3f}"
                }, refresh=True
            )
            pbar.update(1)
            combined_episodes += 1
        combined_episodes += 1   
 
    env.close()

    offline_model_report = ml.get_report(np.array(classes_truth), np.array(classes_pred), (granularity[0] != "c"))
    print(offline_model_report)

    results = None
    if save_results:
        results = utils_rl.Results.save_results(
            experiment_list=experiment_list,
            episodes=episodes_num,
            avg_rewards=all_average_rewards,
            total_rewards=all_total_rewards,
            success_rate=all_episode_success,
            steps=all_episode_steps,
            save_to_csv=save_to_csv,
        )

    print(f"Robot episode {last_participant_episode}, Reward: {total_reward:.2f}, Success: {success:.2f}")
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
