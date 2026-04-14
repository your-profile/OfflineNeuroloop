import numpy as np
import pandas as pd
from typing import List
from src.models.model_training import ModelTrainer
from src.networks.DQN import DQN
from src.envs.lunar_lander import LunarLander
import gymnasium
from tqdm import trange
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
          experiment_list: list, 
          fnirs_channel_names: List[str], 
          episodes_num: int, 
          steps: int, 
          window_duration_s: float, 
          granularity: str,
          fnirs_rate_hz: float = 5.2, 
          shift: float = 0.0, 
          noise = 0.0,
          smoothing_window_size: int = 0,
          target_update: int = 20, 
          buffer_type: str = 'ER', 
          beta: float = 1.0,
          experiment_conditions: List[str] = [], 
          save_results: bool = False, 
          save_to_csv: bool = False,
          verbose: bool = False
    ):

    decay = (0.01 / 1.0) ** (1 / episodes_num)
    learning_rate = agent.lr

    # calculate window size + initialize buffer
    sample_period_s = 1.0 / fnirs_rate_hz
    buffer = fNIRSBuffer(window_duration_s=window_duration_s, sample_period_s=sample_period_s)
    grouped = task_df.groupby(['participantKey', 'episode'])

    if granularity[0] == "b": gr = 0
    if granularity[0] == "t": gr = 1
    if granularity[0] == "c": gr = 2

    # rewards, timesteps, success rate
    all_average_rewards, all_total_rewards, all_episode_steps, all_episode_success = [],[],[],[]
    classes_truth, classes_pred = [],[]
    success, last_success, last_participant_episode, combined_episodes = (0.0, 0.0, 0, 0)
    epsilon = 1.0
    seed = np.random.randint(0, 5000)

    # training progress bar
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt}]'
    pbar = trange(episodes_num, unit="ep", bar_format=bar_format, ascii=True)

    # for loop for participant task data
    for (participant, episode), episode_df in grouped:
        total_participant_episodes = task_df['episode'].nunique()
   
        total_reward, last_state_action_value, state_action_value = 0, 0, 0
        combined_episodes += 1
        done = False

        for step, (idx, row) in enumerate(episode_df.iterrows()):
            action = row["actions"] # action the participant observed
            action_dist = row["optimal_actions"] # action distribution
            reward = row["rewards"] # reward
            state = row["states"] # state the participant observed
            final_step = row['steps'] # length of episode the participant observed

            # get next state unless episode ends
            try:
                next_state = episode_df["states"][idx+1]
            except:
                done = True
                next_state = state

            # get rl task statistic tuple (state, action, reward) timestamp
            rl_timestamp = row["time"]

            # get associated fNIRS sample given timestep
            neural_features = buffer.get_features()
            neural_signal = utils_rl.get_neural_signal(features = neural_features, clf = clf)

            # update neural buffer
            fnirs_sample = processor.get_fnirs_sample(timestamp = rl_timestamp, temporal_shift = -shift, fnirs_channels = fnirs_channel_names)
            buffer.add_sample(timestamp = rl_timestamp, x = fnirs_sample, classification=neural_signal)
            
            # get + adjust neural classification
            new_neural_signal = buffer.get_neural_credit(X = smoothing_window_size)

            if noise > 0.0:
                new_neural_signal  = ml.noisy_output(clf,  new_neural_signal, granularity, flip_rate = noise)

            adjusted_neural_signal = utils_rl.adjust_neural_classification(neural_signal, beta=beta)
            class_truth = processor.get_label_sample(timestamp = rl_timestamp, temporal_shift = -shift)
            # print(class_truth.tolist())
            
            # if action is Nan, skip
            if action != action:
                continue

            # Reward Augmentation Experiment
            if 1 in experiment_conditions:
                if verbose:
                    print(f"Experiment Condition 0: Reward Augmentation -- Episode {episode} -- Participant: {participant}")
                reward = reward + adjusted_neural_signal
            
            # Priorirization experiment
            if 2 in experiment_conditions:
                if verbose:
                    print(f"Experiment Condition 1: Prioritization -- Episode {episode} -- Participant: {participant}")
                
                # Calculate td error from last and current action distribution
                try:
                    next_action_dist = episode_df["optimal_actions"][idx + 1] if idx > 0 else action_dist
                    prev_action_value = next_action_dist[action] if action < len(next_action_dist) else 0
                    curr_action_value = action_dist[action] if action < len(action_dist) else 0
                    priority = reward + curr_action_value - prev_action_value
                except:
                    print("Skipping TD Error")
                    priority = 0

                priority += adjusted_neural_signal
       
                agent.remember(state, action, reward, next_state, done, priority = priority)
            
            # Q Augmentation Experiment
            if 3 in experiment_conditions:
                if verbose:
                    print(f"Experiment Condition 2: Q-Augmentation -- Episode {episode} -- Participant: {participant}")
                agent.remember(state, action, reward, next_state, done, q_augmentation = adjusted_neural_signal)
            
            if buffer_type == "ER":
                agent.remember(state, action, reward, next_state, done)
            else:
                state_action_value = action_dist[action]
                td_error = reward + state_action_value - last_state_action_value
                agent.remember(state, action, reward, next_state, done, priority = td_error)

            # Epsilon/Exploration Adjustment
            if 4 in experiment_conditions:
                if verbose:
                    print(f"Experiment Condition 4: Exploration Modulation -- Epsilon {epsilon}")
                epsilon = utils_rl.adjust_epsilon(epsilon, adjusted_neural_signal)

            # Learning Rate Adjustment
            if 5 in experiment_conditions:
                if verbose:
                    print(f"Experiment Condition 5: Learning Rate Modulation -- Learning Rate {learning_rate}")
                learning_rate = utils_rl.adjust_epsilon(learning_rate, adjusted_neural_signal)
                agent.set_lr(learning_rate)

            # print(f"Step: {step} | Action: {action} | Reward: {reward} | State: {state} | idx: {idx}")

            total_reward += reward
            last_state_action_value = state_action_value
        
        # model optimality predictions
        if smoothing_window_size > 1:
            classes_pred.append(new_neural_signal) #predictions with smoothing
        else:
            classes_pred.append(int(neural_signal)) #raw predictions
            
        classes_truth.append(class_truth.to_list()[gr])


        # save average reward, total reward and timesteps
        all_average_rewards.append(round(total_reward/step, 2))
        all_total_rewards.append(round(total_reward, 2))
        all_episode_steps.append(step)

        # episodes needed to complete training
        new_episode_num = episodes_num//total_participant_episodes

        # observe new states outside of data
        for new_epsiode in range(0, new_episode_num):
            # set seed
            state  = env.reset(seed=seed)
            seed += 1
            total_reward, state_action_value, last_state_action_value = 0, 0, 0

            for step in range(steps):
                # choose action
                action, state_action_value = agent.chooseAction(state, epsilon)
                
                # take action in env
                next_state, reward, done, _ = env.step(action)
                
                # td_error for PER
                td_error = reward + state_action_value - last_state_action_value
                
                # save trajectory in buffer
                agent.remember(state, action, reward, next_state, done, priority = td_error)
                
                state = next_state
                last_state_action_value = state_action_value
                total_reward += reward

                if done:                    
                    break

            combined_episodes += 1

            all_average_rewards.append(round(total_reward/step, 2))
            all_total_rewards.append(round(total_reward, 2))
            all_episode_steps.append(step)
            score_avg = np.mean(all_total_rewards[-50:])

            # decay epsilon
            epsilon = max(0.01, epsilon*decay)

            if combined_episodes % target_update == 0:
                success = utils_rl.evaluate(env=LunarLander(), agent=agent, episodes=30, steps=600)
                all_episode_success.append(success)
                last_success = success
            else:
                all_episode_success.append(success)

            if success > 0.60:
                # save agent if above 60% success rate
                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.policy_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict()}, 
                    "LLPolicy" + str(int(success*100
                )))

            # bar update
            pbar.set_postfix_str(f"Score: {total_reward: 7.2f}, Neural Signal: {neural_signal}, 50 Score Avg: {score_avg: 7.2f}, Eval: {last_success}")
            pbar.update(1)
            combined_episodes += 1

    env.close()
    # print("Classes Truth: ", classes_truth)
    # print("Classes Pred: ", classes_pred)

    offline_model_report = ml.get_report([int(x) for x in classes_truth], [int(x) for x in classes_pred], (granularity[0] != "c"))
    print(offline_model_report)

    if save_results:
        results = utils_rl.Results.save_results(experiment_list = experiment_list, 
                                   episodes = last_participant_episode, 
                                   avg_rewards = all_average_rewards, 
                                   total_rewards = all_total_rewards, 
                                   success_rate = all_episode_success,
                                   steps = all_episode_steps,
                                   save_to_csv = save_to_csv)

    print(f"Episode {episode}, Reward: {total_reward:.2f}, Success: {success:.2f}")

    print("Summation of participant episodes seen: ", total_participant_episodes)

    return results