import numpy as np
import pandas as pd
from typing import List
# import utils
from src.networks.DQN import DQN
from src.envs.lunar_lander import LunarLander
# import utils
from src.envs.lunar_lander import LunarLander
import gymnasium
from tqdm import trange
import numpy as np
import torch
from src.neural.buffer import fNIRSBuffer
from src.rl_loop import utils_rl as utils
from src.neural.preprocessing import DatasetProcessor

def train(env:gymnasium.Env, 
          task_df:pd.DataFrame, 
          agent: DQN, 
          clf, 
          processor: DatasetProcessor,
          experiment_name: str, 
          fnirs_channel_names: List[str], 
          domain: str = "Lunar Lander", 
          episodes_num: int = 500, 
          steps: int = 300, 
          window_duration_s: float = 60.0, 
          fnirs_rate_hz: float = 5.2, 
          shift: float = 0.0, 
          target_update: int = 20, 
          buffer_type: str = 'ER', 
          beta: float = 1.0,
          experiment_conditions: List[str] = [], 
          save_results: bool = False, 
          save_to_csv: bool = False,
          verbose: bool = False
    ):

    decay = (0.01 / 1.0) ** (1 / episodes_num)

    # calculate window size + initialize buffer
    sample_period_s = 1.0 / fnirs_rate_hz
    buffer = fNIRSBuffer(window_duration_s=window_duration_s, sample_period_s=sample_period_s)
    grouped = task_df.groupby(['participantKey', 'episode'])

    # rewards, timesteps, success rate
    all_average_rewards, all_total_rewards, all_episode_steps, all_episode_success = [],[],[],[]
    classes_opt, classes_disc, classes_pred, smoothing_classes_pred = [],[],[],[]
    success, last_success, last_participant_episode, combined_episodes = (0.0, 0.0, 0, 0)
    epsilon = 1.0
    seed = np.random.randint(0, 5000)

    # training progress bar
    bar_format = '{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]'
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
            neural_signal = utils.get_neural_signal(features = neural_features, clf = clf)

            # update neural buffer
            fnirs_sample = processor.get_fnirs_sample(timestamp = rl_timestamp, temporal_shift = -shift, fnirs_channels = fnirs_channel_names)
            buffer.add_sample(timestamp = rl_timestamp, x = fnirs_sample, classification=neural_signal)
            
            # get + adjust neural classification
            new_neural_signal = buffer.get_neural_credit(X = 3)
            adjusted_neural_signal = utils.adjust_neural_classification(neural_signal, beta=beta)
            
            # if action is Nan, skip
            if action != action:
                continue

            # Reward Augmentation Experiment
            if 0 in experiment_conditions:
                if verbose:
                    print(f"Experiment Condition 0: Reward Augmentation -- Episode {episode} -- Participant: {participant}")
                reward = reward + adjusted_neural_signal
            
            # Priorirization experiment
            if 1 in experiment_conditions:
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
            if 2 in experiment_conditions:
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
            if 3 in experiment_conditions:
                if verbose:
                    print(f"Experiment Condition 3: Exploration Modulation -- Episode {episode} -- Participant: {participant}")
                epsilon = utils.adjust_epsilon(epsilon, adjusted_neural_signal)

            # print(f"Step: {step} | Action: {action} | Reward: {reward} | State: {state} | idx: {idx}")

            total_reward += reward
            last_state_action_value = state_action_value
        
        # model optimality predictions
        classes_pred.append(int(neural_signal)) #raw predictions
        smoothing_classes_pred.append(new_neural_signal) #predictions with smoothing

        # optimality ground truths (lunar lander)
        if total_reward > 90.0:
            classes_opt.append(0)
            classes_disc.append(0)
        elif total_reward > 17.0:
            classes_opt.append(1)
            classes_disc.append(1)
        else:
            classes_opt.append(1)
            classes_disc.append(2)

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

            # bar update
            pbar.set_postfix_str(f"Score: {total_reward: 7.2f}, Neural Signal: {neural_signal}, 50 Score Avg: {score_avg: 7.2f}, Eval: {last_success}")
            pbar.update(0)

            if combined_episodes % target_update == 0:
                success = utils.evaluate(env=LunarLander(), agent=agent, episodes=30, steps=600)
                all_episode_success.append(success)
                last_success = success
            else:
                all_episode_success.append(success)

            if success > 0.40:
                # save agent if above 40% success rate
                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.policy_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict()}, 
                    "LLPolicy" + str(int(success*100
                )))

            combined_episodes += 1

    print(combined_episodes, len(all_episode_steps))
    env.close()
    if save_results:
        parameters = utils.Results.save_parameters(domain = domain,
                                #   task = task,
                                #   participant_list = participant_list,
                                #   trial_key = trial_key,
                                    experiment_key = experiment_name, 
                                    algorithm_name = agent.algorithm, 
                                    episodes = combined_episodes, 
                                    state_dim = int(agent.n_observations), 
                                    action_dim = int(agent.n_actions), 
                                    learning_rate = float(agent.lr), 
                                    gamma = agent.gamma, 
                                    target_update = target_update, 
                                    buffer_type = buffer_type, 
                                    epsilon_type = "decay", 
                                    credit_type = "none",
                                    save_to_csv = save_to_csv
                                #   temporal_shift = temporal_shift,
                                #   resample_rate = resample_rate,
                                #   window_size = window_size,
                                #   step_size = step_size,
                                #   random_state = random_state,
                                #   model_name = model_name,
                                #   model_granularity = model_granularity,
                                #   model_architecture = model_architecture,
                                #   model_solver = model_solver,
                                #   model_activation = model_activation,
                                #   model_report = model_report,
                                    )

        results = utils.Results.save_results(domain = domain,
                                   experiment_name = experiment_name, 
                                   episodes = last_participant_episode, 
                                   avg_rewards = all_average_rewards, 
                                   total_rewards = all_total_rewards, 
                                   success_rate = all_episode_success,
                                   steps = all_episode_steps,
                                   save_to_csv = save_to_csv)

    print(f"Episode {episode}, Reward: {total_reward:.2f}, Success: {success:.2f}")

    print("Summation of participant episodes seen: ", total_participant_episodes)

    return results, parameters