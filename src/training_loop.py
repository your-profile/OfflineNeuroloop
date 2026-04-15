import numpy as np
import pandas as pd
from typing import List
from src.models.model_training import ModelTrainer
from src.networks.DQN import DQN
from src.envs.lunar_lander import LunarLander
from src.envs.flappy_bird import FlappyBirdEnv as FlappyBird
import gymnasium
from tqdm import trange
import numpy as np
import src.utils as utils
import torch
from src.neural.buffer import fNIRSBuffer
from src.rl_loop import utils_rl
from src.neural.preprocessing import DatasetProcessor


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
    # trial.py passes experiment_list; legacy kw is experiment_conditions
    flags = experiment_conditions if experiment_conditions else experiment_list

    decay = (0.01 / 1.0) ** (1 / episodes_num)
    learning_rate = agent.lr

    # calculate window size + initialize buffer
    sample_period_s = 1.0 / fnirs_rate_hz
    buffer = fNIRSBuffer(window_duration_s=window_duration_s, sample_period_s=sample_period_s)
    grouped = task_df.groupby(['participantKey', 'episode'])
    
    # # Calculate total number of participant episodes by counting unique (participantKey, episode) pairs
    total_participant_episodes = task_df.drop_duplicates(subset=["participantKey", "episode"]).shape[0]
    # print(f"Total participant episodes: {total_participant_episodes}")

    if granularity[0] == "b": gr = 0
    if granularity[0] == "t": gr = 1
    if granularity[0] == "c": gr = 2


    # rewards, timesteps, success rate
    all_average_rewards, all_total_rewards, all_episode_steps, all_episode_success = [],[],[],[]
    classes_truth, classes_pred = [],[]
    success, last_success, last_participant_episode, combined_episodes = (0.0, 0.0, 0, 0)
    epsilon = 1.0
    seed = np.random.randint(0, 5000)
    domain_key = task_df["condition"].iloc[0][0]

    # training progress bar
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt}]'
    pbar = trange(episodes_num, unit="ep", bar_format=bar_format, ascii=True)

    # for loop for participant task data
    for (participant, episode), episode_df in grouped:  
        total_reward, last_state_action_value, state_action_value = 0, 0, 0
        combined_episodes += 1
        done = False

        for step, (idx, row) in enumerate(episode_df.iterrows()):
            action = row["actions"] # action the participant observed
            action_dist = row["optimal_actions"] # action distribution
            reward = row["rewards"] # reward
            state = row["states"] # state the participant observed
            final_step = row['steps'] # length of episode the participant observed

            # if action is Nan, skip
            if isinstance(action, (float, int)):
                if action != action:
                    continue
                action = int(action)
            if isinstance(action, (list)):
                print(action)

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
            new_neural_signal = buffer.get_neural_credit(granularity = granularity, X = smoothing_window_size)

            if noise > 0.0:
                new_neural_signal  = ml.noisy_output(clf,  new_neural_signal, granularity, flip_rate = noise)

            adjusted_neural_signal = utils_rl.adjust_neural_classification(neural_signal, beta=beta)
            class_truth = processor.get_label_sample(timestamp = rl_timestamp, temporal_shift = -shift)
            # print(class_truth.tolist())
            
            next_action_dist = (
                episode_df["optimal_actions"][idx + 1] if idx < final_step - 1 else action_dist
            )
            if "desired_goal" in episode_df.columns:
                priority = ddpg_priority(reward, action, action_dist, next_action_dist)
            else:
                priority = dqn_priority(reward, action, action_dist, next_action_dist)

            if buffer_type == "ER":
                priority = 0.0

            # Reward Augmentation Experiment
            if 1 in flags:
                if verbose:
                    print(f"Experiment Condition 0: Reward Augmentation -- Episode {episode} -- Participant: {participant}")
                reward = reward + adjusted_neural_signal
            
            # Priorirization experiment
            if 2 in flags:
                if verbose:
                    print(f"Experiment Condition 1: Prioritization -- Episode {episode} -- Participant: {participant}")
                
                priority += adjusted_neural_signal
            else:
                priority = 0.0
       
            # Epsilon/Exploration Adjustment
            if 3 in flags:
                if verbose:
                    print(f"Experiment Condition 4: Exploration Modulation -- Epsilon {epsilon}")
                epsilon = utils_rl.adjust_epsilon(epsilon, adjusted_neural_signal)

            # Learning Rate Adjustment
            if 4 in flags:
                if verbose:
                    print(f"Experiment Condition 5: Learning Rate Modulation -- Learning Rate {learning_rate}")
                learning_rate = utils_rl.adjust_epsilon(learning_rate, adjusted_neural_signal)
                agent.set_lr(learning_rate)

            # Q Augmentation Experiment
            if 5 in flags:
                if verbose:
                    print(f"Experiment Condition 5: Q-Augmentation -- Episode {episode} -- Participant: {participant}")
                agent.remember(state, action, reward, next_state, done, q_augmentation = adjusted_neural_signal)

            if buffer_type == "ER":
                agent.remember(state, action, reward, next_state, done)
            if buffer_type == "PER":
                agent.remember(state, action, reward, next_state, done, priority = priority)

            # print(f"Step: {step} | Action: {action} | Reward: {reward} | State: {state} | idx: {idx}")

            total_reward += reward
            last_state_action_value = state_action_value
        
        # model optimality predictions
        if smoothing_window_size > 1:
            classes_pred.append(new_neural_signal) #predictions with smoothing
        else:
            classes_pred.append(neural_signal) #raw predictions
        
        # print(class_truth.to_list()[gr], neural_signal)
        classes_truth.append(class_truth.to_list()[gr])


        # save average reward, total reward and timesteps
        # all_average_rewards.append(round(total_reward/step, 2))
        # all_total_rewards.append(round(total_reward, 2))
        # all_episode_steps.append(step)

        # episodes needed to complete training
        new_episode_num = max(0, episodes_num // max(total_participant_episodes, 1))
        #print(f"New Episode Num: {new_episode_num}, Total Participant Episodes: {total_participant_episodes}")

        # observe new states outside of data
        for new_epsiode in range(0, new_episode_num):
            # set seed
            if domain_key == "F":
                state, _ = env.reset(seed=seed)
            else:
                state = env.reset(seed=seed)

            seed += 1
            total_reward, state_action_value, last_state_action_value = 0, 0, 0

            for step in range(steps):
                # choose action
                action, state_action_value = agent.chooseAction(state, epsilon)
                
                # take action in env
                if domain_key == "F":
                    next_state, reward, done, _, _ = env.step(action)
                else:
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
                if domain_key == "F":
                    success = utils_rl.evaluate(env=FlappyBird(score_limit=100), agent=agent, episodes=30, steps=600)
                else:
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

    offline_model_report = ml.get_report(np.array(classes_truth), np.array(classes_pred), (granularity[0] != "c"))
    print(offline_model_report)

    if save_results:
        results = utils_rl.Results.save_results(experiment_list = experiment_list, 
                                   episodes = episodes_num, 
                                   avg_rewards = all_average_rewards, 
                                   total_rewards = all_total_rewards, 
                                   success_rate = all_episode_success,
                                   steps = all_episode_steps,
                                   save_to_csv = save_to_csv)

    print(f"Episode {episode}, Reward: {total_reward:.2f}, Success: {success:.2f}")

    print("Summation of participant episodes seen: ", total_participant_episodes)

    return results


def vectorize_action(x, dtype=np.float32):
    return np.asarray(x, dtype=dtype).ravel()


def train_robot(
                    env: gymnasium.Env,
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
                    save_results: bool = False,
                    save_to_csv: bool = False,
                    verbose: bool = False):
    """
    Offline neuro + online Fetch (DDPG + HER) with the same experiment_list flags as ``train``:
    0 baseline, 1 reward aug, 2 prioritization (PER seed + TD updates), 3 exploration (randomprob),
    4 LR modulation, 5 extra reward shaping (Q-aug analogue).
    """
    from src.networks.DDPG import DDPG

    if not isinstance(agent, DDPG):
        raise TypeError("train_robot requires a DDPG agent from load_ddpg_agent / load_agent(DDPG).")

    flags = list(experiment_list)
    batch_size = agent.batch_size
    warmup = 200

    sample_period_s = 1.0 / fnirs_rate_hz
    buffer = fNIRSBuffer(window_duration_s=window_duration_s, sample_period_s=sample_period_s)

    rw_df = task_df[task_df["desired_goal"].notna()].copy()

    if rw_df.empty:
        rw_df = task_df[task_df["participantKey"].astype(str).str.contains("RW", na=False)].copy()
    if rw_df.empty:
        raise ValueError("No robot rows in task_df (need desired_goal or RW in participantKey).")

    grouped = rw_df.groupby(["participantKey", "episode"])

    if granularity[0] == "b":
        gr = 0
    elif granularity[0] == "t":
        gr = 1
    else:
        gr = 2

    all_average_rewards, all_total_rewards, all_episode_steps, all_episode_success = ([],[],[],[])
    classes_truth, classes_pred = [], []
    success, last_success = 0.0, 0.0
    last_participant_episode = 0
    combined_episodes = 0

    learning_rate = float(agent.actor_lr)
    seed = np.random.randint(0, 5000)

    bar_format = ("{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt}]")
    pbar = trange(episodes_num, unit="ep", bar_format=bar_format, ascii=True)

    def maybe_train(n_updates: int = 8):
        if agent.ram.len_transition < warmup:
            return
        for _ in range(n_updates):
            agent.train()
            agent.update_target_networks()

    total_participant_episodes = int(task_df["episode"].nunique())

    for (participant, episode), episode_df in grouped:
        last_participant_episode = int(episode)
        combined_episodes += 1

        rows = episode_df.reset_index(drop=True)
        ep = {
            "state": [],
            "action": [],
            "reward": [],
            "next_state": [],
            "achieved_goal": [],
            "next_achieved_goal": [],
            "desired_goal": [],
            "done": [],
        }

        transition_priority = [] if buffer_type == "PER" else None

        total_reward = 0.0
        neural_signal = 0.0
        new_neural_signal = 0.0
        class_truth = None

        for t in range(len(rows)):
            row = rows.iloc[t]
            action = row["actions"]
            action_dist = row["optimal_actions"]
            reward = float(row["rewards"])
            state = row["states"]

            a = vectorize_action(action)
            if a.size == 0 or not np.isfinite(a).all():
                continue

            if t + 1 < len(rows):
                nrow = rows.iloc[t + 1]
                next_state = nrow["states"]
                next_ag = nrow["achieved_goal"]
                done = 0.0
            else:
                nrow = row
                next_state = state
                next_ag = row["achieved_goal"]
                done = 1.0

            s = vectorize_action(state)
            sn = vectorize_action(next_state)
            ag = vectorize_action(row["achieved_goal"])
            nag = vectorize_action(next_ag)
            dg = vectorize_action(row["desired_goal"])

            rl_timestamp = row["time"]

            neural_features = buffer.get_features()
            neural_signal = utils_rl.get_neural_signal(clf, neural_features)
            fnirs_sample = processor.get_fnirs_sample(timestamp=rl_timestamp, temporal_shift=-shift, fnirs_channels=fnirs_channel_names)
            buffer.add_sample(timestamp=rl_timestamp, x=fnirs_sample, classification=neural_signal)
            new_neural_signal = buffer.get_neural_credit(granularity=granularity, X=smoothing_window_size)

            if noise > 0.0:
                new_neural_signal = ml.noisy_output(clf, new_neural_signal, granularity, flip_rate=noise)

            adjusted_neural = utils_rl.adjust_neural_classification(neural_signal, beta=beta)
            class_truth = processor.get_label_sample(timestamp=rl_timestamp, temporal_shift=-shift)

            nopt = nrow["optimal_actions"] if t + 1 < len(rows) else action_dist
            priority = ddpg_priority(reward, a, action_dist, nopt)

            if buffer_type == "ER":
                priority = 0.0

            if 1 in flags:
                if verbose:
                    print(f"Reward Augmentation — ep {episode} participant {participant}")
                reward = reward + adjusted_neural

            if 2 in flags:
                if verbose:
                    print(f"Prioritization — ep {episode} participant {participant}")
                priority = priority + adjusted_neural
            else:
                priority = 0.0

            if 3 in flags:
                if verbose:
                    print(f"Exploration modulation — randomprob {agent.randomprob:.3f}")

                agent.randomprob = float(np.clip(agent.randomprob - 0.02 * float(adjusted_neural), 0.05, 0.5))

            if 4 in flags:
                if verbose:
                    print(f"LR modulation — lr {learning_rate:.6f}")
                learning_rate = float(np.clip(learning_rate - 1e-4 * float(adjusted_neural), 1e-5, 1e-2))
                agent.set_lr(learning_rate)

            r_store = float(reward)
            if 5 in flags:
                if verbose:
                    print(f"Q-aug analogue — ep {episode} participant {participant}")
                r_store = r_store + float(adjusted_neural)

            ep["state"].append(s)
            ep["action"].append(a.astype(np.float32))
            ep["reward"].append(r_store)
            ep["next_state"].append(sn.astype(np.float32))
            ep["achieved_goal"].append(ag.astype(np.float32))
            ep["next_achieved_goal"].append(nag.astype(np.float32))
            ep["desired_goal"].append(dg.astype(np.float32))
            ep["done"].append(float(done))

            if transition_priority is not None:
                transition_priority.append(priority)

            total_reward += r_store

        if len(ep["state"]) == 0:
            continue

        if transition_priority is not None:
            ep["transition_priority"] = transition_priority

        agent.ram.add_episode(ep)
        maybe_train(16)

        if smoothing_window_size > 1:
            classes_pred.append(new_neural_signal)
        else:
            classes_pred.append(neural_signal)

        classes_truth.append(class_truth.to_list()[gr])

        step_count = len(ep["state"]) - 1
        step_count = max(step_count, 1)
        all_average_rewards.append(round(total_reward / step_count, 2))
        all_total_rewards.append(round(total_reward, 2))
        all_episode_steps.append(len(ep["state"]) - 1)

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
                "done": [],
            }
            prios = [] if buffer_type == "PER" else None

            for _step in range(steps):
                s = obs["observation"].astype(np.float32).ravel()
                g = obs["desired_goal"].astype(np.float32).ravel()
                ag = obs["achieved_goal"].astype(np.float32).ravel()

                act = agent.choose_action(s, g, train_mode=True)
                next_obs, rew, terminated, truncated, info = env.step(act)
                d = float(terminated or truncated)
                sn = next_obs["observation"].astype(np.float32).ravel()
                nag = next_obs["achieved_goal"].astype(np.float32).ravel()

                td_proxy = 0.0
                if buffer_type == "PER":
                    td_proxy = abs(float(rew)) + 1e-6

                online_ep["state"].append(s)
                online_ep["action"].append(act.astype(np.float32))
                online_ep["reward"].append(float(rew))
                online_ep["next_state"].append(sn)
                online_ep["achieved_goal"].append(ag)
                online_ep["next_achieved_goal"].append(nag)
                online_ep["desired_goal"].append(g)
                online_ep["done"].append(d)
                if prios is not None:
                    prios.append(td_proxy)

                obs = next_obs
                total_reward += float(rew)
                if terminated or truncated:
                    break

            if len(online_ep["state"]) > 0:
                if prios is not None:
                    online_ep["transition_priority"] = prios
                agent.ram.add_episode(online_ep)
            maybe_train(16)

            combined_episodes += 1
            st = max(len(online_ep["state"]) - 1, 1)
            all_average_rewards.append(round(total_reward / st, 2))
            all_total_rewards.append(round(total_reward, 2))
            all_episode_steps.append(len(online_ep["state"]) - 1)

            score_avg = np.mean(all_total_rewards[-50:])

            if combined_episodes % target_update == 0:
                success = utils_rl.evaluate_fetch(env, agent, steps=steps, episodes=15)
                all_episode_success.append(success)
                last_success = success
            else:
                all_episode_success.append(success)

            if success > 0.60:
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

            pbar.set_postfix_str(
                f"Score: {total_reward:7.2f} Neural: {neural_signal} "
                f"Avg50: {score_avg:7.2f} Eval: {last_success}"
            )
    pbar.update(1)
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

    return results