import numpy as np
import csv
import os
import json 
import torch

class Results():
    '''
    Results: Saving hyperparameters and final results for experiments
    '''
    def save_results(episodes, total_rewards, success_rate, steps, experiment_list, save_to_csv = False, filepath = "/Users/juliasantaniello/Desktop/OfflineNeuroloop/results"):
        import datetime

        print(f"Len Total Rewards: {len(total_rewards)}, Len Success Rate: {len(success_rate)}, Len Steps: {len(steps)}, Episodes: {episodes}")

        row = {
            "date": datetime.date.today(),
            "time": datetime.datetime.now(),
            "experiment_list": experiment_list,
            "episodes": episodes,
            "total_reward": json.dumps(list(map(float, total_rewards))),
            "success_rate": json.dumps(list(map(float, success_rate))),
            "steps": json.dumps(list(map(float, steps))),
        }

        if save_to_csv:
            write_header = not os.path.exists(filepath)

            with open(filepath, mode='a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=row.keys())

                if write_header:
                    writer.writeheader()

                writer.writerow(row)

        return row
        

    def save_parameters(domain:str, participant_list, algorithm_name, experiment_list, episodes:int, state_dim:int, action_dim:int, learning_rate:float, gamma:float, epsilon_type:str, eval_update:int, resample_rate, window_size, step_size, buffer_type:str, credit_type:str, temporal_shift, save_to_csv = False, filepath = "/Users/juliasantaniello/Desktop/OfflineNeuroloop/parameters"):
        import datetime
        row = {"date": datetime.date.today(),
            "time": datetime.datetime.now(),
            "domain": domain,
            # "task": task,
            "participant_list": participant_list,
            "experiment_list": experiment_list,
            "algorithm": algorithm_name,

            "episodes": episodes,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "epsilon_type": epsilon_type,
            "eval_update": eval_update,
            "buffer_type": buffer_type,

            "temporal_shift": temporal_shift,
            "resample_rate": resample_rate,
            "window_size": window_size,
            "step_size": step_size,
            # "random_state": random_state,
            
            "credit_type": credit_type,
            # "model_name": model_name,
            # "model_granularity": model_granularity,
            # "model_architecture": model_architecture,
            # "model_solver": model_solver,
            # "model_activation": model_activation,
            # "model_report": model_report,
            }
        
        if save_to_csv:
            fieldnames = row.keys()
            write_header = not os.path.exists(filepath)


            with open(filepath, mode='a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if write_header:
                    writer.writeheader()

                writer.writerow(row)

        return row

def adjust_neural_classification(output: int|None, beta = 1.0, verbose = False) -> float:
    """
    adjust_neural_classification(output: int, 
        beta: float = 1.0):
        Receives classifier/regressor output. Scales output based on beta parameter. Beta default set to 1.0
    """
    original_output = output

    if output is None:
        return 0.0

    # binary and ternary model output
    if output == 0.0:
        return 1.0
    elif output == 1.0:
        return -0.0
    elif output == 2.0:
        return -1.0
    # regressor output (continuous)
    else:
        adjustment = -output

    if verbose:
        print(f"Original Neural Classification: {original_output}, Adjusted w/o Beta {output}, Beta: {beta}, Adjusted Neural Classification: {adjustment*beta}")
    
    return adjustment*beta #scale value with beta parameter


def adjust_reward(
    reward: float,
    neural_signal: int | float,
    clf_probs=None,
    means: tuple[float, float, float] = (1.0, -0.1, -1.0),
    stds: tuple[float, float, float] = (0.25, 0.25, 0.25),
    mix_scale: float = 1.0,
    sample_bonus: bool = False,
    clip_bonus: float | None = None,
    beta: float = 1.0,
):
    if clf_probs is not None and not np.isscalar(clf_probs):
        probs = np.asarray(clf_probs, dtype=np.float64).ravel()
        means_array = np.asarray(means, dtype=np.float64).ravel()
        std_array = np.asarray(stds, dtype=np.float64).ravel()

        k = int(min(len(probs), len(means_array), len(std_array)))
        if k > 0:
            probs = probs[:k]
            means_array = means_array[:k]
            std_array = np.maximum(std_array[:k], 1e-8)

            p_sum = float(probs.sum())
            if p_sum > 0.0:
                probs = probs / p_sum
                mu = float(np.sum(probs * means_array))

                if sample_bonus:
                    second_moment = float(np.sum(probs * (std_array**2 + means_array**2)))
                    var = max(second_moment - mu**2, 1e-8)
                    bonus = float(np.random.normal(loc=mu, scale=np.sqrt(var)))
                else:
                    bonus = mu

                bonus *= float(mix_scale)
                if clip_bonus is not None:
                    c = float(abs(clip_bonus))
                    bonus = float(np.clip(bonus, -c, c))
                # print(f"Probs: {probs}")
                # print(f"Neural Signal: {neural_signal}")
                # print(f"Bonus: {bonus}")
                # print(f"Reward: {reward}")
                # print(f"Adjusted Reward: {reward + bonus}")
                return float((reward + bonus) * beta)
    # print("not working")
    return float((reward + means[neural_signal])*beta)

def adjust_epsilon(epsilon: float, neural_signal: float, verbose = False):
    """
    adjust_epsilon(epsilon: float, 
        neural_signal: float) -> adjusted_epsilon: float
    
    This function adjusts epsilon to modulate exploration given the neural signal. 
    """

    # neural signal in tenths place, subtracted from current decay
    new_epsilon = epsilon - (neural_signal/20)

    if verbose:
        print(f"Original Epsilon: {epsilon}, Neural Signal: {neural_signal}, Adjusted Epsilon: {new_epsilon}")
    
    # epsilon is not lower than 0.05 or higher than 1.0
    return min(max(0.05, new_epsilon), 1.0)


# def adjust_action(state, neural_signal, agent, verbose = False):
#     action_distribution = agent.chooseAction(state, epislon = 0.0, returnDist = True)
    
#     argmax_idx = np.argmax(action_distribution)
#     action_distribution[argmax_idx] += neural_signal

#     new_argmax_idx = np.argmax(action_distribution)

#     if verbose:
#         print(f"Old Action: {argmax_idx}, New Action: {new_argmax_idx}")

#     return new_argmax_idx

def get_neural_signal(clf, features):

        if features is not None:
            classification = clf.predict(features[None, :])[0] #predict with model
            probs = clf.predict_proba(features[None, :])[0]
            # bonus = adjust_neural_classification(neural_classification)
        else:
            classification = 0.0
            probs = 0.0

        return classification, probs

def evaluate(env, agent, steps=600, episodes=20, domain_key=None):
    """
    Agent evaluation function
    """
    rewards = []
    successes = 0
    for i in range(episodes):
        ep_reward = 0.0
        if domain_key == "F":
            state, _  = env.reset()
        else:
            state = env.reset()

        for idx_step in range(steps):
            action, _ = agent.chooseAction(state, epsilon=0)

            if domain_key == "F":
                state, reward, done, win, info = env.step(action)
                if info['score'] >= 10:
                    win = True
                    done = True
            else:
                state, reward, done, win = env.step(action)
            
            ep_reward += reward

            if done:
                if win:
                    successes += 1
                break
        
        rewards.append(ep_reward)
    return np.array(rewards), successes/episodes

def evaluate_fetch(env, agent, steps=50, episodes=20):
    """Evaluate DDPG on a goal-conditioned Fetch env (dict observations)."""
    successes = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        for _ in range(steps):
            action = agent.choose_action(
                obs["observation"],
                obs["desired_goal"],
                train_mode=False,
            )
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        successes += int(float(info.get("is_success", 0.0)))
    return successes / max(episodes, 1)
    
def torch_load_checkpoint(path: str, map_location=None):
    kwargs = {}
    if map_location is not None:
        kwargs["map_location"] = map_location
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        # PyTorch < 2.0 has no weights_only
        return torch.load(path, **kwargs)