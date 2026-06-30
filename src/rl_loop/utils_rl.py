import numpy as np
import csv
import os
import json
import torch

_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def td_priority(agent, algorithm, reward, action, state, next_state, done=None, goal=None, q_augmentation=0.0, buffer_type="PER"):
    """TD Error for PER"""

    # Set priority to 1.0 for ER buffers (uniform sampling)
    if buffer_type != "PER":
        return 1.0

    # Calculate TD error for DQN or DDPG
    with torch.no_grad():
        if algorithm.upper() == "DQN": # DQN
            action = int(action)
            s = torch.from_numpy(np.asarray(state, dtype=np.float32)).float().unsqueeze(0).to(_device)
            ns = torch.from_numpy(np.asarray(next_state, dtype=np.float32)).float().unsqueeze(0).to(_device)
            q_eval = agent.policy_net(s).squeeze(0)[action]
            if done is not None and bool(np.asarray(done).item()):
                target = float(reward)
            else:
                target = float(reward) + agent.gamma * agent.target_net(ns).squeeze(0).max().item()
            return abs(target - q_eval.item())

        # DDPG TD error
        dev = agent.device
        state_n = agent.state_normalizer.normalize(np.asarray(state, dtype=np.float32))
        next_state_n = agent.state_normalizer.normalize(np.asarray(next_state, dtype=np.float32))
        goal_n = agent.goal_normalizer.normalize(np.asarray(goal, dtype=np.float32))
        sg = torch.tensor(np.concatenate([state_n, goal_n]), dtype=torch.float32, device=dev).unsqueeze(0)
        nsg = torch.tensor(np.concatenate([next_state_n, goal_n]), dtype=torch.float32, device=dev).unsqueeze(0)
        a = torch.tensor(np.asarray(action, dtype=np.float32), device=dev).unsqueeze(0)
        q = agent.critic(sg, a).squeeze()
        tq = agent.critic_target(nsg, agent.actor_target(nsg)).squeeze()
        target = torch.clamp(torch.tensor(float(reward), device=dev) + agent.gamma * tq + float(q_augmentation), -1 / (1 - agent.gamma), 0)
        return float(torch.abs(target - q).item())


class Results():
    '''
    Results: Saving hyperparameters and final results for experiments
    '''
    def save_results(episodes, total_rewards, success_rate, steps, experiment_list, index_of_interest, save_to_csv = False, filepath = "/Users/juliasantaniello/Desktop/OfflineNeuroloop/results"):
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
            "index_of_interest": index_of_interest,
        }

        if save_to_csv:
            write_header = not os.path.exists(filepath)

            with open(filepath, mode='a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=row.keys())

                if write_header:
                    writer.writeheader()

                writer.writerow(row)

        return row

def adjust_signal(
    reward: float,
    neural_signal: int | float,
    clf_probs=None,
    means: tuple[float, float, float] = (1.0, -0.1, -1.0),
    stds: tuple[float, float, float] = (0.25, 0.25, 0.25),
    clip_bonus: float | None = None,
    beta: float = 1.0,
):
    """Adjust reward based on neural signal and classification probabilities. 
    Gaussian mixture model used to adjust reward.
    """

    # estimate standard deviation of the target distribution
    std = (means[0]-means[2])*0.1

    # if zero, set to a small value
    if std == 0:
        std = 0.1

    stds = [std, std, std]

    # for continuous output
    if isinstance(clf_probs, str):
        # Reverse error to mean optimality: 1 - error = optimality
        optimal_neural_value = (1 - neural_signal)

        #shift distribution to be between -1 and 1
        optimal_neural_value = (optimal_neural_value - 0.5) * 2

        # adjust reward based on the optimal neural value
        return float((reward + optimal_neural_value*means[0])*beta)
        
    elif clf_probs is not None and not np.isscalar(clf_probs):
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
                # weight means by probabilities
                means_array = probs * means_array

                #return weighted mean associated with the neural signal classification
                return float((reward + means_array[neural_signal])*beta)

    return float((reward + means[neural_signal])*beta)

def get_neural_signal(clf, features):
    """Get neural signal and classification probabilities from features"""
    if features is not None:
        classification = clf.predict(features[None, :])[0] #predict with model

        try:
            probs = clf.predict_proba(features[None, :])[0]
        except:
            probs = "regression"
        
    else:
        classification = 0.0
        probs = 0.0

    return classification, probs

def evaluate(env, agent, steps=600, episodes=20, domain_key=None, random_seed=0):
    """
    Agent evaluation function
    """
    rewards = []
    successes = 0
    np.random.seed(random_seed)
    seed =np.random.randint(0, 1000000)
    for i in range(episodes):

        ep_reward = 0.0

        if domain_key == "F":
            state, _  = env.reset(seed=seed)
        else:
            state = env.reset(seed=seed)

        final_win = False
        for idx_step in range(steps):
            action, _ = agent.chooseAction(state, epsilon=0)

            if domain_key == "F":
                state, reward, done, win, info = env.step(action)
                if info['score'] >= 10:
                    #done = True
                    final_win = True
            else:
                state, reward, done, win = env.step(action)
                final_win = win
            
            ep_reward += reward
            if done or idx_step == steps-1:
                if final_win:
                    successes += 1
                break
        
        seed += 1
        rewards.append(ep_reward)
    return np.array(rewards), successes/episodes

def evaluate_fetch(env, agent, steps=50, episodes=20, random_seed=0):
    """Evaluate DDPG on a goal-conditioned Fetch env (dict observations)."""
    successes = 0
    np.random.seed(random_seed)
    seed = np.random.randint(0, 1000000)
    rewards = []

    for _ in range(episodes):
        obs, _ = env.reset(seed=seed)
        ep_reward = 0.0
        for _ in range(steps):
            action = agent.choose_action(
                obs["observation"],
                obs["desired_goal"],
                train_mode=False,
            )
            obs, _, terminated, truncated, info = env.step(action)
            ep_reward += info.get("reward", 0.0)
            if terminated or truncated:
                break
        successes += int(float(info.get("is_success", 0.0)))
        seed += 1
        rewards.append(ep_reward)
    return successes / max(episodes, 1), np.array(rewards)
    
def torch_load_checkpoint(path: str, map_location=None):
    kwargs = {}
    if map_location is not None:
        kwargs["map_location"] = map_location
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        # PyTorch < 2.0 has no weights_only
        return torch.load(path, **kwargs)