""" Utilities for the RL loop """

import csv
import datetime
import json
import os

import numpy as np
import torch

_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Calculates the TD error for the PER buffer
def td_priority(agent, algorithm, reward, action, state, next_state, done=None, goal=None, q_augmentation=0.0, buffer_type="PER"):
    """TD error magnitude for PER (matches the stored Bellman target, including q_aug)."""

    # Set priority to 1.0 for ER buffers (uniform sampling)
    if buffer_type != "PER":
        return 1.0

    q_aug = float(q_augmentation)
    # Calculate TD error for DQN or DDPG
    with torch.no_grad():
        if algorithm.upper() == "DQN":  # DQN
            action = int(action)
            s = torch.from_numpy(np.asarray(state, dtype=np.float32)).float().unsqueeze(0).to(_device)
            ns = torch.from_numpy(np.asarray(next_state, dtype=np.float32)).float().unsqueeze(0).to(_device)
            q_eval = agent.policy_net(s).squeeze(0)[action]
            # Match DQN.learn: y = r + q_aug + gamma * max Q' * (1 - done)
            if done is not None and bool(np.asarray(done).item()):
                target = float(reward) + q_aug
            else:
                target = (
                    float(reward)
                    + q_aug
                    + agent.gamma * agent.target_net(ns).squeeze(0).max().item()
                )
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
        target = torch.clamp(
            torch.tensor(float(reward), device=dev) + agent.gamma * tq + q_aug,
            -1 / (1 - agent.gamma),
            0,
        )
        return float(torch.abs(target - q).item())


class Results():
    '''
    Results: Saving hyperparameters and final results for experiments
    '''
    def save_results(
        episodes,
        total_rewards,
        success_rate,
        steps,
        experiment_list,
        index_of_interest,
        save_to_csv=False,
        filepath=None,
        *,
        offline_metrics=None,
        decoder_fit_reports=None,
        model_hyperparameters=None,
    ):
        import datetime as _dt

        if filepath is None:
            Exception("Filename missing")

        print(f"Len Total Rewards: {len(total_rewards)}, Len Success Rate: {len(success_rate)}, Len Steps: {len(steps)}, Episodes: {episodes}")

        row = {
            "date": _dt.date.today(),
            "time": _dt.datetime.now(),
            "experiment_list": experiment_list,
            "episodes": episodes,
            "total_reward": json.dumps(list(map(float, total_rewards))),
            "success_rate": json.dumps(list(map(float, success_rate))),
            "steps": json.dumps(list(map(float, steps))),
            "index_of_interest": index_of_interest,
        }

        # Flatten a few scalar offline metrics for easy CSV filtering; keep full JSON too.
        om = offline_metrics or {}
        row["offline_metrics_json"] = json.dumps(om, default=_json_default)
        for key in (
            "n",
            "granularity",
            "accuracy",
            "macro_f1",
            "auc",
            "r2",
            "mse",
            "mae",
            "spearman",
            "spearman_p",
            "error",
        ):
            if key in om:
                row[f"offline_{key}"] = om[key]

        if decoder_fit_reports is not None:
            row["decoder_fit_reports_json"] = json.dumps(
                decoder_fit_reports, default=_json_default
            )
            holdouts = []
            if isinstance(decoder_fit_reports, dict):
                for pid, rep in decoder_fit_reports.items():
                    if not isinstance(rep, dict):
                        continue
                    hm = rep.get("holdout_metric")
                    if hm is not None and hm == hm:
                        holdouts.append(float(hm))
                        row[f"decoder_holdout_{pid}"] = float(hm)
                        if rep.get("metric_name"):
                            row[f"decoder_metric_name_{pid}"] = rep["metric_name"]
                        if "n_train" in rep:
                            row[f"decoder_n_train_{pid}"] = rep["n_train"]
                        if "n_holdout" in rep:
                            row[f"decoder_n_holdout_{pid}"] = rep["n_holdout"]
            if holdouts:
                row["decoder_holdout_mean"] = float(np.mean(holdouts))
                row["decoder_holdout_std"] = float(np.std(holdouts))

        if model_hyperparameters is not None:
            row["model_hyperparameters_json"] = json.dumps(
                model_hyperparameters, default=_json_default
            )

        if save_to_csv:
            write_header = not os.path.exists(filepath)

            with open(filepath, mode='a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=row.keys())

                if write_header:
                    writer.writeheader()

                writer.writerow(row)

        return row


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    return str(obj)


def summarize_offline_decoder(ml, classes_truth, classes_pred, granularity, flags) -> dict:
    """Print and return structured offline decoder metrics (empty if Baseline)."""
    if 0 in flags:
        return {}
    yt = np.asarray(classes_truth)
    yp = np.asarray(classes_pred)
    if len(yt) == 0 or len(yp) == 0:
        print("OFFLINE: no scoreable windows collected (n=0)")
        return {"n": 0, "error": "no samples", "granularity": str(granularity).lower()}
    metrics = ml.score_predictions(yt, yp, granularity)
    report = ml.get_report(yt, yp, (str(granularity)[0].lower() != "c"))
    print("OFFLINE (eval-aligned windows only, n=%d):\n" % len(yt), report)
    if metrics.get("auc") is not None:
        print(
            f"OFFLINE AUC={metrics['auc']:.3f}  "
            f"macroF1={metrics.get('macro_f1', float('nan')):.3f}"
        )
    elif metrics.get("spearman") is not None:
        print(
            f"OFFLINE Spearman={metrics['spearman']:.3f}  "
            f"R2={metrics.get('r2', float('nan')):.3f}  "
            f"MAE={metrics.get('mae', float('nan')):.3f}"
        )
    return metrics


def adjust_signal(
    reward: float,
    neural_signal: int | float,
    clf_probs=None,
    means: tuple[float, float, float] = (1.0, -0.1, -1.0),
    beta: float = 1.0,
):
    """Adjust reward based on neural signal and classification probabilities.

    Classification uses the predicted class only: ``P(ŷ) μ_ŷ``, not ``Σ_c P(c) μ_c``.
    """

    # for continuous output
    if isinstance(clf_probs, str):
        # Reverse error to mean optimality: 1 - error = optimality
        optimal_neural_value = (1 - neural_signal)

        # shift distribution to be between -1 and 1
        optimal_neural_value = (optimal_neural_value - 0.5) * 2

        # adjust reward based on the optimal neural value
        return float((reward + optimal_neural_value * means[0]) * beta)

    elif clf_probs is not None and not np.isscalar(clf_probs):
        probs = np.asarray(clf_probs, dtype=np.float64).ravel()
        means_array = np.asarray(means, dtype=np.float64).ravel()

        k = int(min(len(probs), len(means_array)))

        if k > 0:
            probs = probs[:k]
            means_array = means_array[:k]

            p_sum = float(probs.sum())
            if p_sum > 0.0:
                probs = probs / p_sum
                # weight means by probabilities
                means_array = probs * means_array

                # return weighted mean associated with the neural signal classification
                idx = int(neural_signal)
                idx = int(np.clip(idx, 0, k - 1))
                return float((reward + means_array[idx]) * beta)

    idx = int(neural_signal)
    means_array = np.asarray(means, dtype=np.float64).ravel()
    idx = int(np.clip(idx, 0, max(len(means_array) - 1, 0)))
    return float((reward + means_array[idx]) * beta)


def neuro_augment_transition(
    reward,
    *,
    neural_signal,
    clf_probs,
    means,
    beta,
    flags,
    agent,
    algorithm,
    action,
    state,
    next_state,
    done=None,
    goal=None,
    buffer_type="PER",
):
    """Apply reward / Q / priority flags; TD priority uses the final stored values.

    Order: reward aug → Q-aug → TD(|y−Q|) → optional neural priority nudge.
    Only reward uses domain expected-reward ``means`` (class-conditional).
    Q-aug and priority use the default class offsets ``(1.0, -0.1, -1.0)``.
    """
    q_augmentation = 0.0
    if 1 in flags:
        reward = adjust_signal(
            reward, neural_signal, clf_probs=clf_probs, means=means, beta=beta
        )
    if 3 in flags:
        q_augmentation = adjust_signal(
            0.0, neural_signal, clf_probs=clf_probs, beta=beta
        )

    priority = td_priority(
        agent,
        algorithm,
        reward,
        action,
        state,
        next_state,
        done=done,
        goal=goal,
        q_augmentation=q_augmentation,
        buffer_type=buffer_type,
    )
    if 2 in flags:
        priority = adjust_signal(
            abs(float(priority)),
            neural_signal,
            clf_probs=clf_probs,
            beta=beta,
        )
        priority = abs(float(priority))

    return float(reward), float(priority), float(q_augmentation)

def get_neural_signal(clf, features=None, participant=None, raw_window=None):
    """Get neural signal and classification probabilities.

    Prefer ``raw_window`` [T, C] when ``clf`` is an eval-compatible
    ``RobustWindowDecoder`` / ``SubjectDecoderBank`` so robust scaling +
    features match the within-subject eval pipeline.
    """
    if participant is not None and hasattr(clf, "set_participant"):
        clf.set_participant(participant)

    if raw_window is not None:
        Xraw = np.asarray(raw_window, dtype=float)
        if participant is not None and hasattr(clf, "predict_raw_for"):
            classification = clf.predict_raw_for(Xraw, participant)[0]
            try:
                probs = clf.predict_proba_raw_for(Xraw, participant)[0]
            except Exception:
                probs = "regression"
            return classification, probs
        if hasattr(clf, "predict_raw"):
            classification = clf.predict_raw(Xraw)[0]
            try:
                probs = clf.predict_proba_raw(Xraw)[0]
            except Exception:
                probs = "regression"
            return classification, probs

    if features is None:
        return 0.0, 0.0

    X = np.asarray(features)[None, :] if np.ndim(features) == 1 else np.asarray(features)

    if participant is not None and hasattr(clf, "predict_for"):
        classification = clf.predict_for(X, participant)[0]
        try:
            probs = clf.predict_proba_for(X, participant)[0]
        except Exception:
            probs = "regression"
        return classification, probs

    classification = clf.predict(X)[0]
    try:
        probs = clf.predict_proba(X)[0]
    except Exception:
        probs = "regression"

    return classification, probs


def classify_fnirs_at_time(
    clf,
    processor,
    timestamp,
    *,
    participant=None,
    window_duration_s: float = 8.0,
    fnirs_channels=None,
    granularity: str = "binary",
    buffer=None,
    shift: float = 0.0,
):
    """Eval-aligned classify with hemodynamic lag.

    For an event at ``timestamp``, features and pre-shifted labels are read from
    the window ending at ``timestamp + shift`` (brain at ``t`` ↔ label at
    ``t - shift``, matching ``build_windows`` / ``shift_labels_for_delay``).

    Returns
    -------
    neural_signal, clf_probs, y_true, scoreable
        ``scoreable`` is False when the window is incomplete or the label would
        be dropped under eval ambiguity rules (do not count toward OFFLINE F1).
    """
    from src.eval.channels import CHANNELS_8
    from src.models.decoder_bank import normalize_pid

    # Prefer channels the decoder was trained with (e.g. intensity-only / no DSphi).
    if fnirs_channels is not None:
        channels = list(fnirs_channels)
    elif hasattr(clf, "channels"):
        channels = list(clf.channels)
    elif hasattr(clf, "models") and participant is not None:
        m = clf.models.get(normalize_pid(participant))
        channels = list(getattr(m, "channels", None) or CHANNELS_8)
    else:
        channels = list(CHANNELS_8)

    lag = float(shift)
    raw_window = None
    if hasattr(processor, "get_fnirs_window"):
        raw_window = processor.get_fnirs_window(
            timestamp,
            window_duration_s=window_duration_s,
            fnirs_channels=channels,
            temporal_shift=lag,
        )

    # keep streaming buffer in sync for credit/smoothing (same delayed sample)
    if buffer is not None:
        fnirs_sample = processor.get_fnirs_sample(
            timestamp=timestamp, temporal_shift=lag, fnirs_channels=channels
        )
        buffer.add_sample(timestamp=timestamp, x=fnirs_sample, classification=0.0)

    if raw_window is None:
        return 0.0, 0.0, None, False

    neural_signal, clf_probs = get_neural_signal(
        features=None, clf=clf, participant=participant, raw_window=raw_window
    )
    if buffer is not None and len(buffer.classifications):
        buffer.classifications[-1] = neural_signal

    y_true, valid = processor.get_window_label(
        timestamp,
        window_duration_s=window_duration_s,
        granularity=granularity,
        temporal_shift=lag,
    )
    return neural_signal, clf_probs, y_true, bool(valid)

# Evaluates the agent on the Lunar nad Flappy environments
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

# Evaluates the DDPG agent on the Fetch environment
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
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        successes += int(float(info.get("is_success", 0.0)))
        seed += 1
        rewards.append(ep_reward)
    return successes / max(episodes, 1), np.array(rewards)
    
# Loads the checkpoint for the agent
def torch_load_checkpoint(path: str, map_location=None):
    kwargs = {}
    if map_location is not None:
        kwargs["map_location"] = map_location
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)