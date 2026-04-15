import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
# from torch.utils.tensorboard import SummaryWriter

from DDPG import MemoryBuffer, PrioritizedMemoryBuffer, DDPG

# Fast baseline for FetchPickAndPlace-v4 using episode/step loops.
NUM_EPISODES = 8000
NUM_STEPS = 50
WARMUP_TRANSITIONS = 200
UPDATES_PER_STEP = 1
EVAL_EVERY = 20
EVAL_EPISODES = 20
USE_PER = True

BATCH_SIZE = 128
BUFFER_CAPACITY = 100000
K_FUTURE = 4
PER_ALPHA = 0.6

TAU = 0.005
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
GAMMA = 0.98
DECAY_RATE = 0.995


def build_episode():
    return {
        "state": [],
        "action": [],
        "reward": [],
        "next_state": [],
        "achieved_goal": [],
        "next_achieved_goal": [],
        "desired_goal": [],
        "done": [],
    }


def evaluate_policy(env, agent, episodes, max_steps):
    successes = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        info = {}
        for _ in range(max_steps):
            action = agent.choose_action(
                obs["observation"],
                obs["desired_goal"],
                train_mode=False,
            )
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                break
        successes.append(float(info.get("is_success", 0.0)))
    return float(np.mean(successes))


def main():
    gym.register_envs(gymnasium_robotics)
    env = gym.make("FetchPickAndPlace-v4", max_episode_steps=NUM_STEPS)
    # writer = SummaryWriter("logs/fetch_ddpg_her")

    obs, _ = env.reset()
    n_states = (obs["observation"].shape[0],)
    n_goals = obs["desired_goal"].shape[0]
    n_actions = env.action_space.shape[0]
    action_bounds = (env.action_space.low, env.action_space.high)

    print(n_states, n_goals, n_actions, action_bounds)

    if USE_PER:
        ram = PrioritizedMemoryBuffer(
            size=BUFFER_CAPACITY,
            k_future=K_FUTURE,
            env=env.unwrapped,
            alpha=PER_ALPHA,
        )
    else:
        ram = MemoryBuffer(
            size=BUFFER_CAPACITY,
            k_future=K_FUTURE,
            env=env.unwrapped,
        )

    agent = DDPG(
        n_states=n_states,
        n_actions=n_actions,
        n_goals=n_goals,
        action_bounds=action_bounds,
        capacity=BUFFER_CAPACITY,
        env=env.unwrapped,
        k_future=K_FUTURE,
        batch_size=BATCH_SIZE,
        ram=ram,
        action_size=n_actions,
        tau=TAU,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        gamma=GAMMA,
        decay_rate=DECAY_RATE,
    )

    best_eval_success = 0.0
    global_transitions = 0

    print("Starting FetchPickAndPlace-v4 DDPG+HER training...")

    for episode_idx in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset()
        ep = build_episode()
        episode_success = 0.0

        for _ in range(NUM_STEPS):
            state = obs["observation"]
            achieved_goal = obs["achieved_goal"]
            desired_goal = obs["desired_goal"]

            action = agent.choose_action(state, desired_goal, train_mode=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep["state"].append(state)
            ep["action"].append(action)
            ep["reward"].append(reward)
            ep["next_state"].append(next_obs["observation"])
            ep["achieved_goal"].append(achieved_goal)
            ep["next_achieved_goal"].append(next_obs["achieved_goal"])
            ep["desired_goal"].append(desired_goal)
            ep["done"].append(float(done))

            obs = next_obs
            global_transitions += 1
            episode_success = float(info.get("is_success", 0.0))

            if ram.len_transition >= WARMUP_TRANSITIONS:
                for _ in range(UPDATES_PER_STEP):
                    agent.train()
                    agent.update_target_networks()

            if done:
                break

        ram.add_episode(ep)

        # writer.add_scalar("train/episode_success", episode_success, episode_idx)
        # writer.add_scalar("train/random_action_prob", agent.randomprob, episode_idx)

        if episode_idx % EVAL_EVERY == 0:
            eval_success = evaluate_policy(env, agent, EVAL_EPISODES, NUM_STEPS)
            # writer.add_scalar("eval/success_rate", eval_success, episode_idx)
            print(
                f"Episode {episode_idx:4d} | Eval success: {eval_success:.3f} "
                f"| Replay transitions: {ram.len_transition}"
            )

            if eval_success >= best_eval_success:
                best_eval_success = eval_success
                torch.save(
                    {
                        "episode": episode_idx,
                        "actor_state_dict": agent.actor.state_dict(),
                        "critic_state_dict": agent.critic.state_dict(),
                        "actor_target_state_dict": agent.actor_target.state_dict(),
                        "critic_target_state_dict": agent.critic_target.state_dict(),
                        "actor_optimizer_state_dict": agent.actor_optim.state_dict(),
                        "critic_optimizer_state_dict": agent.critic_optim.state_dict(),
                        "best_eval_success": best_eval_success,
                    },
                    "best_fetch_ddpg_her.pth",
                )

    env.close()
    # writer.close()
    print(f"Training complete. Best eval success: {best_eval_success:.3f}")


if __name__ == "__main__":
    main()