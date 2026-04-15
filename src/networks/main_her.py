import gymnasium as gym
import gymnasium_robotics
from torch.utils.tensorboard import SummaryWriter

import DDPG_HER

# Hyperparameters
NUM_EPOCHS = 500
NUM_CYCLES = 50
NUM_EPISODES = 16
NUM_STEPS = 50
TRAIN_BATCHES = 40
WARMUP_TRANSITIONS = 10000
BATCH_SIZE = 256
BUFFER_CAPACITY = 1000000

# PER parameters
PER_ALPHA = 0.6  # Prioritization exponent (0 = uniform, 1 = full prioritization)
PER_BETA_START = 0.4  # Initial importance sampling correction
PER_BETA_FRAMES = NUM_EPOCHS * NUM_CYCLES * TRAIN_BATCHES  # Anneal over full training (500*50*40 = 1,000,000)

# HER parameters
K_FUTURE = 4  # Number of future goals to sample

# Setup
gym.register_envs(gymnasium_robotics)
env = gym.make("FetchPickAndPlace-v4", max_episode_steps=NUM_STEPS)
# writer = SummaryWriter("logs/DDPG_HER")

# Initialize replay buffer and agent
ram = DDPG_HER.PrioritizedReplayBuffer(
    capacity=BUFFER_CAPACITY,
    k_future=K_FUTURE,
    env=env.unwrapped,
    alpha=PER_ALPHA,
    beta_start=PER_BETA_START,
    beta_frames=PER_BETA_FRAMES,
)

agent = DDPG_HER.Agent(
    n_states=(25,),
    n_actions=4,
    n_goals=3,
    action_bounds=(-1, 1),
    ram=ram,
    batch_size=BATCH_SIZE,
)

# Training loop
max_success_rate = 0.0
global_step = 0

print("Starting DDPG+HER+PER training...")
print(f"Environment: FetchPickAndPlace-v4")
print(f"PER Alpha: {PER_ALPHA}, Beta Start: {PER_BETA_START}")
print("-" * 50)

for epoch in range(NUM_EPOCHS):
    epoch_success = 0
    epoch_episodes = 0
    epoch_metrics = {"actor_loss": 0, "critic_loss": 0, "q_pred_mean": 0, "td_error": 0, "reward": 0}
    train_count = 0

    for cycle in range(NUM_CYCLES):
        # Collect episodes
        for _ in range(NUM_EPISODES):
            obs, info = env.reset()
            episode = {
                "state": [],
                "action": [],
                "reward": [],
                "next_state": [],
                "achieved_goal": [],
                "next_achieved_goal": [],
                "desired_goal": [],
                "done": [],
            }

            for step in range(NUM_STEPS):
                state = obs["observation"]
                achieved_goal = obs["achieved_goal"]
                desired_goal = obs["desired_goal"]

                action = agent.choose_action(state, desired_goal)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = truncated

                episode["state"].append(state)
                episode["action"].append(action)
                episode["reward"].append(reward)
                episode["next_state"].append(next_obs["observation"])
                episode["achieved_goal"].append(achieved_goal)
                episode["next_achieved_goal"].append(next_obs["achieved_goal"])
                episode["desired_goal"].append(desired_goal)
                episode["done"].append(float(done))

                obs = next_obs
                global_step += 1

                if done:
                    epoch_success += int(info.get("is_success", 0))
                    epoch_episodes += 1
                    break

            ram.add_episode(episode, agent.state_normalizer, agent.goal_normalizer)

        # Train after each cycle
        if ram.len_transition >= WARMUP_TRANSITIONS:
            for _ in range(TRAIN_BATCHES):
                metrics = agent.train()
                if metrics:
                    for k, v in metrics.items():
                        epoch_metrics[k] += v
                    train_count += 1

            agent.update_target_networks()

    # Logging
    success_rate = epoch_success / max(epoch_episodes, 1)

    if train_count > 0:
        avg_actor_loss = epoch_metrics["actor_loss"] / train_count
        avg_critic_loss = epoch_metrics["critic_loss"] / train_count
        avg_q_pred = epoch_metrics["q_pred_mean"] / train_count
        avg_td_error = epoch_metrics["td_error"] / train_count
        avg_reward = epoch_metrics["reward"] / train_count

        # writer.add_scalar("Loss/Actor", avg_actor_loss, epoch)
        # writer.add_scalar("Loss/Critic", avg_critic_loss, epoch)
        # writer.add_scalar("Q/Predicted", avg_q_pred, epoch)
        # writer.add_scalar("TD_Error", avg_td_error, epoch)
        # writer.add_scalar("Reward", avg_reward, epoch)
        # writer.add_scalar("PER/Beta", ram.beta, epoch)

        print(f"Epoch {epoch:3d} | Success Rate: {success_rate:.2f} | "
              f"Actor Loss: {avg_actor_loss:.4f} | Critic Loss: {avg_critic_loss:.4f} | "
              f"TD Error: {avg_td_error:.4f} | Beta: {ram.beta:.3f}")
    else:
        print(f"Epoch {epoch:3d} | Success Rate: {success_rate:.2f} | Warming up...")

    # writer.add_scalar("Success_Rate", success_rate, epoch)

    # Save best model
    if success_rate > max_success_rate:
        max_success_rate = success_rate
        agent.save("best_model.pth")
        print(f"  -> New best model saved! (Success Rate: {success_rate:.2f})")

env.close()
# writer.close()

print("-" * 50)
print(f"Training finished!")
print(f"Max success rate achieved: {max_success_rate:.2f}")
print("Best model saved as best_model.pth")