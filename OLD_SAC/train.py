# train.py

import torch
import gym
from sac_agent import SACAgent
from config import HYPERPARAMETERS, DEVICE
from loguru import logger
from tensorboardX import SummaryWriter

def train():
    """Train the SAC agent in the Humanoid-v4 environment."""
    # Initialize the environment and agent
    env = gym.make("Humanoid-v4")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(state_dim, action_dim)
    writer = SummaryWriter()

    # Training loop
    for episode in range(HYPERPARAMETERS["total_episodes"]):
        state = env.reset()
        done = False
        episode_reward = 0
        timestep = 0

        while not done and timestep < HYPERPARAMETERS["max_timesteps_per_episode"]:
            action = agent.select_action(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            next_state, reward, done, _ = env.step(action)

            agent.store_experience((state, action, reward, next_state, done))
            agent.update()

            state = next_state
            episode_reward += reward
            timestep += 1

        logger.info(f"Episode {episode+1}, Reward: {episode_reward}")

        # Record episode reward to TensorBoard
        writer.add_scalar("Episode Reward", episode_reward, episode)

    writer.close()

if __name__ == "__main__":
    train()