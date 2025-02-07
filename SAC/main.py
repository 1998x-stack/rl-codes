"""
Main script for training the SAC agent.
"""

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from config import *
from sac_agent import SACAgent
from loguru import logger
from utils import set_seed

def train_sac():
    """
    Train the SAC agent on the Humanoid-v4 environment.
    """
    # 设置随机种子
    set_seed(SEED)
    
    logger.add(f"{LOG_DIR}/train.log", rotation="10 MB", encoding="utf-8")
    logger.info(f"Using device: {DEVICE}")

    # Initialize environment
    env = gym.make(ENV_NAME, render_mode=RENDER_MODE)
    env = RecordEpisodeStatistics(env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize agent
    agent = SACAgent(state_dim, action_dim)

    # Training loop
    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        episode_reward = 0.0

        for step in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store experience in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Train agent
            agent.train()

            state = next_state
            episode_reward += reward

            if done:
                break

        logger.info(f"Episode: {episode + 1}, Reward: {episode_reward:.2f}")
        agent.writer.add_scalar("Reward/episode", episode_reward, episode)

    env.close()
    agent.writer.close()

if __name__ == "__main__":
    train_sac()