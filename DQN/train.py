# train.py
import gym
from dqn_agent import DQNAgent
from config import ENV_NAME, MAX_EPISODES, MAX_STEPS
from loguru import logger  # 导入loguru用于日志记录

def train_dqn(env_name: str):
    """
    训练DQN智能体
    Args:
        env_name (str): Gym环境名称
    """
    # 创建环境
    env = gym.make(env_name)
    agent = DQNAgent(env)

    total_rewards = []

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            action = agent.epsilon_greedy(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.replay_buffer.store((state, action, reward, next_state, done))
            agent.train()

            state = next_state
            total_reward += reward

            if done:
                break

        total_rewards.append(total_reward)
        logger.info(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")

    return total_rewards

if __name__ == "__main__":
    logger.add("logs/train_log.log", level="INFO", rotation="1 day")  # 日志记录到文件
    train_dqn(ENV_NAME)