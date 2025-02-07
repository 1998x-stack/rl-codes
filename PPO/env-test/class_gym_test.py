import gym
from loguru import logger

# Configure the logger to write logs to a file with a timestamp
logger.add("env-test/logs/gym_tests.log", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")

def test_environment(env_name: str):
    """
    Function to test a single environment by running a simple test.
    
    Args:
        env_name (str): The name of the environment to test.
    """
    try:
        # Create the environment
        env = gym.make(env_name)
        env.reset()

        # Running a simple test: Take 10 random actions and log results
        for _ in range(10):
            action = env.action_space.sample()  # Random action
            state, reward, done, truncated, info = env.step(action)
            logger.info(f"Action: {action} | State: {state} | Reward: {reward} | Done: {done}")
            
            if done:
                break

        env.close()
        logger.success(f"Successfully tested environment: {env_name}")
    except Exception as e:
        logger.error(f"Failed to test environment {env_name}: {e}")

def test_all_classical_environments():
    """
    Function to test all classical environments in Gym.
    """
    classical_envs = [
        'CartPole-v1',
        'MountainCar-v0',
        'Pendulum-v1',
        'Acrobot-v1',
        'LunarLander-v2',
        'FrozenLake-v1',
        'Taxi-v3',
    ]

    for env_name in classical_envs:
        logger.info(f"Starting test for {env_name}")
        test_environment(env_name)

if __name__ == "__main__":
    test_all_classical_environments()
