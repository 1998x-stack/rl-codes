import gym
import matplotlib.pyplot as plt
env = gym.make('Humanoid-v4')
episode_rewards = []
episode_lengths = []

for episode in range(100):
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
# Plotting the rewards per episode
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(episode_rewards)
plt.title('Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

# Plotting the episode lengths
plt.subplot(1, 2, 2)
plt.plot(episode_lengths)
plt.title('Episode Lengths')
plt.xlabel('Episode')
plt.ylabel('Length')

plt.tight_layout()
plt.savefig('test/humanoid_test.png')
