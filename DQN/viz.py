import re
import matplotlib.pyplot as plt

# Function to parse the log file and extract relevant data
def extract_log_data(log_file):
    episodes = []
    rewards = []
    epsilons = []

    with open(log_file, 'r') as file:
        for line in file:
            # Regex pattern to extract the Episode, Total Reward, and Epsilon
            match = re.search(r"Episode: (\d+), Total Reward: (\d+\.\d+), Epsilon: (\d+\.\d+)", line)
            if match:
                episode = int(match.group(1))
                reward = float(match.group(2))
                epsilon = float(match.group(3))
                
                episodes.append(episode)
                rewards.append(reward)
                epsilons.append(epsilon)

    return episodes, rewards, epsilons

# Function to plot the extracted data
def plot_data(episodes, rewards, epsilons):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Total Reward vs Episode
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward', color='tab:blue')
    ax1.plot(episodes, rewards, color='tab:blue', label='Total Reward')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a second y-axis to plot Epsilon
    ax2 = ax1.twinx()
    ax2.set_ylabel('Epsilon', color='tab:green')
    ax2.plot(episodes, epsilons, color='tab:green', label='Epsilon', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # Title and legend
    plt.title('Total Reward and Epsilon per Episode')
    fig.tight_layout()

    # Save the plot to a file
    plt.savefig('training_data_plot.png')
    plt.show()

# Main code to execute the extraction and plotting
log_file = 'logs/train_log.log'  # Path to your log file
episodes, rewards, epsilons = extract_log_data(log_file)
plot_data(episodes, rewards, epsilons)