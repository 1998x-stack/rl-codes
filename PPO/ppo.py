import torch, os
import torch.nn.functional as F
from torch import optim
from policy import PolicyNetwork
from utils import Logger
from config import Config
import gym
from tensorboardX import SummaryWriter


class PPO:
    def __init__(self):
        self.config = Config()
        self.env = gym.make(self.config.ENV_NAME)
        # self.env.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)
        self.model_dir = self.config.MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Check if CUDA is available and use it; otherwise, use CPU
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        
        self.policy = PolicyNetwork(self.env.observation_space.shape[0], self.env.action_space.shape[0]).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.LR)
        
        self.logger = Logger()
        self.writer = SummaryWriter(self.config.TENSORBOARD_LOG_DIR)

    def update(self, batch_data):
        states, actions, rewards, next_states, dones = batch_data
        
        # Convert to torch tensors and move to the appropriate device (CPU or CUDA)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Get current policy
        old_log_probs = self.policy.get_log_prob(states, actions)

        # Compute Advantage Estimates (GAE)
        advantages, returns = self.compute_gae(rewards, next_states, dones)

        # Update PPO
        for _ in range(self.config.EPOCHS):
            log_probs = self.policy.get_log_prob(states, actions)
            entropy = self.policy.entropy(states)
            
            # Clipped Surrogate Loss
            ratio = torch.exp(log_probs - old_log_probs)
            surrogate_loss = torch.min(ratio * advantages, torch.clamp(ratio, 1 - self.config.CLIP_EPSILON, 1 + self.config.CLIP_EPSILON) * advantages)
            
            # Value Loss
            value_loss = F.mse_loss(self.policy.value(states), returns)
            
            # Total loss (surrogate + value loss + entropy)
            total_loss = -surrogate_loss.mean() + 0.5 * value_loss - 0.01 * entropy.mean()
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.MAX_GRAD_NORM)
            self.optimizer.step()
        
        return total_loss.item()

    def compute_gae(self, rewards, next_states, dones):
        advantages = []
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.config.GAMMA * self.policy.value(next_states[i]) * (1 - dones[i]) - self.policy.value(next_states[i])
            gae = delta + self.config.GAMMA * self.config.TAU * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.policy.value(next_states[i]))
        return advantages, returns

    def train(self, num_episodes=10000):
        for episode in range(num_episodes):
            state, info = self.env.reset()
            done = False
            episode_reward = 0
            batch_data = []

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                action, log_prob, value = self.policy.act(state_tensor)
                next_state, reward, done, truncated, _ = self.env.step(action.cpu().numpy())
                
                batch_data.append((state, action, reward, next_state, done))
                state = next_state
                episode_reward += reward

                if len(batch_data) >= self.config.MINI_BATCH_SIZE:
                    loss = self.update(batch_data)
                    self.logger.log(f"Episode {episode}, Loss: {loss}")
                    self.writer.add_scalar('Loss/train', loss, episode)

            self.writer.add_scalar('Episode Reward', episode_reward, episode)
            self.logger.log(f"Episode {episode}, Reward: {episode_reward}")
            
            if episode % self.config.SAVE_MODEL_INTERVAL == 0:
                torch.save(self.policy.state_dict(), f"{self.model_dir}/model_{episode}.pth")
        
        self.writer.close()


if __name__ == '__main__':
    agent = PPO()
    agent.train()