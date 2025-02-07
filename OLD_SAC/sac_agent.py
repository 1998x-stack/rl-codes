# sac_agent.py

import torch
import torch.optim as optim
from loguru import logger
from networks import ActorNetwork, QNetwork, ValueNetwork
from replay_buffer import ReplayBuffer
from config import HYPERPARAMETERS, DEVICE


class SACAgent:
    """Soft Actor-Critic agent for reinforcement learning on the Humanoid-v4 environment."""

    def __init__(self, state_dim: int, action_dim: int):
        """
        Initializes the SAC agent with networks, optimizers, and replay buffer.

        Args:
            state_dim (int): The dimensionality of the state space.
            action_dim (int): The dimensionality of the action space.
        """
        self.actor = ActorNetwork(state_dim, action_dim).to(DEVICE)
        self.critic_1 = QNetwork(state_dim, action_dim).to(DEVICE)
        self.critic_2 = QNetwork(state_dim, action_dim).to(DEVICE)
        self.value = ValueNetwork(state_dim).to(DEVICE)
        self.target_value = ValueNetwork(state_dim).to(DEVICE)
        
        # Target network initialization
        self.target_value.load_state_dict(self.value.state_dict())

        # Optimizers for networks
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=HYPERPARAMETERS["learning_rate"])
        self.critic_optimizer = optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=HYPERPARAMETERS["learning_rate"]
        )
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=HYPERPARAMETERS["learning_rate"])

        # Replay buffer initialization
        self.replay_buffer = ReplayBuffer(HYPERPARAMETERS["buffer_size"], HYPERPARAMETERS["batch_size"])

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Selects an action based on the policy network (actor).

        Args:
            state (torch.Tensor): The current state of the environment.

        Returns:
            torch.Tensor: The selected action.
        """
        self.actor.eval()  # Set actor to evaluation mode
        with torch.no_grad():
            action = self.actor(state.to(DEVICE)).cpu().numpy()
        return action

    def store_experience(self, experience: tuple) -> None:
        """Store experience in the replay buffer."""
        self.replay_buffer.add(experience)

    def update(self) -> None:
        """Update the SAC agent's networks."""
        if self.replay_buffer.size() < HYPERPARAMETERS["batch_size"]:
            return

        batch = self.replay_buffer.sample()
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.float32).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(DEVICE)

        with torch.no_grad():
            target_value = self.target_value(next_states)
            target_q1 = rewards + HYPERPARAMETERS["gamma"] * (1 - dones) * target_value
            target_q2 = rewards + HYPERPARAMETERS["gamma"] * (1 - dones) * target_value

        # Critic loss
        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        critic_loss = torch.mean((q1 - target_q1) ** 2 + (q2 - target_q2) ** 2)

        # Value loss
        value_loss = torch.mean((self.value(states) - target_value) ** 2)

        # Actor loss
        new_actions = self.actor(states)
        q1_new = self.critic_1(states, new_actions)
        q2_new = self.critic_2(states, new_actions)
        actor_loss = torch.mean(HYPERPARAMETERS["alpha"] * (q1_new + q2_new) - self.value(states))

        # Update networks
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target value network
        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(HYPERPARAMETERS["tau"] * param.data + (1.0 - HYPERPARAMETERS["tau"]) * target_param.data)

        logger.info("Updated networks: Critic loss: {:.3f}, Value loss: {:.3f}, Actor loss: {:.3f}".format(
            critic_loss.item(), value_loss.item(), actor_loss.item()))