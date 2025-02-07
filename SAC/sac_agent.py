"""
SAC agent implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
import numpy as np
from tensorboardX import SummaryWriter
from config import *
from actor import Actor
from critic import Critic
from replay_buffer import ReplayBuffer

class SACAgent:
    """
    Soft Actor-Critic (SAC) agent.

    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
    """

    def __init__(self, state_dim: int, action_dim: int):
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, ACTOR_HIDDEN_SIZES).to(DEVICE)
        self.critic1 = Critic(state_dim, action_dim, CRITIC_HIDDEN_SIZES).to(DEVICE)
        self.critic2 = Critic(state_dim, action_dim, CRITIC_HIDDEN_SIZES).to(DEVICE)
        self.target_critic1 = Critic(state_dim, action_dim, CRITIC_HIDDEN_SIZES).to(DEVICE)
        self.target_critic2 = Critic(state_dim, action_dim, CRITIC_HIDDEN_SIZES).to(DEVICE)
        
        self.init_weights(self.actor)
        self.init_weights(self.critic1)
        self.init_weights(self.critic2)
        self.init_weights(self.target_critic1)
        self.init_weights(self.target_critic2)

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=LR_CRITIC)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=LR_CRITIC)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

        # Initialize loggers
        self.logger = logger
        self.writer = SummaryWriter(TB_LOG_DIR)

        # Soft update target networks
        self._update_target_networks(tau=1.0)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.01)

    def _update_target_networks(self, tau: float = None) -> None:
        """
        Soft update target networks.

        Args:
            tau (float): Interpolation factor for soft update.
        """
        if tau is None:
            tau = TAU
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def act(self, state: np.ndarray) -> np.ndarray:
        """
        Select an action using the actor network.

        Args:
            state (np.ndarray): Current state.

        Returns:
            action (np.ndarray): Selected action.
        """
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        mu, log_std = self.actor(state)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        return action.clamp(-1.0, 1.0).detach().cpu().numpy()[0]

    def train(self) -> None:
        """
        Train the SAC agent.
        """
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        states = torch.tensor(states, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_mu, next_log_std = self.actor(next_states)
            next_std = torch.exp(next_log_std)
            dist = torch.distributions.Normal(next_mu, next_std)
            next_actions = dist.rsample()
            qf1_next = self.target_critic1(next_states, next_actions)
            qf2_next = self.target_critic2(next_states, next_actions)
            min_qf_next = torch.min(qf1_next, qf2_next)
            next_log_prob = dist.log_prob(next_actions).sum(dim=-1, keepdim=True)
            target_q = rewards + (1 - dones) * GAMMA * (min_qf_next - ALPHA * next_log_prob)

        # Optimize critics
        qf1 = self.critic1(states, actions)
        qf2 = self.critic2(states, actions)
        qf1_loss = torch.mean((qf1 - target_q) ** 2)
        qf2_loss = torch.mean((qf2 - target_q) ** 2)

        self.critic1_optimizer.zero_grad()
        qf1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        qf2_loss.backward()
        self.critic2_optimizer.step()

        # Optimize actor
        mu, log_std = self.actor(states)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        actions = dist.rsample()
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        qf1_pi = self.critic1(states, actions)
        qf2_pi = self.critic2(states, actions)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = torch.mean(ALPHA * log_prob - min_qf_pi)

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self._update_target_networks()

        # Log training metrics
        self.writer.add_scalar("Loss/critic1", qf1_loss.item())
        self.writer.add_scalar("Loss/critic2", qf2_loss.item())
        self.writer.add_scalar("Loss/actor", policy_loss.item())