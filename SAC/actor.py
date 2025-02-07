"""
Actor network implementation for SAC.
"""

import torch
import torch.nn as nn
from typing import Tuple, List

class Actor(nn.Module):
    """
    Actor network that outputs a Gaussian distribution for actions.

    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        hidden_sizes (List[int]): List of hidden layer sizes.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int]):
        super(Actor, self).__init__()
        layers = []
        input_size = state_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.ReLU())
            input_size = size
        self.net = nn.Sequential(*layers)
        self.mu = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute mean and log standard deviation of the action distribution.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            mu (torch.Tensor): Mean of the action distribution.
            log_std (torch.Tensor): Log standard deviation of the action distribution.
        """
        x = self.net(state)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # 限制 log_std 的范围
        return mu, log_std