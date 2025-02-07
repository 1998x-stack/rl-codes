"""
Critic network implementation for SAC.
"""

import torch
import torch.nn as nn
from typing import List

class Critic(nn.Module):
    """
    Critic network that estimates the action-value function.

    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        hidden_sizes (List[int]): List of hidden layer sizes.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int]):
        super(Critic, self).__init__()
        layers = []
        input_size = state_dim + action_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.ReLU())
            input_size = size
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the action-value.

        Args:
            state (torch.Tensor): Input state tensor.
            action (torch.Tensor): Input action tensor.

        Returns:
            q_value (torch.Tensor): Estimated action-value.
        """
        x = torch.cat([state, action], dim=1)
        return self.net(x)