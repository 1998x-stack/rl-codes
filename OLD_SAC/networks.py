# networks.py

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer Perceptron network architecture."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256) -> None:
        """
        Initializes a multi-layer perceptron.

        Args:
            input_dim (int): Input dimension.
            output_dim (int): Output dimension.
            hidden_dim (int): Hidden layer dimension.
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class QNetwork(nn.Module):
    """Q-network for SAC, used as the critic."""

    def __init__(self, input_dim: int, action_dim: int) -> None:
        """
        Initializes the Q-network.

        Args:
            input_dim (int): State dimension.
            action_dim (int): Action dimension.
        """
        super(QNetwork, self).__init__()
        self.fc1 = MLP(input_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass for Q-network."""
        x = torch.cat([state, action], dim=-1)
        return self.fc2(self.fc1(x))


class ValueNetwork(nn.Module):
    """Value function network for SAC."""

    def __init__(self, input_dim: int) -> None:
        """
        Initializes the value network.

        Args:
            input_dim (int): State dimension.
        """
        super(ValueNetwork, self).__init__()
        self.fc1 = MLP(input_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass for value network."""
        return self.fc2(self.fc1(state))