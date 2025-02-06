# model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

class DQN(nn.Module):
    """
    DQN模型，基于两层全连接网络，输出每个动作的Q值
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        初始化DQN网络
        Args:
            input_dim (int): 状态空间的维度
            output_dim (int): 动作空间的维度
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，计算每个动作的Q值
        Args:
            x (torch.Tensor): 输入状态
        Returns:
            torch.Tensor: 每个动作对应的Q值
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)