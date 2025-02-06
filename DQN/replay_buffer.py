# replay_buffer.py
from collections import deque
import random
from typing import List, Tuple

class ReplayBuffer:
    """
    回放缓冲区，用于存储和采样智能体的经验
    """

    def __init__(self, max_size: int = 100000):
        """
        初始化回放缓冲区
        Args:
            max_size (int): 回放缓冲区的最大容量
        """
        self.buffer = deque(maxlen=max_size)

    def store(self, experience: Tuple):
        """
        将经验存储到回放缓冲区
        Args:
            experience (Tuple): 一次经验，包含(state, action, reward, next_state, done)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Tuple]:
        """
        随机从缓冲区采样一个batch的经验
        Args:
            batch_size (int): 采样的批量大小
        Returns:
            List[Tuple]: 采样的经验列表
        """
        return random.sample(self.buffer, batch_size)

    def size(self) -> int:
        """
        获取缓冲区的当前大小
        Returns:
            int: 当前缓冲区中的经验数量
        """
        return len(self.buffer)