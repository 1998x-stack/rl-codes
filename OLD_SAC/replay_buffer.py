# replay_buffer.py

from collections import deque
import random
from typing import List, Tuple


class ReplayBuffer:
    """Replay buffer for storing experiences for the SAC agent."""

    def __init__(self, buffer_size: int, batch_size: int) -> None:
        """
        Initializes the replay buffer with specified size and batch size.

        Args:
            buffer_size (int): The size of the buffer.
            batch_size (int): The batch size for sampling experiences.
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience: Tuple) -> None:
        """Adds an experience to the buffer."""
        self.buffer.append(experience)

    def sample(self) -> List[Tuple]:
        """Samples a batch of experiences from the buffer."""
        return random.sample(self.buffer, self.batch_size)

    def size(self) -> int:
        """Returns the current size of the buffer."""
        return len(self.buffer)