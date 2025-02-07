"""
Experience replay buffer implementation.
"""

import numpy as np
from typing import Tuple

class ReplayBuffer:
    """
    Replay buffer for storing and sampling experiences.

    Args:
        capacity (int): Maximum number of experiences to store.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Add a new experience to the buffer.

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            done (bool): Whether the episode ended.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences from the buffer.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            states (np.ndarray): Batch of states.
            actions (np.ndarray): Batch of actions.
            rewards (np.ndarray): Batch of rewards.
            next_states (np.ndarray): Batch of next states.
            dones (np.ndarray): Batch of done flags.
        """
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self) -> int:
        return len(self.buffer)