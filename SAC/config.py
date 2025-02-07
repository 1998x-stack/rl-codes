"""
Configuration file for SAC implementation.
"""

from typing import List

import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Environment Configuration
ENV_NAME: str = "Humanoid-v5"
RENDER_MODE: str = "rgb_array"
SEED: int = 42

# Network Architecture
ACTOR_HIDDEN_SIZES: List[int] = [256, 256]
CRITIC_HIDDEN_SIZES: List[int] = [256, 256]

# Training Hyperparameters
BATCH_SIZE: int = 256
LR_ACTOR: float = 3e-4
LR_CRITIC: float = 3e-4
GAMMA: float = 0.99
TAU: float = 0.005
ALPHA: float = 0.2
REPLAY_BUFFER_CAPACITY: int = 1000000
MAX_EPISODES: int = 10000
MAX_STEPS: int = 1000

# Logging and Visualization
LOG_DIR: str = "logs"
TB_LOG_DIR: str = "tensorboard_logs"