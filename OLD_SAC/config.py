# config.py

import os, torch

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Hyperparameters for SAC
HYPERPARAMETERS = {
    "buffer_size": int(1e6),         # Replay buffer size
    "batch_size": 256,               # Mini-batch size
    "gamma": 0.99,                   # Discount factor
    "tau": 0.005,                    # Target network update rate
    "alpha": 0.2,                    # Entropy regularization coefficient
    "learning_rate": 3e-4,           # Learning rate for all networks
    "epsilon": 1e-8,                 # Small value for numerical stability
    "total_episodes": 1000,          # Total number of episodes for training
    "max_timesteps_per_episode": 1000 # Max steps per episode
}

# Device settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TensorboardX and Loguru logs
TENSORBOARD_LOG_DIR = os.path.join(LOG_DIR, "tensorboard_logs")
LOGURU_LOG_FILE = os.path.join(LOG_DIR, "training.log")