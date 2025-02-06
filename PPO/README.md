# PPO on Humanoid in MuJoCo

## Overview
This project implements the Proximal Policy Optimization (PPO) algorithm to train an agent in the MuJoCo humanoid environment using PyTorch.

## Setup

1. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the training**:
    ```bash
    python train.py
    ```

3. **TensorBoard**:
    To visualize the training process:
    ```bash
    tensorboard --logdir=./tensorboard_logs
    ```

## Hyperparameters
- Gamma: 0.99
- Tau: 0.95
- Learning Rate: 3e-4
- Number of epochs: 10
- Mini batch size: 64