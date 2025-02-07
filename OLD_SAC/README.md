# SAC for Humanoid-v4 using PyTorch

This project implements the Soft Actor-Critic (SAC) reinforcement learning algorithm on the `Humanoid-v4` environment in the `MuJoCo` physics simulator.

## Setup

1. Install dependencies using the following command:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure that you have `MuJoCo` and the corresponding license installed. Please refer to the [MuJoCo website](https://www.roboti.us/index.html) for installation instructions.

## Training

To train the SAC agent, simply run the `train.py` script:
```bash
python train.py
```

## Logs

Logs will be saved in the `logs/` directory, including:
- TensorboardX logs for visualization of the training process.
- Loguru logs for detailed logging of the training process.