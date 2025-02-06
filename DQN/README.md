# DQN for Gym Environments

This project implements a Deep Q-Network (DQN) using PyTorch to train an agent on various classical environments from OpenAI's Gym. It includes the following components:

- **DQN model**: A neural network with two fully connected layers.
- **Replay buffer**: Stores agent's experiences for batch training.
- **Training pipeline**: Trains the model with an epsilon-greedy policy and updates the target network periodically.

## Requirements

The following libraries are required to run the project:

- Python 3.7+
- gym==0.26.0
- torch==1.12.1
- loguru==0.6.0

You can install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

## How to Run

1. **Set the environment**:
   Modify the `config.py` file to select the Gym environment you want to train the agent on (e.g., `CartPole-v1`).

2. **Start Training**:
   Run the `train.py` script to begin training the DQN agent. The agent will interact with the environment, store experiences in a replay buffer, and update its Q-values.

```bash
python train.py
```

## Logging

Training logs will be stored in `train_log.log`. You can track the progress of training, loss, and target network updates.

## Hyperparameters

- The hyperparameters such as learning rate, batch size, epsilon decay, and target network update frequency can be modified in the `config.py` file.

## License

This project is licensed under the MIT License.