# config.py
"""
配置文件：定义了超参数和训练参数，用于深度Q网络（DQN）模型训练
"""

# 超参数
GAMMA = 0.99  # 折扣因子
LEARNING_RATE = 0.001  # 学习率
BATCH_SIZE = 32  # 批量大小
BUFFER_SIZE = 40000  # 回放缓冲区大小
TARGET_UPDATE_FREQ = 1000  # 更新目标网络的频率
EPSILON_START = 1.0  # 初始epsilon值（探索率）
EPSILON_MIN = 0.1  # 最小epsilon值
EPSILON_DECAY = 0.995  # epsilon衰减因子
MAX_EPISODES = 5000  # 最大训练回合数
MAX_STEPS = 200  # 每回合最大步数

# 环境配置
ENV_NAME = 'CartPole-v1'  # 默认环境，可以根据需求修改