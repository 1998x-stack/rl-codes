# 配置文件
class Config:
    # 环境设置
    ENV_NAME = 'Humanoid-v4'    # 环境名称
    SEED = 42                   # 随机种子
    
    # 超参数
    GAMMA = 0.99               # 折扣因子
    TAU = 0.95                 # GAE lambda参数
    LR = 3e-4                  # Adam优化器学习率
    EPOCHS = 10                # 每次更新的训练轮数
    MINI_BATCH_SIZE = 64       # 小批量大小
    CLIP_EPSILON = 0.2         # PPO裁剪参数
    VALUE_CLIP_EPSILON = 0.2   # 价值函数裁剪参数
    MAX_GRAD_NORM = 0.5        # 梯度裁剪阈值
    UPDATE_STEPS = 10          # 每次更新的步数
    
    # 日志设置
    LOG_DIR = './logs'                    # 日志保存目录
    TENSORBOARD_LOG_DIR = './logs/tensorboard_logs'  # Tensorboard日志目录
    MODEL_DIR = './model'                 # 模型保存目录
    
    # 模型保存
    SAVE_MODEL_INTERVAL = 1000  # 保存模型的步数间隔