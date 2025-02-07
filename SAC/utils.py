import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """
    设置全局随机种子，确保结果可复现。

    Args:
        seed (int): 随机种子。
    """
    # 设置 Python 的随机种子
    random.seed(seed)

    # 设置 NumPy 的随机种子
    np.random.seed(seed)

    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU

    # 确保 PyTorch 的卷积操作是确定性的（可能导致性能下降）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False