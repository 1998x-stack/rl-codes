import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """策略网络：包含演员网络(Actor)和评论家网络(Critic)的神经网络结构"""
    def __init__(self, obs_size, action_size):
        """
        初始化策略网络
        obs_size: 观察空间维度
        action_size: 动作空间维度
        """
        super(PolicyNetwork, self).__init__()
        self.obs_size = obs_size          # 观察空间大小
        self.action_size = action_size    # 动作空间大小
        
        # 演员网络层定义
        self.actor_fc1 = nn.Linear(obs_size, 64)  # 第一层全连接层
        self.actor_fc2 = nn.Linear(64, 64)        # 第二层全连接层
        self.actor_fc3 = nn.Linear(64, action_size)  # 输出层，生成动作
        
        # 评论家网络层定义
        self.critic_fc1 = nn.Linear(obs_size, 64)  # 第一层全连接层
        self.critic_fc2 = nn.Linear(64, 64)        # 第二层全连接层
        self.critic_fc3 = nn.Linear(64, 1)         # 输出层，估计价值

    def forward(self, state):
        """前向传播函数"""
        return self.act(state)
    
    def act(self, state):
        """
        根据状态生成动作
        返回：动作，动作的对数概率，状态价值
        """
        value = self.value(state)  # 获取状态价值
        state = F.relu(self.actor_fc1(state))  # 演员网络第一层
        state = F.relu(self.actor_fc2(state))  # 演员网络第二层
        action_mean = self.actor_fc3(state)    # 动作均值
        action_std = F.softplus(action_mean)   # 动作标准差，使用softplus确保为正
        dist = torch.distributions.Normal(action_mean, action_std)  # 创建正态分布
        action = dist.sample()      # 从分布中采样动作
        log_prob = dist.log_prob(action)  # 计算动作的对数概率
        return action, log_prob, value

    def value(self, state):
        """评论家网络：计算状态价值"""
        state = F.relu(self.critic_fc1(state))  # 评论家网络第一层
        state = F.relu(self.critic_fc2(state))  # 评论家网络第二层
        return self.critic_fc3(state)           # 输出状态价值

    def get_log_prob(self, state, action):
        """计算给定状态和动作对的对数概率"""
        state = F.relu(self.actor_fc1(state))   # 演员网络第一层
        state = F.relu(self.actor_fc2(state))   # 演员网络第二层
        action_mean = self.actor_fc3(state)     # 动作均值
        action_std = F.softplus(action_mean)    # 动作标准差，使用softplus确保为正
        dist = torch.distributions.Normal(action_mean, action_std)  # 创建正态分布
        return dist.log_prob(action)            # 返回动作的对数概率
    
    def entropy(self, state):
        """计算策略的熵，用于鼓励探索"""
        state = F.relu(self.actor_fc1(state))   # 演员网络第一层
        state = F.relu(self.actor_fc2(state))   # 演员网络第二层
        action_mean = self.actor_fc3(state)     # 动作均值
        action_std = F.softplus(action_mean)    # 动作标准差，使用softplus确保为正
        dist = torch.distributions.Normal(action_mean, action_std)  # 创建正态分布
        return dist.entropy()                   # 返回分布的熵