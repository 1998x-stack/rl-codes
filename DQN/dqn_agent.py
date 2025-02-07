# dqn_agent.py
import torch
import torch.optim as optim
import numpy as np
from model import DQN
from replay_buffer import ReplayBuffer
from config import GAMMA, LEARNING_RATE, EPSILON_START, EPSILON_MIN, EPSILON_DECAY, BATCH_SIZE, TARGET_UPDATE_FREQ
from loguru import logger  # 导入loguru用于日志记录

class DQNAgent:
    """
    DQN智能体，负责训练、决策和策略更新
    """

    def __init__(self, env):
        """
        初始化DQN智能体
        Args:
            env (gym.Env): Gym环境
        """
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.model = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_model = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())  # 复制参数
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer()
        self.epsilon = EPSILON_START
        self.step_counter = 0

    def epsilon_greedy(self, state: np.ndarray) -> int:
        """
        epsilon-贪婪策略选择动作
        Args:
            state (np.ndarray): 当前状态
        Returns:
            int: 选择的动作
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)  # 探索
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()  # 利用

    def train(self):
        """
        使用回放缓冲区的经验进行训练
        """
        if self.replay_buffer.size() < BATCH_SIZE:
            return  # 不足以训练

        # 随机采样一个批次
        batch = self.replay_buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 计算目标Q值
        next_q_values = self.target_model(next_states)
        max_next_q_values = next_q_values.max(1)[0]
        target_q_values = rewards + (GAMMA * max_next_q_values * (1 - dones))

        # 计算当前Q值
        q_values = self.model(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算损失
        loss = torch.nn.functional.mse_loss(q_value, target_q_values)

        # 更新模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新epsilon
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

        # 每隔一定步数更新目标网络
        self.step_counter += 1
        if self.step_counter % TARGET_UPDATE_FREQ == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            logger.info(f"Target network updated at step {self.step_counter}")

        logger.debug(f"Training loss: {loss.item()}")  # 打印训练损失