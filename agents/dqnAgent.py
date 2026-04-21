import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from .base import BaseAgent

class QNetwork(nn.Module):
    def __init__(self, state_size=9, action_size=5):  
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
 
    def forward(self, x):
        return self.fc(x)

class DQNAgent(BaseAgent):
    def __init__(self, action_space):
        
        self.action_space = action_space
        self.state_size = 9  
        self.action_size = 5  # [KEEP, LEFT, RIGHT, ACCEL, DECEL]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
        
        self.policy = QNetwork(self.state_size, self.action_size).to(self.device)
        self.target_policy = QNetwork(self.state_size, self.action_size).to(self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())    
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 64

    def predict(self, obs):
        """
        obs: 현재 도로 위 일반 차량들의 observation 리스트
        return: 각 차량에 대한 action 리스트
        """
        actions = []
        for o in obs:
            if np.random.rand() <= self.epsilon:
                actions.append(random.randrange(self.action_size))
            else:
                state = torch.FloatTensor(o).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.policy(state)
                actions.append(torch.argmax(q_values).item())
        return actions

    def update(self, obs, actions, rewards, next_obs, dones):
        # 1. rewards가 단일 숫자일 경우를 위한 예외 처리
        if not isinstance(rewards, (list, np.ndarray)):
            rewards = [rewards] * len(obs)
            
        if not isinstance(dones, (list, np.ndarray)):
            dones_list = [dones] * len(obs)
        else:
            dones_list = dones
            
        # 2. 경험 저장
        for o, a, r, no in zip(obs, actions, rewards, next_obs):
            self.memory.append((o, a, r, no, dones))

        # 3. 충분한 데이터가 쌓이면 학습
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        s_lst, a_lst, r_lst, ns_lst, d_lst = zip(*batch)

        s = torch.FloatTensor(np.array(s_lst)).to(self.device)
        a = torch.LongTensor(a_lst).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r_lst).to(self.device)
        ns = torch.FloatTensor(np.array(ns_lst)).to(self.device)
        d = torch.FloatTensor(d_lst).to(self.device)

        # Q-Learning 업데이트
        curr_q = self.policy(s).gather(1, a)
        next_q = self.target_policy(ns).max(1)[0].detach()
        target_q = r + (1 - d) * self.gamma * next_q

        loss = nn.MSELoss()(curr_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_policy(self, save_dir):
        torch.save(self.policy.state_dict(), save_dir)
        print(f"Policy saved to {save_dir}")

    def load_policy(self, policy_dir):
        self.policy.load_state_dict(torch.load(policy_dir, map_location=self.device))
        self.target_policy.load_state_dict(self.policy.state_dict())
        print(f"Policy loaded from {policy_dir}")
        
    def episode_done(self):
        # Target Network 동기화 (에피소드 끝날 때 호출)
        self.target_policy.load_state_dict(self.policy.state_dict())