from .base import BaseAgent
import numpy as np
import pickle

class SarsaAgent(BaseAgent):
    def __init__(self, action_space, alpha=0.1, gamma=0.95, epsilon=1.0):
        self.n_actions = action_space.n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.qTable = {}
        self.next_actions = None  # 다음 스텝 행동 저장

    def discretize(self, o):
        lane = int(np.clip(o[0], 0, 2))
        speed = int(np.clip(o[1], 0, 19) / 4) 
        gap = int(np.clip(o[2], 0, 99) / 20)
        emv_near = 0 if o[7] < 0 else 1
        emv_lane = int(np.clip(o[7], 0, 2)) if emv_near else 0

        if emv_near:
            rel_dist = int(np.clip(o[5], -80, 80) / 32)  # -2~2 (5구간)
            rel_dist = rel_dist + 2  # 0~4로 변환
        else:
            rel_dist = 0
        
        return (lane, speed, gap, emv_near, emv_lane, rel_dist)
    
    def getQ(self, key):
        if key not in self.qTable:
            self.qTable[key] = [0.0] * self.n_actions 
        
        return self.qTable[key]

    def predict(self, obs):
        actions = []
        for o in obs:
            if np.random.random() < self.epsilon:
                actions.append(np.random.randint(self.n_actions))
            else:
                key = self.discretize(o)
                actions.append(int(np.argmax(self.getQ(key))))
        
        return actions
    
    def update(self, obs, actions, reward, next_obs, done):
        if len(obs) == 0 or len(next_obs) == 0: 
            return

        # SARSA 핵심: 업데이트 전에 다음 행동을 미리 결정
        # Q-Learning은 이 줄이 없고 max()를 씀
        # current_next_actions[i]가 실제로 다음 스텝에서 선택될 행동
        if self.next_actions is None or len(self.next_actions) != len(obs):
            current_next_actions = self.predict(next_obs)
        else:
            current_next_actions = self.next_actions

        rewards = reward if isinstance(reward, list) else [float(reward)] * len(obs)

        for i, (o, a) in enumerate(zip(obs, actions)):
            if i >= len(next_obs):
                break

            s  = self.discretize(o)
            s_ = self.discretize(next_obs[i])
            q  = self.getQ(s)
            q_ = self.getQ(s_)

            r = rewards[i]
        
            if done:
                target = r
            else:
                a_ = current_next_actions[i]
                target = r + self.gamma * q_[a_] # 실제 선택될 행동의 Q값

            q[a] += self.alpha * (target - q[a])

        self.next_actions = self.predict(next_obs)

        if done:
            self.next_actions = None

    def save_policy(self, save_dir):
        if not save_dir:
            save_dir = "sarsa_policy.pkl"
        with open(save_dir, 'wb') as f:
            pickle.dump({'qTable': self.qTable, 'epsilon': self.epsilon}, f)
        print(f"[SarsaAgent] 저장: {save_dir} | 상태 수: {len(self.qTable)}")
    
    def load_policy(self, policy_dir):
        if not policy_dir:
            policy_dir = "sarsa_policy.pkl"
        with open(policy_dir, 'rb') as f:
            data = pickle.load(f)
        self.qTable = data['qTable']
        self.epsilon = data.get('epsilon', self.epsilon_min)
        print(f"[SarsaAgent] 로드: {policy_dir} | 상태 수: {len(self.qTable)}")