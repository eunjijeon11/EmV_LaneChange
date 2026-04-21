from .base import BaseAgent

class DummyAgent(BaseAgent):
    def __init__(self, action_space):
        self.action_space = action_space
        self.policy = None

    def predict(self, obs):
        actions = [0] * len(obs)
        return actions
    
    def update(self, obs, actions, reward, next_obs, done):
        pass
    
    def save_policy(self, save_dir):
        pass
    
    def load_policy(self, policy_dir):
        pass