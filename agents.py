import numpy as np


class Agent:
  def __init__(self, n_actions, n_states):
    self.n_actions = n_actions
    self.n_states = n_states
    
  def action(self):
    return np.random.randint(self.n_actions)
    
  def reevaluate(self, info, reward, done):
    pass


class CrossEntropyAgent(Agent):
  def __init__(self, n_actions, n_states):
    super().__init__(n_actions, n_states)
    self.policy = ...
    
  def action(self):
    return ...
    
  def reevaluate(self, info, reward, done):
    ...
