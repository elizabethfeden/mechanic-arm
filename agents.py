import numpy as np
from sklearn.neural_network import MLPClassifier as Classifier


class Agent:
  def __init__(self, n_actions):
    self.n_actions = n_actions
    
  def action(self):
    return np.random.randint(self.n_actions)
    
  def reevaluate(self, info):
    pass


class CrossEntropyAgent(Agent):
  def __init__(self, n_actions, policy, state):
    super().__init__(n_actions)
    self.policy = policy
    self.reevaluate(state)
    self.history_actions = []
    self.history_states = []
    
  def action(self):
    proba = self.policy.predict_proba(self.state)[0]
    action = np.random.choice(self.n_actions, p=proba)
    #print(proba)
    self.history_actions += [action]
    self.history_states += [self.state[0]]
    return action
    
  def reevaluate(self, info):
    self.state = info

 
class CrossEntropyFitter:
  def __init__(self, env, n_sessions=100, n_elites=10):
    self.env = env
    self.n_sessions = n_sessions
    self.n_elites = n_elites
    self.policy = Classifier()
    self.best_reward = env.MIN_REWARD + 1
    self._pseudo_fit()
    
  def _pseudo_fit(self):
    state = self.env.reset()[0]
    X = np.array([state] * self.env.N_ACTIONS)
    y = np.arange(self.env.N_ACTIONS)
    self.policy.fit(X, y)
    for _ in range(4):
      self.fit_epoch(random=True)
    
  def _simulate_session(self, random):
    if random:
      self.env.reset()
      agent = Agent(self.env.N_ACTIONS)
    else:
      agent = CrossEntropyAgent(self.env.N_ACTIONS, self.policy, self.env.reset())
    done = False
    total_reward = 0
    while not done:
      state, reward, done, _ = self.env.step_scalar_action(agent.action())
      agent.reevaluate(state)
      total_reward += reward
    return agent, total_reward
    
  def _refit(self, elites):
    X = np.array(np.vstack([elite.history_states for elite in elites]))
    y = np.array(sum([elite.history_actions for elite in elites], []))
    self.policy.partial_fit(X, y)
    
  def fit_epoch(self, random=False, verbose=False):
    agents = []
    rewards = []
    for _ in range(self.n_sessions):
      new_agent, reward = self._simulate_session(random)
      agents += [new_agent]
      rewards += [reward]

      if verbose:
        print(reward, new_agent.history_actions)

    indices = np.argsort(np.array(rewards))
    elites = [agents[i] for i in indices][-self.n_elites:]
    if rewards[indices[0]] >= self.best_reward:
      print('here', rewards[indices[0]], self.best_reward)
      self.best_reward = rewards[indices[0]]
      self._refit(elites)
    #print('coefs', self.policy.coefs_)
        
    return elites[-1]
    
