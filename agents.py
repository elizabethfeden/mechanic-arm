import numpy as np
from classifiers import MLPClassifier


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
  def __init__(self, env, n_sessions=50, n_elites=10):
    self.env = env
    self.n_sessions = n_sessions
    self.n_elites = n_elites
    self.policy = MLPClassifier(hidden_layer_sizes=(25,), random_state=16)
    self.best_reward = env.MIN_REWARD + 1
    self.mean_rewards = []
    self.median_rewards = []
    self._pseudo_fit()
    
  def _pseudo_fit(self):
    state = self.env.reset()[0]
    X = np.array([state] * self.env.N_ACTIONS)
    y = np.arange(self.env.N_ACTIONS)
    self.policy.partial_fit(X, y, list(range(self.env.N_ACTIONS)))
    
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
    
  def fit_epoch(self, random=False, verbose=()):
    agents = []
    rewards = []
    for _ in range(self.n_sessions):
      new_agent, reward = self._simulate_session(random)
      agents += [new_agent]
      rewards += [reward]

      if 'history' in verbose:
        print('=== history ===')
        print(reward, new_agent.history_actions)

    rewards = np.array(rewards)
    indices = np.argsort(rewards)
    elites = [agents[i] for i in indices][-self.n_elites:]

    mean, median = rewards.mean(), np.median(rewards)
    self.mean_rewards += [mean]
    self.median_rewards += [median]
    if 'rewards' in verbose:
      print('=== rewards ===')
      print(mean, median, rewards[indices])

    if rewards[indices[-1]] >= self.best_reward:
      self.best_reward = rewards[indices[-1]]
      self._refit(elites)
      if 'coefs' in verbose:
        print('=== coefs ===')
        print(self.policy.coefs_)

    return elites[-1]
    
