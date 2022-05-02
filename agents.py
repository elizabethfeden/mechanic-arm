import numpy as np
from typing import Any, List, Tuple

from classifiers import MLPClassifier
import environment


class Agent:
  def __init__(self, n_actions: int):
    self.n_actions = n_actions
    
  def action(self) -> int:
    return np.random.randint(self.n_actions)
    
  def reevaluate(self, info: Any):
    pass


class CrossEntropyAgent(Agent):
  def __init__(self, n_actions: int, policy: Any, state: np.ndarray):
    super().__init__(n_actions)
    self.policy = policy
    self.reevaluate(state)
    self.history_actions = []
    self.history_states = []
    
  def action(self) -> int:
    proba = self.policy.predict_proba(self.state)[0]
    action = np.random.choice(self.n_actions, p=proba)
    self.history_actions += [action]
    self.history_states += [self.state[0]]
    return action
    
  def reevaluate(self, info: np.ndarray):
    self.state = info

 
class CrossEntropyFitter:
  def __init__(self, env: environment.Environment,
               n_sessions: int = 50, n_elites: int = 10):
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
    
  def _simulate_session(self) -> Tuple[CrossEntropyAgent, int]:
    agent = CrossEntropyAgent(self.env.N_ACTIONS, self.policy, self.env.reset())
    total_reward = 0
    done = False
    while not done:
      state, reward, done = self.env.step_scalar_action(agent.action())
      agent.reevaluate(state)
      total_reward += reward
    return agent, total_reward
    
  def _refit(self, elites: List[CrossEntropyAgent]):
    X = np.array(np.vstack([elite.history_states for elite in elites]))
    y = np.array(sum([elite.history_actions for elite in elites], []))
    self.policy.partial_fit(X, y)
    
  def fit_epoch(self, verbose: Tuple[str] = ()):
    """
    Args:
      verbose is a tuple of possible options, which are:
        'history' - print history of actions in each agent's simulated session;
        'rewards' - print rewards for all the sessions in the epoch,
            as well as their mean and median;
        'coefs' - print classifier's coefficients after every refitting.

    Returns: the best performer of the epoch.
    """
    agents = []
    rewards = []
    for _ in range(self.n_sessions):
      new_agent, reward = self._simulate_session()
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
    
