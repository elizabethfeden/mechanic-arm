"""Learning agents and algorithms for fitting them."""

import multiprocessing as mp
import numpy as np
from typing import Any, List, Optional, Tuple

from classifiers import MLPClassifier
from environment import Environment


class Agent:
  """Default agent. Chooses random action based on probabilities in `policy` (uniform, by default)."""
  def __init__(self, n_actions: int, policy: Optional[np.ndarray] = None):
    self.n_actions = n_actions
    self.policy = policy or np.ones((n_actions,)) / n_actions
    
  def action(self) -> int:
    return np.random.choice(self.n_actions, p=self.policy)
    
  def reevaluate(self, info: Any):
    pass


class Fitter:
  """Default fitter. This class may be considered abstract."""
  def __init__(self):
    self.mean_rewards = []
    self.median_rewards = []
    self.best_reward = 0

  def fit_epoch(self, verbose: Tuple[str] = (), n_jobs: int = 2) -> Agent:
    pass


class CrossEntropyAgent(Agent):
  """Agent for cross entropy learning algorithms.

  Chooses actions based on probabilities returned by `predict_proba` method
  of `policy`. Also records all the states and actions chosen.
  """
  def __init__(self, n_actions: int, policy: Any, state: np.ndarray):
    super().__init__(n_actions)
    self.policy = policy

    self.state = None
    self.history_actions = []
    self.history_states = []

    self.reevaluate(state)
    
  def action(self) -> int:
    proba = self.policy.predict_proba(self.state)[0]
    action = np.random.choice(self.n_actions, p=proba)
    self.history_actions += [action]
    self.history_states += [self.state[0]]
    return action
    
  def reevaluate(self, info: np.ndarray):
    self.state = info


class CrossEntropyFitter(Fitter):
  def __init__(self, n_sessions: int = 50, n_elites: int = 10,
               policy_hidden_layers: Tuple[int] = (5,5,5),
               policy_random_state: int = 16):
    super().__init__()
    self.n_sessions = n_sessions
    self.n_elites = n_elites
    
    self.seed_coef = 39
    np.random.seed(self.seed_coef - 2)

    self._policy = MLPClassifier(hidden_layer_sizes=policy_hidden_layers,
                                 random_state=policy_random_state)
    self._pseudo_fit()  # Needed when using partial_fit
    
  def _pseudo_fit(self):
    env = Environment()
    self.best_reward = env.MIN_REWARD + 1

    state = env.reset()[0]
    X = np.array([state] * env.N_ACTIONS)
    y = np.arange(env.N_ACTIONS)
    self._policy.partial_fit(X, y, list(range(env.N_ACTIONS)))

  def _simulate_session(self, seed: int = 42) -> Tuple[CrossEntropyAgent, int]:
    np.random.seed(seed)
    env = Environment()
    agent = CrossEntropyAgent(env.N_ACTIONS, self._policy, env.reset())
    total_reward = 0
    done = False
    while not done:
      state, reward, done = env.step_buffer(agent)
      agent.reevaluate(state)
      total_reward += reward
    return agent, total_reward
    
  def _refit(self, elites: List[CrossEntropyAgent]):
    X = np.array(np.vstack([elite.history_states for elite in elites]))
    y = np.array(sum([elite.history_actions for elite in elites], []))
    self._policy.partial_fit(X, y)
    
  def fit_epoch(self, verbose: Tuple[str] = (),
                n_jobs: int = 4) -> CrossEntropyAgent:
    """Simulates `n_sessions` and fits the classifier based on simulation results.

    Args:
      verbose is a tuple of possible options, which are:
        'history' - print history of actions in each agent's simulated session;
        'rewards' - print rewards for all the sessions in the epoch,
            as well as their mean and median;
        'coefs' - print classifier's coefficients after every refitting.

      n_jobs is how many processes to use to simulate sessions.

    Returns: the best performer of the epoch.
    """
    self.seed_coef += 13
    
    pool = mp.Pool(n_jobs)
    async_results = []
    for i in range(self.n_sessions):
      # Guarantee different random seeds for subprocesses
      async_results += [pool.apply_async(self._simulate_session,
                                         args=(i * self.seed_coef + 54,))]
    pool.close()
    pool.join()

    agents = []
    rewards = []
    for result in async_results:
      agent, reward = result.get()
      agents += [agent]
      rewards += [reward]

      if 'history' in verbose:
        print('=== history ===')
        print(reward, agent.history_actions)

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
        print(self._policy.coefs_)

    return elites[-1]
    
