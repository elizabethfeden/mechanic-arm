"""Unites fitting and representation, with or without multiprocessing."""

import enum
import multiprocessing as mp
import numpy as np
import pickle

from typing import Optional, Tuple

import agents
from environment import Environment


def _run_fitter(fitter: agents.Fitter,
                best_pipe_in: mp.connection.Connection,
                results_pipe_in: mp.connection.Connection,
                running_pipe_out: mp.connection.Connection):
  while running_pipe_out.recv():
    best = fitter.fit_epoch(verbose=('rewards',))
    best_pipe_in.send(best)

  results_pipe_in.send((fitter.mean_rewards, fitter.median_rewards))
  best_pipe_in.close()
  running_pipe_out.close()
  results_pipe_in.close()


class SaveOption(enum.Enum):
  MODEL = 0,
  MOVING_AVERAGES = 1,
  MEAN_REWARDS = 2,
  MEDIAN_REWARDS = 3,


class Simulation:
  """Manages the interactions between fitters and presentation.

  Attributes:
    `agent` or `fitter`: correspondent objects of `agents.py` entities. Exactly
        one of these two must be not None.
    `parallel`: should multiprocessing be used (one process for fitting, another
        for presentation). Will be ignored if `agent` is not None.
    `save_options`: which of the results should be saved to file.
    `moving_average_num`: specified for the moving average metric.
    `env`: presentation Environment.
   Will be available after run() execution:
    `averages`: rewards moving averages after each epoch.
    `mean_rewards` and `median_rewards`: corresponding metrics for each epoch
      of cross entropy fitting.

  """
  def __init__(self,
               agent: Optional[agents.Agent] = None,
               fitter: Optional[agents.Fitter] = None,
               parallel: bool = False,
               save_options: Tuple[SaveOption, ...] = (SaveOption.MOVING_AVERAGES,),
               moving_average_num: int = 20):
    self.agent = agent
    self.fitter = fitter
    self.parallel = parallel and (agent is None)
    self.save_options = save_options
    self.moving_average_num = moving_average_num

    self.env = Environment(pygame_render=True)

    self.averages, self.mean_rewards, self.median_rewards = [], [], []

    if self.parallel:
      self._best_pipe_out, self._best_pipe_in = mp.Pipe()
      self._results_pipe_out, self._results_pipe_in = mp.Pipe()
      self._running_pipe_out, self._running_pipe_in = mp.Pipe()

  def _start_fitter_process(self):
    mp.Process(
      target=_run_fitter,
      args=(self.fitter, self._best_pipe_in,
            self._results_pipe_in, self._running_pipe_out)
    ).start()

  def _get_agent(self):
    if self.parallel:
      self.agent = self._best_pipe_out.recv()
    return self.agent

  def _simulate_one_game(self, total, wins):
    agent = self._get_agent()
    agent.reevaluate(self.env.reset())

    running = True
    done = False
    total_reward = 0
    while not done:
      running = running and self.env.window.check_close_event()

      info, reward, done = self.env.step_scalar_action(agent.action())
      total_reward += reward
      agent.reevaluate(info)

      self.env.window.render_clear()
      self.env.window.render_stats(total, wins, running)
      self.env.render()

    return running, total_reward

  def run(self):
    if self.parallel:
      self._start_fitter_process()
      self._running_pipe_in.send(True)

    wins, total = 0, 0
    rewards = np.array([0] * self.moving_average_num)
    current_index = 0
    running = True
    while running:
      if not self.parallel and self.fitter is not None:
        self.agent = self.fitter.fit_epoch(verbose=('rewards',))
      elif self.parallel:
        self._running_pipe_in.send(running)
      running, total_reward = self._simulate_one_game(total, wins)
      if self.parallel and not running:
        self._running_pipe_in.send(False)

      total += 1
      if total_reward == self.env.MAX_REWARD:
        wins += 1
      rewards[current_index] = total_reward
      current_index = (current_index + 1) % self.moving_average_num
      self.averages += [rewards.mean()]

    if self.parallel:
      self.mean_rewards, self.median_rewards = self._results_pipe_out.recv()
    elif self.fitter is not None:
      self.mean_rewards = self.fitter.mean_rewards
      self.median_rewards = self.fitter.median_rewards

  def _save_results(self):
    if not self.save_options:
      return

    if SaveOption.MODEL in self.save_options:
      with open('model', 'wb') as file:
        pickle.dump(self.agent.policy, file)
      if len(self.save_options) == 1:  # Only model should be saved
        return

    with open('results.txt', 'w') as file:
      option_values = {
        SaveOption.MEAN_REWARDS: self.mean_rewards,
        SaveOption.MEDIAN_REWARDS: self.median_rewards,
        SaveOption.MOVING_AVERAGES: self.averages,
      }
      for option in self.save_options:
        if option in option_values:
          file.write(str(option_values[option]) + '\n')

  def __del__(self):
    if self.parallel:
      self._best_pipe_out.close()
      self._running_pipe_in.close()
      self._results_pipe_out.close()

    self._save_results()
