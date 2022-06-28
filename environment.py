import enum
import gym
import numpy as np

from typing import List, Tuple

import objects
import physics
import pygame_window


class Action(enum.IntEnum):
  NOTHING = 0,
  FIX = 1,
  LEFT = 2,
  RIGHT = 3,


class Environment(gym.Env):
  N_ACTIONS = 16
  MAX_REWARD = 50
  MIN_REWARD = 0

  def __init__(self, pygame_render: bool = False):
    super().__init__()
    self.reset()
    
    self._pygame_render = pygame_render
    if pygame_render:
      self.window = pygame_window.PygameWindow()
      self._pymunk_options = physics.get_print_options(self.window.screen)
    else:
      self._pymunk_options = physics.get_print_options()
    
  def reset(self) -> np.ndarray:
    self._objects = objects.Objects()
    self._physics = physics.Simulation(self._objects)
    self.total_reward = 0
    return self._objects.get_info()
    
  def render(self):
    self._physics.render(self._pymunk_options)
    if self._pygame_render:
      self.window.render_text('reward: ' + str(self.total_reward), (0, 0))
      self.window.finish_render()

  def _calculate_reward(self) -> Tuple[int, bool]:
    box_state = self._objects.box_state()
    done = box_state == objects.BoxState.OUT_OF_BOUNDS
    reward = 1 if box_state == objects.BoxState.TOUCHES_FLOOR else 0
    self.total_reward += reward
    if self.total_reward >= self.MAX_REWARD:
      self.total_reward = self.MAX_REWARD
      done = True
    return reward, done
    
  def step(self, action: List[int]) -> Tuple[np.ndarray, int, bool]:
    for index, cur_action in zip([0, 2], action):
      if cur_action == Action.FIX:
        if index == 0:
          self._objects.arm.fix_velocity([index, index + 1])
        else:
          self._objects.arm.fix_angle([index, index + 1])
      elif cur_action == Action.LEFT:
        self._objects.arm.apply_force_to_circle(index, -1)
      elif cur_action == Action.RIGHT:
        self._objects.arm.apply_force_to_circle(index, 1)
      elif cur_action == Action.NOTHING:
        pass
      else:
        raise Exception('Invalid action')

    self._physics.step()
    return self._objects.get_info(), *self._calculate_reward()
    
  def step_scalar_action(self, action: int) -> Tuple[np.ndarray, int, bool]:
    return self.step([action // 4, action % 4])

