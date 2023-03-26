"""Learning's environment. Responds to agent's actions by changing its state."""

import collections
import enum
import gym
import math
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


class RewardParams:
  class Type(enum.Enum):
    NONE = -1
    COMPLETE = 0  # box touches floor -- end goal
    CONTROL = 1   # box touches arm -- motivation for control

  @staticmethod
  def box_state_to_reward(box_state: objects.BoxState) -> Type:
    if (box_state & objects.BoxState.TOUCHES_FLOOR) != 0:
      return RewardParams.Type.COMPLETE
    if (box_state & objects.BoxState.HIGHER) != 0:
      return RewardParams.Type.CONTROL
    return RewardParams.Type.NONE

  coefs = {
    Type.COMPLETE: 1.,
    Type.CONTROL: 0.2,
    Type.NONE: 0.,
  }

  maxs = {
    Type.COMPLETE: 70,
    Type.CONTROL: 15,
    Type.NONE: 0,
  }


class Environment(gym.Env):
  N_ACTIONS = 16
  MAX_REWARD = RewardParams.maxs[RewardParams.Type.COMPLETE]
  MIN_REWARD = 0
  ACTION_COOLDOWN = 10
  MAX_TIME = 15 * 60

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
    self.cooldown_counter = 0
    self.prev_action = None
    self.timer = 0
    return self._objects.get_info()
    
  def render(self):
    self._physics.render(self._pymunk_options)
    if self._pygame_render:
      self.window.render_text('reward: {:.1f}'.format(self.total_reward), (0, 0))
      self.window.finish_render()

  def _calculate_reward(self) -> Tuple[int, bool]:
    box_state = self._objects.box_state()
    done = box_state == objects.BoxState.OUT_OF_BOUNDS or self.timer >= self.MAX_TIME
    reward_type = RewardParams.box_state_to_reward(box_state)
    reward = RewardParams.coefs[reward_type]
    prev = self.total_reward
    self.total_reward += reward

    if (self.total_reward > RewardParams.maxs[reward_type]
        or math.isclose(self.total_reward, RewardParams.maxs[reward_type])):
      self.total_reward = max(self.total_reward - reward, RewardParams.maxs[reward_type])
      reward = self.total_reward - prev
    if self.total_reward >= self.MAX_REWARD:
      done = True

    return reward, done
    
  def step(self, action: List[int]) -> Tuple[np.ndarray, int, bool]:
    """Responds to the action, which is a number of Actions applied to each motor."""
    self.timer += 1
    
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
    
  def step_buffer(self, agent) -> Tuple[np.ndarray, int, bool]:
    if self.cooldown_counter == 0:
      self.prev_action = agent.action()
    self.cooldown_counter = (self.cooldown_counter + 1) % self.ACTION_COOLDOWN
    return self.step_scalar_action(self.prev_action)

