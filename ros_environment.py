import math

import gym
import numpy as np
import rospy
from numpy.linalg import norm

from ros_adapter import Adapter
from ros_objects import Robot

from typing import List, Tuple


class WindowStump:
  def check_close_event(self):
    return not rospy.is_shutdown()

  def render_clear(self):
    pass

  def render_stats(self, *args):
    pass

class RosEnvironment(gym.Env):
  ACTION_COOLDOWN = 30
  MAX_TIME = 15 * 6000
  MIN_REWARD = -1000
  MAX_REWARD = 500

  GOAL = np.array([100, 50, 50])

  def __init__(self, render: bool = False, adapter: Adapter = None):
    super().__init__()

    self._render = render
    if adapter is None:
      self.ros_adapter = Adapter()
    else:
      self.ros_adapter = adapter

    if self._render:
      self.ros_adapter.init_publisher()
      self.window = WindowStump()

    self.reset()

  def reset(self) -> np.ndarray:
    self.total_reward = 0
    self.cooldown_counter = 0
    self.prev_action = None
    self.timer = 0

    self.ros_adapter.reset()
    if self._render:
      self.ros_adapter.send_state()
    self.robot = Robot(self.ros_adapter.joint_mins(), self.ros_adapter.joint_maxs())

    self.prev_pos = self.robot.tip_pos()
    self.n_joints = self.robot.n_joints
    self.N_ACTIONS = 3 ** self.n_joints
    return self.robot.get_positions()

  def render(self):
    if self._render:
      self.ros_adapter.set_state(self.robot.get_positions()[0])
      self.ros_adapter.send_state()

  def _calculate_reward(self) -> Tuple[int, bool]:
    cur_pos = self.robot.tip_pos()
    done = self.timer >= self.MAX_TIME

    prev_dist = norm(self.GOAL - self.prev_pos)
    cur_dist = norm(self.GOAL - cur_pos)
    self.prev_pos = cur_pos

    if math.isclose(cur_dist, 0):
      done = True

    return prev_dist - cur_dist, done

  def step(self, action: List[int]) -> Tuple[np.ndarray, int, bool]:
    self.timer += 1

    self.robot.apply_action(np.radians(np.array(action) / 10))
    return self.robot.get_positions(), *self._calculate_reward()

  def step_scalar_action(self, action: int) -> Tuple[np.ndarray, int, bool]:
    action_list = []
    for i in range(self.n_joints):
      action_list += [(action % 3) - 1]
      action //= 3
    return self.step(action_list)

  def step_buffer(self, agent, eps=1) -> Tuple[np.ndarray, int, bool]:
    if self.cooldown_counter == 0:
      self.prev_action = agent.action(eps)
    self.cooldown_counter = (self.cooldown_counter + 1) % self.ACTION_COOLDOWN
    return self.step_scalar_action(self.prev_action)
