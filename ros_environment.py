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
  ACTION_COOLDOWN = 10
  MAX_TIME = 12_000
  MIN_REWARD = -1000
  MAX_REWARD = 500
  DECAY_DELTA = 0.00001

  GOAL = np.array([130, 100, 130])

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
    self.decay = 1 + self.DECAY_DELTA

    # self.GOAL = np.array([np.random.randint(-100, 100), 100, 130])

    self.ros_adapter.reset()
    self.robot = Robot(self.ros_adapter.joint_mins(), self.ros_adapter.joint_maxs())
    if self._render:
      self.ros_adapter.send_state()
      self.ros_adapter.send_marker_state(1, np.dot(
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        self.GOAL))

    self.prev_pos = self.robot.tip_pos()

    self.MAX_REWARD = norm(self.GOAL - self.prev_pos)

    self.n_joints = self.robot.n_joints
    self.N_ACTIONS = 3 ** self.n_joints
    self.prev_action_list = [0] * self.n_joints
    return self._transform_state(self.prev_action_list)

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

    return self.decay * (prev_dist - cur_dist) / self.MAX_REWARD, done

  def _transform_state(self, prev_action_list):
    pos = self.robot.get_positions()[0]
    state = [math.sin(x) for x in pos] + [math.cos(x) for x in pos]
    state += prev_action_list
    state += (self.GOAL / norm(self.GOAL)).tolist()
    tip_pos = self.robot.tip_pos()
    state += (tip_pos / norm(tip_pos)).tolist()
    return np.array(state).reshape(-1, 1)

  def step(self, action: List[int]) -> Tuple[np.ndarray, int, bool]:
    self.timer += 1
    temp = self.prev_action_list
    self.prev_action_list = action

    self.robot.apply_action(np.radians(np.array(action) / 10))
    return self._transform_state(temp), *self._calculate_reward()

  def step_scalar_action(self, action: int) -> Tuple[np.ndarray, int, bool]:
    action_list = []
    for i in range(self.n_joints):
      action_list += [(action % 3) - 1]
      action //= 3
    return self.step(action_list)

  def step_buffer(self, agent, eps=1) -> Tuple[np.ndarray, int, bool]:
    if self.cooldown_counter == 0:
      self.prev_action = agent.action(eps)
      self.decay -= self.DECAY_DELTA
    self.cooldown_counter = (self.cooldown_counter + 1) % self.ACTION_COOLDOWN
    return self.step_scalar_action(self.prev_action)
