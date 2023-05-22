import math

import numpy as np


class Robot:
  def __init__(self, joint_mins, joint_maxs):
    # robot config
    self.n_joints = 3
    self.l1, self.l2, self.l3 = 138, 135, 147

    self.joint_mins = np.array(joint_mins)
    self.joint_maxs = np.array(joint_maxs)

    self.reset()

  def reset(self):
    # self.positions = (self.joint_mins + self.joint_maxs) / 2
    self.positions = np.zeros((self.n_joints,))
    self.apply_action(0)  # clamp

  def get_positions(self):
    return self.positions.reshape((1, -1))

  def apply_action(self, action: np.array):
    self.positions += action
    np.clip(self.positions, self.joint_mins, self.joint_maxs, out=self.positions)

  def tip_pos(self):
    s1, s2, s3 = [math.sin(x) for x in self.positions]
    c1, c2, c3 = [math.cos(x) for x in self.positions]
    c3 = -c3

    '''
    full matrix (we only use right column because base is always 0)
    fw_kinematic_mat = np.array([
      [
        c1 * c2 * c3 + c1 * c2 * s3,
        c1 * c2 * c3 - c1 * s2 * s3,
        -s1,
        l2 * c1 * s2 + l3 * c1 * (c2 * s3 + s2 * c3)
      ],
      [
        s1 * s2 * c3 + s1 * c2 * s3,
        s1 * c2 * c3 - s1 * s2 * s3,
        c1,
        l2 * s1 * s2 + l3 * s1 * (s2 * c3 + c2 * s3)
      ],
      [
        c2 * c3 - s2 * s3,
        - c2 * s3 + s2 * c3,
        0,
        l1 + l3 * (c2 * c3 - s2 * s3) + l2 * c2
      ],
      [0, 0, 0, 1]
    ])
    '''

    return np.array([
      self.l2 * c1 * s2 + self.l3 * c1 * (c2 * s3 + s2 * c3),
      self.l2 * s1 * s2 + self.l3 * s1 * (s2 * c3 + c2 * s3),
      self.l1 + self.l3 * (c2 * c3 - s2 * s3) + self.l2 * c2,
    ])

