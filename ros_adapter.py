import math
from copy import deepcopy as copy

import numpy as np

import rospy
from sensor_msgs import msg
from xml.dom import minidom as xml

import dataclasses
from typing import Optional

@dataclasses.dataclass
class Joint:
  name: str
  min: float
  max: float
  zero: float
  position: Optional[float] = None
  velocity: Optional[float] = None
  effort: Optional[float] = None
  continuous: bool = False

  def get_effort(self):
    return self._get(self.effort)

  def get_velocity(self):
    return self._get(self.velocity)

  def get_position(self):
    return self._get(self.position)

  def set_position(self, pos: float):
    if self.min <= pos <= self.max:
      self.position = pos
    elif self.min > pos:
      self.position = self.min
    elif self.max < pos:
      self.position = self.max

  def add_to_position(self, delta):
    self.set_position(self._get(self.position) + delta)

  def _get(self, param):
    return param if param is not None else 0.

@dataclasses.dataclass
class DependentJoint:
  name: str
  parent_name: str
  factor: Optional[float] = None
  offset: Optional[float] = None

  def __init__(self, name, mimic_tag):
    self.name = name
    self.parent_name = mimic_tag.getAttribute('joint')
    if mimic_tag.hasAttribute('multiplier'):
      self.factor = float(mimic_tag.getAttribute('multiplier'))
    if mimic_tag.hasAttribute('offset'):
      self.offset = float(mimic_tag.getAttribute('offset'))

class Adapter:
  def __init__(self):
    self.robot = xml.parseString(self._get_param('robot_description')
                                 ).getElementsByTagName('robot')[0]
    self.pub = None

    self.free_joints = {}
    self.joint_list = []
    self.export_joint_list = ['joint_1', 'joint_2', 'joint_5']
    self.dependent_joints = self._get_param('dependent_joints', {})
    self.zeros = self._get_param('zeros', {})

    self.use_mimic = self._get_param('use_mimic_tags', True)
    self.use_small = self._get_param('use_smallest_joint_limits', True)

    self.pub_def_positions = self._get_param('publish_default_positions', True)
    self.pub_def_vels = self._get_param('publish_default_velocities', False)
    self.pub_def_efforts = self._get_param('publish_default_efforts', False)

    for child in self.robot.childNodes:
      self._process_joints_for_child(child)

    self.num_joints = (len(self.free_joints.items()) +
                    len(self.dependent_joints.items()))

    self.default_params = (copy(self.free_joints), copy(self.joint_list), copy(self.dependent_joints))

  def init_publisher(self):
    self.pub = rospy.Publisher('joint_states', msg.JointState, queue_size=5)

  def joint_mins(self):
    return [self.free_joints[j].min for j in self.export_joint_list]

  def joint_maxs(self):
    return [self.free_joints[j].max for j in self.export_joint_list]

  def set_state(self, positions: np.array):
    for i, name in enumerate(self.export_joint_list):
      self.free_joints[name].set_position(positions[i])

  def reset(self):
    self.free_joints, self.joint_list, self.dependent_joints = copy(self.default_params)

  def send_state(self):
    self.pub.publish(self._compose_message())

  def _compose_message(self):
    message = msg.JointState()
    message.header.stamp = rospy.Time.now()

    has_position = len(self.dependent_joints.items()) > 0
    has_velocity = False
    has_effort = False
    for (name, joint) in self.free_joints.items():
      if not has_position and joint.position is not None:
        has_position = True
      if not has_velocity and joint.velocity is not None:
        has_velocity = True
      if not has_effort and joint.effort is not None:
        has_effort = True

    if has_position:
      message.position = self.num_joints * [0.0]
    if has_velocity:
      message.velocity = self.num_joints * [0.0]
    if has_effort:
      message.effort = self.num_joints * [0.0]

    for i, name in enumerate(self.joint_list):
      message.name.append(str(name))
      joint = None
      factor = 1
      offset = 0
      if name in self.free_joints:
        joint = self.free_joints[name]
      elif name in self.dependent_joints:
        dependent = self.dependent_joints[name]
        joint = self.free_joints[dependent.parent]
        factor = dependent.factor
        offset = dependent.offset

      if has_position and joint.position is not None:
        message.position[i] = joint.position * factor + offset
      if has_velocity and joint.velocity is not None:
        message.velocity[i] = joint.velocity * factor
      if has_effort and joint.effort is not None:
        message.effort[i] = joint.effort

      if name == 'joint_6':
        message.position[3] = message.position[2] - message.position[1]

      if name == 'joint_7':
        message.position[4] = message.position[3]

    return message

  def _get_param(self, name, default=None):
    private_name = f'~{name}'
    if rospy.has_param(private_name):
      return rospy.get_param(private_name)
    elif rospy.has_param(name):
      return rospy.get_param(name)
    else:
      return default

  def _process_joints_for_child(self, child):
    if child.nodeType is child.TEXT_NODE or child.localName != 'joint':
      return

    joint_type = child.getAttribute('type')
    if joint_type == 'fixed' or joint_type == 'floating':
      return

    name = child.getAttribute('name')
    if name in self.joint_list:
      return
    self.joint_list.append(name)

    if joint_type == 'continuous':
      minval = -math.pi
      maxval = math.pi
    else:
      try:
        limit = child.getElementsByTagName('limit')[0]
        minval = float(limit.getAttribute('lower'))
        maxval = float(limit.getAttribute('upper'))
      except:
        rospy.logwarn(
          '%s is not fixed, nor continuous, but limits are not specified!' % name)
        return

    safety_tags = child.getElementsByTagName('safety_controller')
    if self.use_small and len(safety_tags) == 1:
      tag = safety_tags[0]
      if tag.hasAttribute('soft_lower_limit'):
        minval = max(minval, float(tag.getAttribute('soft_lower_limit')))
      if tag.hasAttribute('soft_upper_limit'):
        maxval = min(maxval, float(tag.getAttribute('soft_upper_limit')))

    mimic_tags = child.getElementsByTagName('mimic')
    if self.use_mimic and len(mimic_tags) == 1:
      self.dependent_joints[name] = DependentJoint(name, mimic_tags[0])

    if name in self.dependent_joints:
      return

    if name in self.zeros:
      zeroval = self.zeros[name]
    elif minval > 0 or maxval < 0:
      zeroval = (maxval + minval) / 2
    else:
      zeroval = 0

    joint = Joint(
      name=name,
      min=minval,
      max=maxval,
      zero=zeroval,
      position=zeroval if self.pub_def_positions else None,
      velocity=0 if self.pub_def_vels else None,
      effort=0 if self.pub_def_efforts else None,
      continuous=joint_type == 'continuous'
    )
    self.free_joints[name] = joint

