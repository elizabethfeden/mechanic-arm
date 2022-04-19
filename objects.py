"""Package for simulation objects storage management"""

import enum
import numpy as np
import pymunk
from typing import Callable, Optional, Tuple


def static_body() -> pymunk.Body:
  return pymunk.Body(body_type=pymunk.Body.STATIC)


def _create_shape(
    position: Tuple[float, float],
    shape_constructor: Callable[[pymunk.Body], pymunk.Shape],
    mass: Optional[float],
    is_static: bool,
) -> pymunk.Shape:
  if is_static:
    body = static_body()
  else:
    body = pymunk.Body()
  
  body.position = position
  shape = shape_constructor(body)
  shape.mass = mass
  shape.elasticity = 0.999
  shape.friction = 1
  return shape


def create_rect(
    position: Tuple[float, float], size: Tuple[float, float],
    mass: float = 10, is_static: bool = False,
) -> pymunk.Poly:
  shape = _create_shape(position,
                        lambda body: pymunk.Poly.create_box(body, size),
                        mass, is_static)
  shape.properties = {}
  shape.properties['type'] = 1
  shape.properties['width'], shape.properties['height'] = size
  return shape
    
    
def create_circle(
    position: Tuple[float, float], radius: float,
    mass: float = 10, is_static: bool = False,
) -> pymunk.Circle:
  shape = _create_shape(position,
                        lambda body: pymunk.Circle(body, radius),
                        mass, is_static)
  shape.properties = {}
  shape.properties['type'] = 2
  shape.properties['width'], shape.properties['height'] = radius, radius
  return shape


import arm


class BoxState(enum.Enum):
    DEFAULT = 0,
    TOUCHES_FLOOR = 1,
    OUT_OF_BOUNDS = 2,


class Objects:
  def __init__(self):
    self.arm = arm.Arm(pymunk.Vec2d(400, 350))
    self.floor = create_rect((80, 550), (150, 20), is_static=True)
    self.floor.elasticity = 0.2
    self.box = create_rect(
        self.arm.rect2.body.position + (0, -40), (40, 40), 1)
    
    self.dynamic_shapes = self.arm.dynamic_shapes + [self.box]
    self.static_shapes = [self.floor]
    self.all_shapes = self.static_shapes + self.dynamic_shapes
    self.joints = self.arm.joints
    
  def get_info(self):
    num_objects = len(self.all_shapes)
    info = np.zeros((num_objects, NUM_SHAPE_FEATURES))
    for i in range(num_objects):
      info[i] = np.array(shape_info(self.all_shapes[i]))
    return info.T.reshape((1,-1))
    
  def box_state(self) -> BoxState:
    result = BoxState.DEFAULT
    if self.box.shapes_collide(self.floor).points:
      result = BoxState.TOUCHES_FLOOR
    elif (self.box.body.position.x < 0
          or self.box.body.position.x > 900
          or self.box.body.position.y < 0
          or self.box.body.position.y > 900):
      result = BoxState.OUT_OF_BOUNDS
    return result
    
    
NUM_SHAPE_FEATURES = 4


def shape_info(shape: pymunk.Shape):
  # position_x, position_y, mass, velocity_x, velocity_y, kinetic_energy,
  # width, height, type, is_static
  return [
      *shape.body.position,
      #shape.body.mass,
      *shape.body.velocity,
      #shape.body.kinetic_energy,
      #shape.properties['width'],
      #shape.properties['height'],
      #shape.properties['type'],
      #int(shape.body.body_type == pymunk.Body.STATIC),
  ]

  
def discretize(info: np.ndarray, lows, highs, steps):
  result = np.zeros_like(info)
  for i, row in enumerate(info):
    bins = np.arange(lows[i], highs[i] + steps[i], steps[i])
    result[i] = np.digitize(row, bins)
  return result.reshape((1, -1))

    
