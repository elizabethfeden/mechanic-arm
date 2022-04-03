"""Package for simulation objects storage management"""

import pymunk
from typing import Callable, Tuple


def _create_shape(
    position: Tuple[float, float],
    mass: float,
    shape_constructor: Callable[[pymunk.Body], pymunk.Shape]
) -> pymunk.Shape:
  body = pymunk.Body()
  body.position = position
  shape = shape_constructor(body)
  shape.mass = mass
  shape.elasticity = 0.999
  shape.friction = 1
  return shape


def create_rect(
    position: Tuple[float, float], size: Tuple[float, float], mass: float = 10
) -> pymunk.Poly:
  return _create_shape(position, mass,
                       lambda body: pymunk.Poly.create_box(body, size))
    
    
def create_circle(
    position: Tuple[float, float], radius: float, mass: float = 10
) -> pymunk.Circle:
  return _create_shape(position, mass,
                       lambda body: pymunk.Circle(body, radius))


import arm


class Objects:
  def __init__(self, static_body: pymunk.Body):
    self.arm = arm.Arm(static_body, pymunk.Vec2d(400, 350))
    self.dynamic_shapes = self.arm.dynamic_shapes
    self.static_shapes = []
    self.joints = self.arm.joints
    
