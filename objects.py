"""Package for simulation objects storage management"""

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
  return _create_shape(position,
                       lambda body: pymunk.Poly.create_box(body, size),
                       mass, is_static)
    
    
def create_circle(
    position: Tuple[float, float], radius: float,
    mass: float = 10, is_static: bool = False,
) -> pymunk.Circle:
  return _create_shape(position,
                       lambda body: pymunk.Circle(body, radius),
                       mass, is_static)


import arm


class Objects:
  def __init__(self):
    self.arm = arm.Arm(pymunk.Vec2d(400, 350))
    self.floor = create_rect((80, 550), (200, 20), is_static=True)
    self.floor.elasticity = 0.2
    self.box = create_rect(
        self.arm.rect2.body.position + (0, -40), (40, 40), 1)
    
    self.dynamic_shapes = self.arm.dynamic_shapes + [self.box]
    self.static_shapes = [self.floor]
    self.joints = self.arm.joints
    
