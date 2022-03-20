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


class Objects:
  def __init__(self, space: pymunk.Space):
    self.square = create_rect((70, 20), (50, 50))
    self.circle = create_circle((150, 20), 25)
    self.dynamic_shapes = [self.square, self.circle]
    
    self.floor = pymunk.Segment(space.static_body, (0, 500), (500, 550), 0)
    self.static_shapes = [self.floor]
    
