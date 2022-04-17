import pymunk

from objects import create_circle, create_rect, static_body

class Arm:
  CIRCLE1_POWER = 20000
  CIRCLE2_POWER = 2000
  BASE_POWERS = (CIRCLE1_POWER, CIRCLE2_POWER)
  
  def _create_pinjoints(self, circle, rect, rect_shape):
    w, h = rect_shape
    w_half, h_half = w // 2, h // 2
    west = pymunk.PinJoint(rect.body, circle.body, (w_half, h_half), (0, 0))
    east = pymunk.PinJoint(rect.body, circle.body, (w_half, -h_half), (0, 0))
    north = pymunk.PinJoint(rect.body, circle.body, (-w_half, 0), (-circle.radius, 0))
    south = pymunk.PinJoint(rect.body, circle.body, (w_half, 0), (-circle.radius, 0))
    return west, east, north, south

  def __init__(self, start: pymunk.Vec2d):
    self.start = start
  
    # We count shapes starting from `start`
    # and to the fingers or whatewer the arm has
    self.circle1 = create_circle(start, 50)
    self.rect1 = create_rect(start + (-100, 0), (100, 40))
    self.circle2 = create_circle(start + (-150, 0), 20, 2)
    self.rect2 = create_rect(start + (-195, 0), (50, 40), 2)
    
    # Ignore collisions between `rect1` and `circle2`
    self.rect1.filter = pymunk.ShapeFilter(group=1)
    self.circle2.filter = pymunk.ShapeFilter(group=1)
    
    self.dynamic_shapes = [self.circle1, self.rect1, self.circle2, self.rect2]
    
    c1 = pymunk.PivotJoint(static_body(), self.circle1.body, start)
    c2 = pymunk.PivotJoint(self.rect1.body, self.circle2.body, start + (-150, 0))
    
    self.joints = [
        c1, c2,
        *self._create_pinjoints(self.circle1, self.rect1, (100, 40)),
        *self._create_pinjoints(self.circle2, self.rect2, (50, 40)),
    ]
    
  def apply_force_to_circle(self, index: int, multiplier: float = 1):
    circle = self.dynamic_shapes[index]
    circle.body.apply_force_at_world_point(
        (self.BASE_POWERS[index // 2] * multiplier, 0),
        circle.body.position + (0, circle.radius)
    )
    
  def fix(self, indices):
    for i in indices:
      self.dynamic_shapes[i].body.angular_velocity = 0
      
  def fix_velocity(self, indices):
    for i in indices:
      self.dynamic_shapes[i].body.velocity = (0, 0)
