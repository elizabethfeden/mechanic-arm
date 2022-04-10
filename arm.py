import pymunk

from objects import create_circle, create_rect

class Arm:
  CIRCLE1_POWER = 20000
  CIRCLE2_POWER = 2000
  BASE_POWERS = (CIRCLE1_POWER, CIRCLE2_POWER)
  
  def _create_pinjoints(self, circle, rect, rect_shape):
    w, h = rect_shape
    w_half, h_half = w // 2, h // 2
    west = pymunk.PinJoint(rect.body, circle.body, (-w_half, -h_half), (0, 0))
    east = pymunk.PinJoint(rect.body, circle.body, (w_half, -h_half), (0, 0))
    north = pymunk.PinJoint(rect.body, circle.body, (0, -h_half), (0, circle.radius))
    south = pymunk.PinJoint(rect.body, circle.body, (0, h_half), (0, circle.radius))
    return west, east, north, south

  def __init__(self, static_body: pymunk.Body, start: pymunk.Vec2d):
    self.start = start
  
    # We count shapes starting from `start`
    # and to the fingers or whatewer the arm has
    self.circle1 = create_circle(start, 50)
    self.rect1 = create_rect(start + (0, 100), (40, 100))
    self.circle2 = create_circle(start + (0, 150), 20, 2)
    self.rect2 = create_rect(start + (0, 195), (40, 50), 2)
    
    # Ignore collisions between `rect1` and `circle2`
    self.rect1.filter = pymunk.ShapeFilter(group=1)
    self.circle2.filter = pymunk.ShapeFilter(group=1)
    
    self.dynamic_shapes = [self.circle1, self.rect1, self.circle2, self.rect2]
    
    c1 = pymunk.PivotJoint(static_body, self.circle1.body, start)
    c2 = pymunk.PivotJoint(self.rect1.body, self.circle2.body, start + (0, 150))
    
    self.joints = [
        c1, c2,
        *self._create_pinjoints(self.circle1, self.rect1, (40, 100)),
        *self._create_pinjoints(self.circle2, self.rect2, (40, 50)),
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
      
