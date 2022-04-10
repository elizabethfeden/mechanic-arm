import pygame
import pymunk
import pymunk.pygame_util

from objects import Objects


def get_print_options(screen: pygame.Surface = None
                      ) -> pymunk.SpaceDebugDrawOptions:
  return (pymunk.pygame_util.DrawOptions(screen) if screen is not None
          else pymunk.SpaceDebugDrawOptions())
       
          
def debug_print(body: pymunk.Body):
  print(f'velocity {body.velocity}\n'
        f'moment {body.moment}\n'
        f'force {body.force}\n'
        f'energy {body.kinetic_energy}\n')


class Simulation:
  def __init__(self):
    self.space = pymunk.Space()
    self.space.gravity = (0, 400)
    
  def add_objects(self, objects: Objects):
    for shape in objects.dynamic_shapes:
      self.space.add(shape.body, shape)
    
    for shape in objects.static_shapes:
      self.space.add(shape)
      
    for joint in objects.joints:
      self.space.add(joint)
    
  def step(self):
    self.space.step(0.02)
    
  def render(self, print_options: pymunk.SpaceDebugDrawOptions):
    self.space.debug_draw(print_options)

