import gym
import pygame

import objects
import physics


class Environment(gym.Env):
  def __init__(self, pygame_render=True):
    super(Environment, self).__init__()
    
    # self.observation_space = #
    # self.action_space = #
    
    self.reset()
    
    self._pygame_render = pygame_render
    if pygame_render:
      pygame.init()
      self._screen = pygame.display.set_mode((600, 600))
      self._clock = pygame.time.Clock()
      self._pymunk_options = physics.get_print_options(self._screen)
    else:
      self._pymunk_options = physics.get_print_options()
    
  def reset(self):
    self._physics = physics.Simulation()
    self._objects = objects.Objects(self._physics.space)
    self._physics.add_objects(self._objects)
    
  def render(self):
    if self._pygame_render:
      self._screen.fill(pygame.Color('white'))
      
    self._physics.render(self._pymunk_options)
    
    if self._pygame_render:
      pygame.display.flip()
      self._clock.tick(50)
    
  def step(self, action):
    self._physics.step()

