import gym
import pygame

import objects
import physics


class Environment(gym.Env):
  def __init__(self, pygame_render: bool = True):
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
    self._objects = objects.Objects()
    self._physics = physics.Simulation(self._objects)
    
  def render(self):
    if self._pygame_render:
      self._screen.fill(pygame.Color('white'))
      
    self._physics.render(self._pymunk_options)
    
    if self._pygame_render:
      pygame.display.flip()
      self._clock.tick(50)
    
  def step(self, action):
    for index, cur_action in zip([0, 2], action):
      if cur_action == 1:
        if index == 0:
          self._objects.arm.fix_velocity([index, index + 1])
        else:
          self._objects.arm.fix([index, index + 1])
      elif cur_action == 2:
        self._objects.arm.apply_force_to_circle(index, -1)
      elif cur_action == 3:
        self._objects.arm.apply_force_to_circle(index, 1)
      elif cur_action == 0:
        pass
      else:
        raise Exception('Invalid action')
    
    self._physics.step()

