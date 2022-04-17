import gym
import pygame

import objects
import physics


class Environment(gym.Env):
  def __init__(self, pygame_render: bool = True):
    super(Environment, self).__init__()
    
    self.max_reward = 50
    
    self.reset()
    
    self._pygame_render = pygame_render
    if pygame_render:
      pygame.init()
      self._screen = pygame.display.set_mode((600, 600))
      self._clock = pygame.time.Clock()
      self._pymunk_options = physics.get_print_options(self._screen)
      self._font = pygame.font.SysFont(None, 30)
    else:
      self._pymunk_options = physics.get_print_options()
    
  def reset(self):
    self._objects = objects.Objects()
    self._physics = physics.Simulation(self._objects)
    self.total_reward = 0
    
  def render(self):
    if self._pygame_render:
      self._screen.fill(pygame.Color('white'))
      
    self._physics.render(self._pymunk_options)
    
    if self._pygame_render:
      text = self._font.render(str(self.total_reward), True, pygame.Color('black'))
      self._screen.blit(text, (0, 0))
      
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
    
    done = False
    box_state = self._objects.box_state()
    if box_state == objects.BoxState.DEFAULT:
      reward = 0
    elif box_state == objects.BoxState.TOUCHES_FLOOR:
      reward = 1
    else:
      reward = -self.max_reward
      done = True
    self.total_reward += reward
    if self.total_reward >= self.max_reward:
      done = True
      
    return self._objects.get_info(), reward, done, []

