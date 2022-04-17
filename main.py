import pygame
import sys

import agents
from environment import Environment

def process_keys(expected_keys):
  pressed_keys = pygame.key.get_pressed()
  for i, key in enumerate(expected_keys):
    if pressed_keys[key]:
      return i + 1
    
  return 0


def reshape_agent_action(action: int):
  return [action // 4, action % 4]
    

if __name__ == '__main__':
  env = Environment()
  agent = agents.Agent(16, 600 * 600)
  interacting = (len(sys.argv) > 1 and sys.argv[1] == '-i')
  running = True
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
  
    if interacting:
      action = [
          process_keys([pygame.K_SPACE, pygame.K_LEFT, pygame.K_RIGHT]),
          process_keys([pygame.K_f, pygame.K_a, pygame.K_d]),
      ]
    else:
      action = reshape_agent_action(agent.action())
      
    info, reward, done, _ = env.step(action)
    
    if not interacting:
      agent.reevaluate(info, reward, done)
    
    env.render()
    
    if done:
      env.reset()
  
