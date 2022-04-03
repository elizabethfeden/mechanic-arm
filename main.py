import pygame

from environment import Environment

def process_keys(expected_keys):
  pressed_keys = pygame.key.get_pressed()
  for i, key in enumerate(expected_keys):
    if pressed_keys[key]:
      return i + 1
    
  return 0
    

if __name__ == '__main__':
  env = Environment()
  running = True
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
  
    action = [
        process_keys([pygame.K_SPACE, pygame.K_LEFT, pygame.K_RIGHT]),
        process_keys([pygame.K_f, pygame.K_a, pygame.K_d]),
    ]
    env.step(action)
    env.render()
  
