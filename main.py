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
  
  
def run_interactive_simulation():
  env = Environment()
  running = True
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
  
    _, _, done, _ = env.step([
          process_keys([pygame.K_SPACE, pygame.K_LEFT, pygame.K_RIGHT]),
          process_keys([pygame.K_f, pygame.K_a, pygame.K_d]),
    ])
    env.render_clear()
    env.render()
    
    if done:
      env.reset()
      
      
def run_rl():
  env = Environment()
  fitter = agents.CrossEntropyFitter(env)
  #best = agents.Agent(n_actions=16)
  running = True
  wins = 0
  total = 0
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False

    best = fitter.fit_epoch()
    best.reevaluate(env.reset())
    
    done = False
    while not done:
      info, _, done, _ = env.step_scalar_action(best.action())
      best.reevaluate(info)
      env.render_clear()
      if total > 0:
        env.render_text('wins%: {:.2f}'.format(wins / total * 100), (0, 40))
      env.render_text(f'total: {total}', (0, 60))
      env.render()

    total += 1
    if env.total_reward == env.MAX_REWARD:
      wins += 1
    

if __name__ == '__main__':
  if len(sys.argv) > 1 and sys.argv[1] == '-i':
    run_interactive_simulation()
  else:
    run_rl()
  
