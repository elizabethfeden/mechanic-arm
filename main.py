import numpy as np
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
  
    _, _, done = env.step([
          process_keys([pygame.K_SPACE, pygame.K_LEFT, pygame.K_RIGHT]),
          process_keys([pygame.K_f, pygame.K_a, pygame.K_d]),
    ])
    env.render_clear()
    env.render()
    
    if done:
      env.reset()

      
def run_rl():
  env = Environment()
  fitter = agents.CrossEntropyFitter(env, n_sessions=200, n_elites=50)
  running = True
  wins = 0
  total = 0
  moving_average_num = 10
  rewards = np.array([0] * moving_average_num)
  averages = []
  current_index = 0
  while running:
    best = fitter.fit_epoch(verbose=('rewards',))
    best.reevaluate(env.reset())

    done = False
    total_reward = 0
    while not done:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          running = False

      info, reward, done = env.step_scalar_action(best.action())
      total_reward += reward
      best.reevaluate(info)
      env.render_clear()
      if total > 0:
        env.render_text('wins%: {:.2f}'.format(wins / total * 100), (0, 40))
      env.render_text(f'total: {total}', (0, 60))
      env.render()

    total += 1
    if env.total_reward == env.MAX_REWARD:
      wins += 1
    rewards[current_index] = total_reward
    current_index = (current_index + 1) % moving_average_num
    averages += [rewards.mean()]

  with open('results.txt', 'w') as file:
    file.write(str(fitter.mean_rewards) + '\n')
    file.write(str(fitter.median_rewards) + '\n')
    file.write(str(averages))
    

if __name__ == '__main__':
  if len(sys.argv) > 1 and sys.argv[1] == '-i':
    run_interactive_simulation()
  else:
    run_rl()
  
