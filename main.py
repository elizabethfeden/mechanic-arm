import multiprocessing as mp
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


def _run_fitter(best_pipe_in: mp.connection.Connection,
                results_pipe_in: mp.connection.Connection,
                running_pipe_out: mp.connection.Connection):
  env = Environment(pygame_render=False)
  fitter = agents.CrossEntropyFitter(env, n_sessions=200, n_elites=50)
  while running_pipe_out.recv():
    best = fitter.fit_epoch(verbose=('rewards',))
    best_pipe_in.send(best)
  results_pipe_in.send((fitter.mean_rewards, fitter.median_rewards))
  best_pipe_in.close()
  running_pipe_out.close()
  results_pipe_in.close()


def run_rl():
  best_pipe_out, best_pipe_in = mp.Pipe(duplex=False)
  results_pipe_out, results_pipe_in = mp.Pipe(duplex=False)
  running_pipe_out, running_pipe_in = mp.Pipe(duplex=False)
  fitter_process = mp.Process(
    target=_run_fitter, args=(best_pipe_in, results_pipe_in, running_pipe_out))
  fitter_process.start()

  running = True
  running_pipe_in.send(True)

  wins, total = 0, 0
  moving_average_num = 20
  rewards = np.array([0] * moving_average_num)
  averages = []
  current_index = 0
  env = Environment()
  while running:
    running_pipe_in.send(True)
    best = best_pipe_out.recv()
    best.reevaluate(env.reset())

    done = False
    total_reward = 0
    while not done:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          running = False
          running_pipe_in.send(False)

      info, reward, done = env.step_scalar_action(best.action())
      total_reward += reward
      best.reevaluate(info)
      env.render_clear()
      if total > 0:
        env.render_text('wins%: {:.2f}'.format(wins / total * 100), (0, 40))
      env.render_text(f'total: {total}', (0, 60))
      if not running:
        env.render_text('Preparing to shut down...', (200, 40))
      env.render()

    total += 1
    if env.total_reward == env.MAX_REWARD:
      wins += 1
    rewards[current_index] = total_reward
    current_index = (current_index + 1) % moving_average_num
    averages += [rewards.mean()]

  mean_rewards, median_rewards = results_pipe_out.recv()
  best_pipe_out.close()
  running_pipe_in.close()
  results_pipe_out.close()

  with open('results.txt', 'w') as file:
    file.write(str(mean_rewards) + '\n')
    file.write(str(median_rewards) + '\n')
    file.write(str(averages))
    

if __name__ == '__main__':
  if len(sys.argv) > 1 and sys.argv[1] == '-i':
    run_interactive_simulation()
  else:
    run_rl()
  
