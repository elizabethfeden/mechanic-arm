import multiprocessing as mp
import numpy as np
import pickle
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
  env = Environment(pygame_render=True)
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


def run_random_agent():
  env = Environment(pygame_render=True)
  agent = agents.Agent(env.N_ACTIONS)

  moving_average_num = 20
  rewards = np.array([0] * moving_average_num)
  averages = []
  current_index = 0

  running = True
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False

    done = False
    total_reward = 0
    while not done:
      _, reward, done = env.step_scalar_action(agent.action())
      total_reward += reward
      env.render_clear()
      env.render()

    rewards[current_index] = total_reward
    current_index = (current_index + 1) % moving_average_num
    averages += [rewards.mean()]

    env.reset()

  with open('results.txt', 'w') as file:
    file.write(str(averages))


def _run_fitter(best_pipe_in: mp.connection.Connection,
                results_pipe_in: mp.connection.Connection,
                running_pipe_out: mp.connection.Connection):
  fitter = agents.CrossEntropyFitter(n_sessions=200, n_elites=50)
  while running_pipe_out.recv():
    best = fitter.fit_epoch(verbose=('rewards',))
    best_pipe_in.send(best)
  results_pipe_in.send((fitter.mean_rewards, fitter.median_rewards))
  best_pipe_in.close()
  running_pipe_out.close()
  results_pipe_in.close()


def _simulate_best_game(best_pipe_out: mp.connection.Connection,
                        running_pipe_in: mp.connection.Connection,
                        env: Environment, wins: int, total: int):
  best = best_pipe_out.recv()
  best.reevaluate(env.reset())

  running = True
  running_pipe_in.send(True)

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

  return running, total_reward, best


def run_rl():
  best_pipe_out, best_pipe_in = mp.Pipe()
  results_pipe_out, results_pipe_in = mp.Pipe()
  running_pipe_out, running_pipe_in = mp.Pipe()
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
  env = Environment(pygame_render=True)
  while running:
    running, total_reward, best = _simulate_best_game(
      best_pipe_out, running_pipe_in, env, wins, total)

    total += 1
    if total_reward == env.MAX_REWARD:
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
  with open('model', 'wb') as file:
    pickle.dump(best.policy, file)
    

if __name__ == '__main__':
  if len(sys.argv) > 1 and sys.argv[1] == '-i':
    run_interactive_simulation()
  elif len(sys.argv) > 1 and sys.argv[1] == '-r':
    run_random_agent()
  else:
    run_rl()
  
