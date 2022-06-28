import multiprocessing as mp
import numpy as np
import pickle
import pygame
import sys

from typing import List

import agents
from environment import Environment
import simulation


def process_keys(expected_keys: List[int]):
  pressed_keys = pygame.key.get_pressed()
  for i, key in enumerate(expected_keys):
    if pressed_keys[key]:
      return i + 1
    
  return 0

  
def run_interactive_simulation():
  env = Environment(pygame_render=True)
  running = True
  while running:
    running = env.window.check_close_event()
  
    _, _, done = env.step([
          process_keys([pygame.K_SPACE, pygame.K_LEFT, pygame.K_RIGHT]),
          process_keys([pygame.K_f, pygame.K_a, pygame.K_d]),
    ])
    env.window.render_clear()
    env.render()
    
    if done:
      env.reset()


def run_random_agent():
  simulation.Simulation(
    agent=agents.Agent(Environment.N_ACTIONS)
  ).run()


def run_crossentropy_agent():
  simulation.Simulation(
    fitter=agents.CrossEntropyFitter(n_sessions=200, n_elites=50),
    parallel=True,
    save_options=(
      simulation.SaveOption.MODEL,
      simulation.SaveOption.MEAN_REWARDS,
      simulation.SaveOption.MEDIAN_REWARDS,
      simulation.SaveOption.MOVING_AVERAGES,
    ),
  ).run()
    

if __name__ == '__main__':
  if len(sys.argv) > 1 and sys.argv[1] == '-i':
    run_interactive_simulation()
  elif len(sys.argv) > 1 and sys.argv[1] == '-r':
    run_random_agent()
  else:
    run_crossentropy_agent()
  
