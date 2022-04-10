from environment import Environment

if __name__ == '__main__':
  env = Environment()
  while True:
    env.step(None)
    env.render()
  
