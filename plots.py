import matplotlib.pyplot as plt

FILENAME = 'results'

with open(FILENAME + '.txt', 'r') as file:
  for name in ['means', 'medians', 'best averages']:
    y = [float(x) for x in file.readline().strip()[1:-1].split(', ')]
    print(y)
    plt.plot(range(len(y)), y, label=name)
  plt.ylabel('Reward')
  plt.xlabel('Epochs')
  plt.title('Reward trend')
  plt.legend()
  plt.savefig(FILENAME + '.png')
