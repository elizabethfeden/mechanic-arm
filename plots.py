"""Utility package for plotting the saved results."""

import sys
import matplotlib.pyplot as plt

IN_FILENAME = ('results' if len(sys.argv) <= 1 else sys.argv[1]) + '.txt'
OUT_FILENAME = ('results' if len(sys.argv) <= 1 else
                sys.argv[1] if len(sys.argv) <= 2 else sys.argv[2]) + '.png'

with open(IN_FILENAME, 'r') as file:
  for name in ['means', 'medians', 'best averages']:
    y = [float(x) for x in file.readline().strip()[1:-1].split(', ')]
    plt.plot(range(len(y)), y, label=name)
  plt.xlabel('Epochs')
  plt.ylabel('Reward')
  plt.title('Reward trend')
  plt.legend()
  plt.savefig(OUT_FILENAME)
