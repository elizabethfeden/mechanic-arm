import sys
import matplotlib.pyplot as plt

IN_FILENAMES = ['s0', 's1']
OUT_FILENAME = 'results.png'

for name in IN_FILENAMES:
  with open(name + '.txt', 'r') as file:
    y = [float(x) for x in file.readline().strip()[1:-1].split(', ')][:51]
    plt.plot(range(len(y)), y, label=name)
    
plt.xlabel('Epochs')
plt.ylabel('Means (cross-entropy)')
plt.title('State transformation comparison')
plt.legend()
plt.savefig(OUT_FILENAME)
