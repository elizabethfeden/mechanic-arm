import sys
import matplotlib.pyplot as plt

IN_FILENAMES = ['cross-entropy', 'random']
OUT_FILENAME = 'results.png'

for name in IN_FILENAMES:
  with open(name + '.txt', 'r') as file:
    y = [float(x) for x in file.readline().strip()[1:-1].split(', ')][:200]
    plt.plot(range(len(y)), y, label=name)
    
#with open(IN_FILENAME, 'r') as file:
#  for name in ['means', 'medians', 'best averages']:
#    y = [float(x) for x in file.readline().strip()[1:-1].split(', ')]
#    plt.plot(range(len(y)), y, label=name)
plt.xlabel('Epochs')
plt.ylabel('Rewards')
plt.title('Rewards average comparison')
plt.legend()
plt.savefig(OUT_FILENAME)
