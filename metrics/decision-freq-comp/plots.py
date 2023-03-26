import sys
import matplotlib.pyplot as plt

IN_FILENAMES = ['2', '5', '7', '10', '20']
OUT_FILENAME = 'results.png'

for name in IN_FILENAMES:
  with open(name + '.txt', 'r') as file:
    y = [float(x) for x in file.readline().strip()[1:-1].split(', ')][:51]
    plt.plot(range(len(y)), y, label='freq: ' + name)
    
#with open(IN_FILENAME, 'r') as file:
#  for name in ['means', 'medians', 'best averages']:
#    y = [float(x) for x in file.readline().strip()[1:-1].split(', ')]
#    plt.plot(range(len(y)), y, label=name)
plt.xlabel('Epochs')
plt.ylabel('Means (cross-entropy)')
plt.title('Decision frequency comparison')
plt.legend()
plt.savefig(OUT_FILENAME)
