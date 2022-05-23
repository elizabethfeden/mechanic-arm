import sys
import matplotlib.pyplot as plt

IN_FILENAMES = ['0-01', '0-02', '0-03', '0-04']
OUT_FILENAME = ('results' if len(sys.argv) <= 1 else
                sys.argv[1] if len(sys.argv) <= 2 else sys.argv[2]) + '.png'

for name in IN_FILENAMES:
  with open(name + '.txt', 'r') as file:
    y = [float(x) for x in file.readline().strip()[1:-1].split(', ')]
    plt.plot(range(len(y)), y, label=name.replace('-', '.'))
    
#with open(IN_FILENAME, 'r') as file:
#  for name in ['means', 'medians', 'best averages']:
#    y = [float(x) for x in file.readline().strip()[1:-1].split(', ')]
#    plt.plot(range(len(y)), y, label=name)
plt.xlabel('Epochs')
plt.ylabel('Means (cross-entropy)')
plt.title('Physics step comparison')
plt.legend()
plt.savefig(OUT_FILENAME)
