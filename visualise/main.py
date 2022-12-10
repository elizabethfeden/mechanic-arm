import numpy as np
import pickle
import sys

import draw_nn


def main():
  if len(sys.argv) < 2:
    raise Exception('Expected argument: file with model')
  with open(sys.argv[1], 'rb') as file:
    model = pickle.load(file)

  #model.predict([[1]])
    
  structure = np.hstack((
      [model.n_features_in_],
      np.asarray(model.hidden_layer_sizes),
      [len(model.classes_)]
  ))

  network = draw_nn.DrawNN(structure, model.coefs_)
  network.draw()


if __name__ == '__main__':
  main()
