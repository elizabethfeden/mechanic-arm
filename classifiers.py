import numpy as np
from sklearn import neural_network


class MLPClassifier(neural_network.MLPClassifier):
  def _initialize(self, y, layer_units, dtype):
    super()._initialize(y, layer_units, dtype)
    # We want initial classifier output to behave (almost) like uniform distribution for any input
    self.coefs_[-1] = np.ones((layer_units[-2], layer_units[-1])) / layer_units[-2]
    self.intercepts_[-1] = np.zeros((layer_units[-1],))
