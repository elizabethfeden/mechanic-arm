"""Utility package for `agents.py`."""

import numpy as np
from sklearn import neural_network


class MLPClassifier(neural_network.MLPClassifier):
  def _initialize(self, y, layer_units, dtype):
    super()._initialize(y, layer_units, dtype)
    # We want initial classifier output to behave (almost) like uniform distribution for any input
    coef_shape = (layer_units[-2], layer_units[-1])
    self.coefs_[-1] = (np.ones(coef_shape) / coef_shape[0]
                       + np.random.normal(scale=0.2 / coef_shape[0], size=coef_shape))
    self.intercepts_[-1] = np.zeros((layer_units[-1],))
