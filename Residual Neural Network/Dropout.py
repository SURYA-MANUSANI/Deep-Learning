import numpy as np
from .Base import BaseLayer


class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, input_tensor):
        if not self.testing_phase:
            self.x = np.random.binomial(1, self.probability, input_tensor.shape)
            self.x = self.x / self.probability
        else:
            self.x = np.ones(input_tensor.shape)
        return self.x * input_tensor

    def backward(self, error_tensor):
        return error_tensor * self.x