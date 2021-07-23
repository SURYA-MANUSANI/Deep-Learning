import numpy as np


class TanH:

    def _init_(self):
        pass

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.activation = np.tanh(input_tensor)
        return self.activation

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        self.derivative = 1 - np.square(self.activation)
        self.product = self.error_tensor * self.derivative
        return self.product

