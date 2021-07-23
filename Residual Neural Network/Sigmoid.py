import numpy as np


class Sigmoid:

    def _init_(self):
        pass

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.activation = np.divide(1, (1 + np.exp(-self.input_tensor)))
        return self.activation

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        self.derivative = self.activation * (1 - self.activation)
        self.product = self.error_tensor * self.derivative
        return self.product
