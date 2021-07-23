import numpy as np

class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def norm(self, weights):
        norm = self.alpha * (np.sum(weights ** 2))
        return norm

    def calculate_gradient(self, weights):
        return self.alpha * weights


class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def norm(self, weights):
        norm = self.alpha * np.sum(np.absolute(weights))
        return norm

    def calculate_gradient(self, weights):
        return self.alpha * np.sign(weights)