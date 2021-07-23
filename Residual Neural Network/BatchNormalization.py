from .Base import BaseLayer
from .Helpers import compute_bn_gradients
import numpy as np
import copy


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels  # number of channels in the input tensor
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)
        self._optimizer = None
        self.check = False
        self.initialize(0, 0)
        self.input_normalized = None
        self.alpha = 0.8  #  moving decay.
        self.check_first_batch = True
        self.mu = 0
        self.mu_b = 0
        self.sigma = 0
        self.sigma_b = 0

    def initialize(self, a, b):
        # we need to initialize our bias and weights based on channel size
        #  bias will be zero (additive nature) and weights will be one (multiplication)
        # since you do not want the weights γ and bias β to have an impact at the beginning of the training.
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    def forward(self, input_tensor):

        if len(input_tensor.shape) == 4:
            input_tensor = self.reformat(input_tensor)
            self.check = True

        self.input_tensor = input_tensor

        if self.testing_phase:
            mu_b = self.mu
            sigma_b = self.sigma

        else:
            mu_b = np.mean(input_tensor, axis=0)
            sigma_b = np.var(input_tensor, axis=0)

        input_norm = (input_tensor - mu_b) / np.sqrt(sigma_b + np.finfo(np.float).eps)

        if not self.testing_phase:
            self.mu = self.alpha * self.mu + (1 - self.alpha) * mu_b
            self.sigma = self.alpha * self.sigma + (1 - self.alpha) * sigma_b
            self.mu_b = mu_b
            self.sigma_b = sigma_b
            self.input_normalized = input_norm

        if self.check_first_batch:
            self.mu = mu_b
            self.sigma = sigma_b
            self.check_first_batch = False

        output = self.weights * input_norm + self.bias

        if self.check:
            output = self.reformat(output)
            self.check = False

        return output

    def backward(self, error_tensor):

        if len(error_tensor.shape) == 4:
            error_tensor = self.reformat(error_tensor)
            self.check = True

        self.gradient_weights = np.sum(self.input_normalized * error_tensor, axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)

        gradient_input = compute_bn_gradients(error_tensor, self.input_tensor,
                                              self.weights, self.mu_b, self.sigma_b)

        if self._optimizer != None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)

        if (self.check == True):
            gradient_input = self.reformat(gradient_input)
            self.check = False
        return gradient_input

    def reformat(self, tensor):
        # we will reshape 2 dimensions into 4 dimensions and vice-versa
        if (len(tensor.shape)) == 4:
            self.tensor_shape = tensor.shape
            B, H, M, N = tensor.shape
            tensor = np.reshape(tensor, (B, H, M * N))
            tensor = np.transpose(tensor, (0, 2, 1))
            tensor = np.reshape(tensor, (B * M * N, H))
        elif (len(tensor.shape)) == 2:
            B, H, M, N = self.tensor_shape
            tensor = np.reshape(tensor, (B, M * N, H))
            tensor = np.transpose(tensor, (0, 2, 1))
            tensor = np.reshape(tensor, (B, H, M, N))

        return tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = copy.copy(value)

    @property
    def get_gradient_weights(self):
        gradient_weights = np.matmul(np.transpose(self.input_tensor), self.error_tensor)
        return gradient_weights

    def get_regularization_loss(self):
        try:
            return self.optimizer.regularizer.norm(self.weights)
        except:
            return 0