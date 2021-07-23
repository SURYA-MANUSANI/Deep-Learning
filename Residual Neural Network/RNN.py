import numpy as np
from .FullyConnected import FullyConnected
from .TanH import TanH
from .Sigmoid import Sigmoid


class RNN:

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_state = np.zeros((1, self.hidden_size))

        self.fcl1 = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.fcl2 = FullyConnected(self.hidden_size, self.output_size)

        self.memory = False
        self.__optimizer = None

    @property
    def memorize(self):
        return self.memory

    @memorize.setter
    def memorize(self, memory):
        self.memory = memory

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.__optimizer = optimizer

    @property
    def weights(self):
        return self.fcl1.weights

    @weights.setter
    def weights(self, weits):
        self.fcl1.weights = weits

    @property
    def gradient_weights(self):
        return self.__gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.__gradient_weights = gradient_weights

    @property
    def gradient_bias(self):
        return self.__gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self.__gradient_bias = gradient_bias

    def initialize(self, weights_initializer, bias_initializer):
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

        self.fcl1.weights = self.weights_initializer.initialize(self.fcl1.weights.shape,
                                                                self.input_size + self.hidden_size, self.hidden_size)
        self.fcl1.bias = self.bias_initializer.initialize(self.fcl1.weights.shape, self.input_size + self.hidden_size,
                                                          self.hidden_size)

        self.fcl2.weights = self.weights_initializer.initialize(self.fcl2.weights.shape, self.hidden_size,
                                                                self.output_size)
        self.fcl2.bias = self.bias_initializer.initialize(self.fcl2.weights.shape, self.hidden_size, self.output_size)

    def forward(self, input_tensor):

        self.input_tensor = input_tensor
        time = self.input_tensor.shape[0]

        self.output_tensor = np.zeros((time, self.output_size))
        self.hidden_layer = np.zeros((time, self.hidden_size))

        self.tanH = TanH()
        self.sig = Sigmoid()

        if self.memorize is False:  # For resetting the hidden state
            self.hidden_state = np.zeros((1, self.hidden_size))
        else:
            self.hidden_state = self.hidden_state

        for i in range(time):
            self.new_input_tensor = np.concatenate([self.input_tensor[i], self.hidden_state[0]])
            self.new_input_tensor = self.new_input_tensor.reshape(1, len(self.new_input_tensor))

            self.fcl1_output = self.fcl1.forward(self.new_input_tensor)
            self.hidden_state = self.tanH.forward(self.fcl1_output)

            self.hidden_layer[i] = self.hidden_state

            self.fcl2_output = self.fcl2.forward(self.hidden_state)
            output = self.sig.forward(self.fcl2_output)

            self.output_tensor[i] = output

        return self.output_tensor

    def backward(self, error_tensor):

        self.error_tensor = error_tensor

        self.previous_error = np.zeros(self.input_tensor.shape)

        self.weight_gradient_fcl1 = np.zeros((self.fcl1.weights.shape[0], self.fcl1.weights.shape[1]))
        self.weight_gradient_fcl2 = np.zeros((self.fcl2.weights.shape[0], self.fcl2.weights.shape[1]))

        self.bias_gradient_fcl1 = np.zeros((self.fcl1.bias.shape[0], self.fcl1.bias.shape[1]))
        self.bias_gradient_fcl2 = np.zeros((self.fcl2.bias.shape[0], self.fcl2.bias.shape[1]))

        self.hidden_layer_error = np.zeros((1, self.hidden_size))

        time = self.error_tensor.shape[0]

        for i in reversed(range(time)):
            self.sig.activation = self.output_tensor[i]
            self.error = self.sig.backward(self.error_tensor[i, :])

            # self.hidden_backward = np.concatenate([self.hidden_layer[i, :], [1]])
            self.hidden_backward = np.expand_dims(self.hidden_layer[i, :], axis=0)
            # print("hbshape = ", self.hidden_backward.shape)

            self.fcl2.input_tensor = self.hidden_backward
            self.error = self.fcl2.backward(
                self.error.reshape(1, self.error.shape[0]))  # (np.expand_dims(self.error, axis=0))

            # print(self.fcl2.weights.shape)
            #print(self.weight_gradient_fcl2.shape)
            #print(self.fcl2.gradient_weights.shape)

            self.weight_gradient_fcl2 = self.weight_gradient_fcl2 + self.fcl2.gradient_weights
            self.bias_gradient_fcl2 = self.bias_gradient_fcl2 + self.fcl2.gradient_bias

            # print("myerror = ", self.error.shape)
            # print("hiddenlayererror = ", self.hidden_layer_error.shape)

            self.error = self.error + self.hidden_layer_error

            self.tanH.activation = self.hidden_layer[i]
            self.error = self.tanH.backward(self.error)

            if i > 0:
                temp = self.hidden_layer[i - 1]
            else:
                temp = np.zeros((self.hidden_size, 1)).T

            # temp = np.expand_dims(temp, axis=0)
            # print(self.input_tensor[i].shape)

            input_fcl1 = np.concatenate([self.input_tensor[i], temp.flatten()])
            input_fcl1 = np.expand_dims(input_fcl1, axis=0)

            self.fcl1.input_tensor = input_fcl1

            self.error = self.fcl1.backward(self.error)

            # print("wgfc1s = ",self.weight_gradient_fcl1.shape)
            # print(self.fcl1.gradient_weights.shape)

            self.weight_gradient_fcl1 = self.weight_gradient_fcl1 + self.fcl1.gradient_weights
            self.bias_gradient_fcl1 = self.bias_gradient_fcl1 + self.fcl1.gradient_bias

            self.previous_error[i] = self.error[0, 0:self.input_size]

            self.hidden_layer_error = self.error[0, self.input_size:]

        self.gradient_weights = self.weight_gradient_fcl1
        self.gradient_bias = self.bias_gradient_fcl1

        if self.optimizer is not None:
            self.fcl1.weights = self.optimizer.calculate_update(self.fcl1.weights, self.weight_gradient_fcl1)
            self.fcl1.bias = self.optimizer.calculate_update(self.fcl1.bias, self.bias_gradient_fcl1)

            # print(self.fcl2.weights.shape)
            # print(self.weight_gradient_fcl2.shape)

            self.fcl2.weights = self.optimizer.calculate_update(self.fcl2.weights, self.weight_gradient_fcl2)
            self.fcl2.bias = self.optimizer.calculate_update(self.fcl2.bias, self.bias_gradient_fcl2)

        return self.previous_error
