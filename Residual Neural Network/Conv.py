import numpy as np
from scipy.signal import convolve, correlate
import copy


class Conv:

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = int(num_kernels)

        self.weights = np.random.uniform(0, 1, (self.num_kernels, *self.convolution_shape))
        self.bias = np.random.uniform(0, 1, self.num_kernels)
        self.bias_optimizer = None
        self.weights_optimizer = None
        self._optimizer= None

    def forward(self, input_tensor):

        self.input_tensor = input_tensor
        batch_shape_i = self.input_tensor.shape[0]
        channels_i = self.input_tensor.shape[1]

        if len(self.input_tensor.shape) == 3:
            self.output_tensor = np.zeros((batch_shape_i, self.num_kernels, self.input_tensor.shape[2]))
        else:
            self.output_tensor = np.zeros((batch_shape_i, self.num_kernels, self.input_tensor.shape[2], self.input_tensor.shape[3]))

        for i in range(batch_shape_i):
            for j in range(self.num_kernels):
                correlation = 0
                for k in range(channels_i):

                    if len(self.input_tensor.shape) == 3:
                        correlation += correlate(self.input_tensor[i, k, :], self.weights[j, k, :], 'same')
                    else:
                        correlation += correlate(self.input_tensor[i, k, :, :], self.weights[j, k, :, :], 'same')

                self.output_tensor[i, j] = correlation + self.bias[j]

        if len(self.input_tensor.shape) == 3:
            self.output_tensor = self.output_tensor[:, :, ::self.stride_shape[0]]
        else:
            self.output_tensor= self.output_tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]
                
        return self.output_tensor

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

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self.bias_optimizer = copy.deepcopy(optimizer)
        self.weights_optimizer = copy.deepcopy(optimizer)

    def backward(self, error_tensor):

        self.error_tensor = error_tensor

        channel_i = self.input_tensor.shape[1]
        batch_e = self.error_tensor.shape[0]

        self.gradient_weights = np.zeros(self.weights.shape)
        self.gradient_bias = np.zeros(self.num_kernels)
        one_dimensional = len(self.input_tensor.shape) == 3

        if one_dimensional:
            self.error_tensor_upscaled = np.zeros((self.input_tensor.shape[0], self.error_tensor.shape[1], self.input_tensor.shape[2]))
            self.previous_error = np.zeros(self.input_tensor.shape)
        else:
            self.error_tensor_upscaled = np.zeros((self.input_tensor.shape[0], self.error_tensor.shape[1], self.input_tensor.shape[2], self.input_tensor.shape[3]))
            self.previous_error = np.zeros(self.input_tensor.shape)

        if one_dimensional:
            self.gradient_bias = np.sum(self.error_tensor, axis=(0,2))
            self.error_tensor_upscaled[:, :, ::self.stride_shape[0]] = self.error_tensor
        else:
            self.gradient_bias = np.sum(self.error_tensor, axis=(0,2,3))
            self.error_tensor_upscaled[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = self.error_tensor

        for i in range(batch_e):
            for j in range(channel_i):
                correlation = 0
                for k in range(self.num_kernels):

                    if one_dimensional:
                        correlation += convolve(self.error_tensor_upscaled[i, k, :], self.weights[k, j, :], 'same')
                    else:
                        correlation += convolve(self.error_tensor_upscaled[i, k, :, :], self.weights[k, j, :, :], 'same')

                self.previous_error[i, j] = correlation

        if one_dimensional:
            if self.weights.shape[2] %2 == 0:
                self.pad_top = np.floor(self.weights.shape[2]/2).astype(int)
                self.pad_bottom = np.floor(self.weights.shape[2]/2).astype(int) -1
            else:
                self.pad_top = np.floor(self.weights.shape[2]/2).astype(int)
                self.pad_bottom = np.floor(self.weights.shape[2]/2).astype(int)

        else:
            if self.weights.shape[2] %2 == 0:
                self.pad_top = np.floor(self.weights.shape[2]/2).astype(int)
                self.pad_bottom = np.floor(self.weights.shape[2]/2).astype(int) -1

            else:
                self.pad_top = np.floor(self.weights.shape[2]/2).astype(int)
                self.pad_bottom = np.floor(self.weights.shape[2]/2).astype(int)

            if self.weights.shape[3] % 2 == 0:
                self.pad_left = np.floor(self.weights.shape[3] / 2).astype(int)
                self.pad_right = np.floor(self.weights.shape[3] / 2).astype(int) - 1

            else:
                self.pad_left = np.floor(self.weights.shape[3] / 2).astype(int)
                self.pad_right = np.floor(self.weights.shape[3] / 2).astype(int)

        if one_dimensional:
            self.pad_list = [(0, 0), (0, 0), (self.pad_top, self.pad_bottom)]
            self.input_tensor_pad = np.pad(self.input_tensor, self.pad_list, mode="constant", constant_values=0)

        else:
            self.pad_list = [(0, 0), (0, 0), (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right)]
            self.input_tensor_pad = np.pad(self.input_tensor, self.pad_list, mode="constant", constant_values=0)

        for i in range(batch_e):
            for j in range(self.num_kernels):
                for k in range(channel_i):

                    if one_dimensional:
                        self.gradient_weights[j, k] += correlate(self.input_tensor_pad[i, k, :], self.error_tensor_upscaled[i, j, :], 'valid')
                    else:
                        self.gradient_weights[j, k] += correlate(self.input_tensor_pad[i, k, :, :], self.error_tensor_upscaled[i, j, :, :], 'valid')

        if self._optimizer is not None:
            self.weights = self.weights_optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        return self.previous_error

    def initialize(self, weights_initializer, bias_initializer):
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

        fan_in = np.product(self.convolution_shape)
        fan_out = np.product((self.num_kernels, *self.convolution_shape[1:]))

        self.weights = self.weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = self.bias_initializer.initialize(self.bias.shape, fan_in, fan_out)
