import numpy as np
from BaseLayer import Layer
from scipy import signal


class Convolution(Layer):
    def __init__(self, filters, kernel_size, activation, input_shape):
        super().__init__()
        input_depth, input_height, input_width = input_shape
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.output_shape = (filters, input_height-kernel_size+1, input_width-kernel_size+1)
        # for valid convolution, output size is: (image_dimension - kernel_dimension +1)
        # for full convolution, output size is: (image_dimension + kernel_dimension -1)
        self.kernel_shape = (filters, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernel_shape)
        # This is equivalent to: self.kernels = np.random.randn((64,3,3,3))
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)

        for i in range(self.filters):
            for j in range(self.input_depth):
                self.output[i] += signal.convolve2d(self.input[j], self.kernels[i, j], "valid")

        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros_like(self.kernels)
        input_gradient = np.zeros_like(self.input)


