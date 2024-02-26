import torch
import torch.nn as nn
import torch.nn.functional as F
import Loss


class Convolution(nn.Module):
    def __init__(self, filters, kernel_size, input_shape):
        super(Convolution, self).__init__()
        input_depth, input_height, input_width = input_shape
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.filters = filters
        self.kernel_size = kernel_size
        self.output_shape = (filters, input_height - kernel_size + 1, input_width - kernel_size + 1)
        # for valid convolution, output size is: (image_dimension - kernel_dimension +1)
        # for full convolution, output size is: (image_dimension + kernel_dimension -1)
        self.kernel_shape = (filters, input_depth, kernel_size, kernel_size)
        self.kernels = nn.Parameter(torch.randn(*self.kernel_shape))
        # This is equivalent to: self.kernels = np.random.randn((64,3,3,3))
        self.biases = nn.Parameter(torch.randn(*self.output_shape))

    def forward(self, input):
        input = input
        output = F.conv2d(input, self.kernels, bias=self.biases, stride=1, padding=0)
        return output

    def backward(self, output_gradient, learning_rate):
        loss_function = Loss.CrossEntropyLoss()
        loss = loss_function(self.forward(self.input), output_gradient)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
