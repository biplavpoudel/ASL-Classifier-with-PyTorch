# This is the base layer which Convolution and Dense Layer inherits from

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # Returns output
        pass

    def backward(self, output_gradient, learning_rate):
        # Updates parameters and returns input gradient
        pass
