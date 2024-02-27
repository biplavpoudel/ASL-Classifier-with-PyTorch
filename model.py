import torch
import torch.nn as nn
from CustomLayers.Convolution import Convolution
from CustomLayers.Activation import RectifiedLinearUnit
from CustomLayers.MaxPool import MaxPool2d
from CustomLayers.Dropout import Dropout
from CustomLayers.Flatten import Flatten
from CustomLayers.Dense import Dense
from CustomLayers.Activation import LogSoftmax


class ASLClassifier(nn.Module):
    def __init__(self, num_classes=26):
        super(ASLClassifier, self).__init__()
        self.conv1 = Convolution(filters=32, kernel_size=3, input_shape=(3, 224, 224))
        self.activation1 = RectifiedLinearUnit()
        self.pool1 = MaxPool2d(pool_size=2, stride=2)
        self.conv2 = Convolution(filters=64, kernel_size=3, input_shape=(32, 111, 111))
        self.activation2 = RectifiedLinearUnit()
        self.pool2 = MaxPool2d(pool_size=2, stride=2)
        self.conv3 = Convolution(filters=64, kernel_size=3, input_shape=(64, 54, 54))
        self.activation3 = RectifiedLinearUnit()
        self.pool3 = MaxPool2d(pool_size=2, stride=2)    # Output Shape: (64, 26, 26)
        self.flatten = Flatten()
        self.dense1 = Dense(64*26*26, 128)
        self.activation4 = RectifiedLinearUnit()
        self.dropout = Dropout(p=0.5)
        self.dense2 = Dense(128, 26)
        self.activation5 = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.activation3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.activation4(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.activation5(x)
        return x

    def summary(self):
        total_parameters = 0
        input_size = (3, 224, 224)
        print(f"Model name: \"ASL Classifier\"")
        print(f"Input size: {input_size}")
        print(f"{'Layer (type)':<20}{'Input Shape':<20}"
              f"{'Output Shape':<20}{'Param#':<10}\n{'-'*75}")
        for name, layer in self.named_children():
            input_tensor = torch.zeros(1, *input_size)
            output = layer(input_tensor)
            output_shape = tuple(output.size())[1:]
            params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            total_parameters += params
            print(f"{name:<20}{str(input_size):<20}{str(output_shape):<20}{params:<10}")
            input_size = output_shape
            print('-'*75)
        print(f"Total number of parameters: {total_parameters}")


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()

    # move model to GPU if CUDA is available
    if use_cuda:
        model = ASLClassifier().cuda()
    else:
        model = ASLClassifier()
    model.summary()
