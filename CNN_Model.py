import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU as RectifiedLinearUnit
from torch.nn import Conv2d as Convolution
from torch.nn import Dropout
from torch.nn import MaxPool2d
from torch.nn import Flatten
from torch.nn import Linear as Dense
from torch.nn import LogSoftmax as LogSoftmax

# class ASLClassifier(nn.Module):
#     def __init__(self, num_classes=27):
#         super(ASLClassifier, self).__init__()
#         self.conv1 = Convolution(in_channels=3, out_channels=32, kernel_size=3)
#         self.activation1 = RectifiedLinearUnit()
#         self.pool1 = MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = Convolution(in_channels=32, out_channels=64, kernel_size=3)
#         self.activation2 = RectifiedLinearUnit()
#         self.pool2 = MaxPool2d(kernel_size=2, stride=2)
#         self.conv3 = Convolution(in_channels=64, out_channels=64, kernel_size=3)
#         self.activation3 = RectifiedLinearUnit()
#         self.pool3 = MaxPool2d(kernel_size=2, stride=2)  # Output Shape: (64, 26, 26)
#         self.flatten = Flatten()
#         self.dense1 = Dense(64 * 26 * 26, 128)
#         self.activation4 = RectifiedLinearUnit()
#         self.dropout = Dropout(p=0.5)
#         self.dense2 = Dense(128, num_classes)
#         self.activation5 = LogSoftmax(dim=1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.activation1(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.activation2(x)
#         x = self.pool2(x)
#         x = self.conv3(x)
#         x = self.activation3(x)
#         x = self.pool3(x)
#         x = self.flatten(x)
#         x = self.dense1(x)
#         x = self.activation4(x)
#         x = self.dropout(x)
#         x = self.dense2(x)
#         x = self.activation5(x)
#         return x


class ASLClassifier(nn.Module):
    def __init__(self, num_classes=27):
        super(ASLClassifier, self).__init__()
        self.features = nn.Sequential(
            Convolution(in_channels=3, out_channels=32, kernel_size=3),
            RectifiedLinearUnit(),
            MaxPool2d(kernel_size=2, stride=2),
            Convolution(in_channels=32, out_channels=64, kernel_size=3),
            RectifiedLinearUnit(),
            MaxPool2d(kernel_size=2, stride=2),
            Convolution(in_channels=64, out_channels=64, kernel_size=3),
            RectifiedLinearUnit(),
            MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=0.5),
            Dense(64 * 7 * 7, 128),
            RectifiedLinearUnit(),
            Dropout(p=0.5),
            Dense(128, num_classes),
            LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def simple_summary(self):
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

    def summary(self):
        total_parameters = 0
        input_size = (3, 224, 224)
        print(f"Model name: \"ASL Classifier\"")
        print(f"Input size: {input_size}")
        print(f"{'Layer (type)':<20}{'Input Shape':<30}"
              f"{'Output Shape':<30}{'Param#':<10}\n{'-' * 90}")
        for name, layer in self.named_children():
            if isinstance(layer, torch.nn.Sequential):
                for child_name, child_layer in layer.named_children():
                    input_tensor = torch.zeros(1, *input_size)
                    output = child_layer(input_tensor)
                    output_shape = tuple(output.size())[1:]
                    params = sum(p.numel() for p in child_layer.parameters() if p.requires_grad)
                    total_parameters += params
                    print(f"  {type(child_layer).__name__:<20}{str(input_size):<30}"
                          f"{str(output_shape):<30}{params:<10}")
                    input_size = output_shape  # Update input size for the next layer
                    print('-' * 90)
            else:
                input_tensor = torch.zeros(1, *input_size)
                output = layer(input_tensor)
                output_shape = tuple(output.size())[1:]
                params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
                total_parameters += params
                print(f"{type(layer).__name__:<20}{str(input_size):<30}"
                      f"{str(output_shape):<30}{params:<10}")
                input_size = output_shape  # Update input size for the next layer
                print('-' * 90)
        print(f"Total number of parameters: {total_parameters}")


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()

    # Define the device
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"The device used is: {device}")

    # Create the model
    model = ASLClassifier()

    # Sample input tensor (random initialization)
    input_tensor = torch.randn(1, 3, 224, 224)

    # Call the summary method
    print("\nThe basic summary is: \n")
    model.simple_summary()
    print("\n\nThe complete summary is: \n")
    model.summary()
