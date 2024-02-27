import torch
import torch.nn as nn
import torch.nn.functional as F
from CustomLayers.Convolution import Convolution
from CustomLayers.Activation import RectifiedLinearUnit
from CustomLayers.MaxPool import MaxPool2d
from CustomLayers.Loss import CrossEntropyLoss
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
        self.conv3 = Convolution(filters=64, kernel_size=3, input_shape=(32, 54, 54))
        self.activation3 = RectifiedLinearUnit()
        self.pool3 = MaxPool2d(pool_size=2, stride=2)    # Output Shape: (64, 26, 26)
        self.flatten = Flatten()
        self.dense1 = Dense(64*26*26, 128)
        self.activation4 = RectifiedLinearUnit()
        self.dropout = Dropout(p=0.5)
        self.dense2 = Dense(128, 26)
        self.activation5 = LogSoftmax()
