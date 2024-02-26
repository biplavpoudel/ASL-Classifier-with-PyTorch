import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Dense(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias',None)
        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


