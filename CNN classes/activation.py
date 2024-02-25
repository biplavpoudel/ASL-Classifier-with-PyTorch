# Here we define some activation functions classes
import numpy as np
import math
import torch

# class log_softmax():
    # def __init__(self):

def log_softmax(x):
    # x is a tensor of batch of prediction
    return x - x.exp().sum(axis=-1).log().unsqueeze(-1)


def softmax(x):
    return x.exp()/x.sum(axis=-1).unsqueeze(-1)
