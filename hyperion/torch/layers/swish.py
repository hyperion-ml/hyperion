"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import torch
import torch.nn as nn


class SwishImplementation(torch.autograd.Function):
    """Implementation for Swish activation function."""

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    """Swish activation class:
    y = x * sigmoid(x)
    """

    def forward(self, x):
        return SwishImplementation.apply(x)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "{}()".format(self.__class__.__name__)
        return s
