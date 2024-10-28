"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale):
        """
        In the forward pass, we just pass through the input.
        ctx is used to store context information for the backward pass.
        """
        # Store the grad. scale value to be used during the backward pass
        ctx.scale = scale
        return x.clone()  # Return the input as it is

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass, we reverse the gradients by multiplying with -scale.
        """
        scale = ctx.scale
        grad_input = grad_output.neg() * scale  # Reverse the gradient
        # Return the reversed gradient and None for scale (no gradient for scale)
        return grad_input, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer
    Inverts and scale the gradients in the backward pass

    Attributes:
      scale: multiplies the gradient by the - scale in the backward pass
    """

    def __init__(self, scale: float = 1.0):
        super(GradientReversalLayer, self).__init__()
        self.scale = scale

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.scale)
