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


class Swish6(nn.Module):
    """Swish activation class, clamped to 6
    y = min(x, 6) * sigmoid(min(x,6))
    """

    def forward(self, x):
        return SwishImplementation.apply(x.clamp(max=6))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "{}()".format(self.__class__.__name__)
        return s


class DoubleSwishImplementation(torch.autograd.Function):
    """ Implementation for DoubleSwish Activation from
    https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7/scaling.py    

    f(x) = x * torch.sigmoid(x-1) = swish(swish(x)), 
         where swish(x) =  x * sigmoid(x).

    Memory-efficient derivative computation:
     f'(x) = =  x * s'(x) + x' * s(x) = x * s'(x) + s(x).
         where s(x) = simoid(x), and s'(x) = s(x) * (1-s(x)).
     
     f'(x) = x * s(x) * (1-s(x)) + s(x) = f(x) * (1-s(x)) + s(x)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        requires_grad = x.requires_grad
        x_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)

        s = torch.sigmoid(x - 1.0)
        y = x * s

        if requires_grad:
            deriv = y * (1 - s) + s
            # notes on derivative of x * sigmoid(x - 1):
            # https://www.wolframalpha.com/input?i=d%2Fdx+%28x+*+sigmoid%28x-1%29%29
            # min \simeq -0.043638.  Take floor as -0.043637 so it's a lower bound
            # max \simeq 1.1990.   Take ceil to be 1.2 so it's an upper bound.
            # the combination of "+ torch.rand_like(deriv)" and casting to torch.uint8 (which
            # floors), should be expectation-preserving.
            floor = -0.043637
            ceil = 1.2
            d_scaled = (deriv - floor) * (255.0 / (ceil - floor)) + torch.rand_like(
                deriv
            )
            d_int = d_scaled.to(torch.uint8)
            ctx.save_for_backward(d_int)
        if x_dtype == torch.float16 or torch.is_autocast_enabled():
            y = y.to(torch.float16)
        return y

    @staticmethod
    def backward(ctx, y_grad: torch.Tensor) -> torch.Tensor:
        (d,) = ctx.saved_tensors
        # the same constants as used in forward pass.
        floor = -0.043637
        ceil = 1.2
        d = d * ((ceil - floor) / 255.0) + floor
        return y_grad * d


class DoubleSwish(torch.nn.Module):
    """ DoubleSwish activation
    f(x) = x * torch.sigmoid(x-1) = swish(swish(x)), 
         where swish(x) =  x * sigmoid(x).        
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return (x * torch.sigmoid(x - 1.0)).clamp(max=6)

        return DoubleSwishImplementation.apply(x)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "{}()".format(self.__class__.__name__)
        return s


class DoubleSwish6(torch.nn.Module):
    """ DoubleSwish activation clamped to 6
    x = min(x, 6)
    f(x) = x * torch.sigmoid(x-1) = swish(swish(x)), 
         where swish(x) =  x * sigmoid(x).        
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(max=6)
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return (x * torch.sigmoid(x - 1.0)).clamp(max=6)

        return DoubleSwishImplementation.apply(x)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "{}()".format(self.__class__.__name__)
        return s
