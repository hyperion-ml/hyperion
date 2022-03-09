"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
#

import torch
import torch.nn as nn


class SubPixelConv1d(nn.Module):
    """Implements a SubPixel Convolution in 1d proposed in:
       https://arxiv.org/abs/1609.05158

    Attributes:
      in_channels:  Number of input channels.
      out_channels: Number of output channels.
      kernel_size:  Kernel size.
      stride:       Downsampling stride.
      padding:      Int or Int Tuple with the number of left/right padding samples
      dilation:     Kernel dilation.
      groups:       Number of groups in the convolution.
      bias:         If true, the convolution has bias.
      padding_mode: Padding mode in ['zeros', 'reflect', 'replicate' or 'circular'].

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            stride * out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x):
        """Applies subpixel convolution 1d.

        Args:
          x: Input tensor with shape = (batch, in_channels, in_time)

        Returns:
          Output tensor with shape = (batch, out_channels, out_time)
        """
        x = self.conv(x)
        if self.stride == 1:
            return x

        x = (
            x.view(-1, self.stride, self.out_channels, x.size(-1))
            .permute(0, 2, 3, 1)
            .reshape(-1, self.out_channels, x.size(-1) * self.stride)
        )
        return x


class SubPixelConv2d(nn.Module):
    """Implements a SubPixel Convolution in 2d proposed in:
       https://arxiv.org/abs/1609.05158

    Attributes:
      in_channels:  Number of input channels.
      out_channels: Number of output channels.
      kernel_size:  Kernel size.
      stride:       Downsampling stride.
      padding:      Int or Int Tuple with the number of left/right padding samples
      dilation:     Kernel dilation.
      groups:       Number of groups in the convolution.
      bias:         If true, the convolution has bias.
      padding_mode: Padding mode in ['zeros', 'reflect', 'replicate' or 'circular'].

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            (stride ** 2) * out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.stride = stride
        if stride > 1:
            self.pixel_shuffle = nn.PixelShuffle(self.stride)

    def forward(self, x):
        """Applies subpixel convolution 1d.

        Args:
          x: Input tensor with shape = (batch, in_channels, in_W, in_H)

        Returns:
          Output tensor with shape = (batch, out_channels, out_W, out_H)
        """
        x = self.conv(x)
        if self.stride == 1:
            return x

        return self.pixel_shuffle(x)


def ICNR2d(tensor, stride=2, initializer=nn.init.kaiming_normal):
    """Initialization method
    "Initialization to Convolution Nearest neighbours Resize (ICNR)"
    for subpixel convolutions described in
    described in "Andrew Aitken et al. (2017) Checkerboard artifact free sub-pixel convolution"
        https://arxiv.org/abs/1707.02937

    Args:
        tensor: torch.Tensor containing the conv weights
        stride: subpixel conv stride
        initializer: initizializer to be used for sub_kernel inizialization
    Examples:
        >>> conv = SubPixelConv2d(in_channels, out_channels, kernel_size=3, stride=upscale)
        >>> ICNR2d(conv_shuffle.weight, stride=upscale)

    """
    with torch.no_grad():
        new_shape = [int(tensor.shape[0] / (stride ** 2))] + list(tensor.shape[1:])
        subkernel = torch.zeros(new_shape)
        subkernel = initializer(subkernel)
        subkernel = subkernel.transpose(0, 1).contiguous()
        subkernel = subkernel.view(subkernel.shape[0], subkernel.shape[1], -1)

        kernel = subkernel.repeat(1, 1, stride ** 2)

        transposed_shape = [tensor.shape[1], tensor.shape[0]] + list(tensor.shape[2:])
        kernel = kernel.contiguous().view(transposed_shape).transpose(0, 1).contiguous()
        tensor.copy_(kernel)


def ICNR1d(tensor, stride=2, initializer=nn.init.kaiming_normal):
    """1d version of the initialization method
    "Initialization to Convolution Nearest neighbours Resize (ICNR)"
    for subpixel convolutions described in
    described in "Andrew Aitken et al. (2017) Checkerboard artifact free sub-pixel convolution"
        https://arxiv.org/abs/1707.02937

    Args:
        tensor: torch.Tensor containing the conv weights
        stride: subpixel conv stride
        initializer: initizializer to be used for sub_kernel inizialization
    Examples:
        >>> conv = SubPixelConv1d(in_channels, out_channels, kernel_size=3, stride=upscale)
        >>> ICNR1d(conv_shuffle.weight, stride=upscale)

    """
    with torch.no_grad():
        new_shape = [int(tensor.shape[0] / stride)] + list(tensor.shape[1:])
        subkernel = torch.zeros(new_shape)
        subkernel = initializer(subkernel)
        subkernel = subkernel.transpose(0, 1).contiguous()
        subkernel = subkernel.view(subkernel.shape[0], subkernel.shape[1], -1)

        kernel = subkernel.repeat(1, 1, stride)

        transposed_shape = (tensor.shape[1], tensor.shape[0], tensor.shape[2])
        kernel = kernel.contiguous().view(transposed_shape).transpose(0, 1).contiguous()
        tensor.copy_(kernel)
