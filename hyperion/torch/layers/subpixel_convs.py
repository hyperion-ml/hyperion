"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
#

import torch
import torch.nn as nn
from typing import Callable, Tuple, Union


Padding1d = Union[int, Tuple[int]]
Padding2d = Union[int, Tuple[int, int]]
KernelSize1d = int
KernelSize2d = Union[int, Tuple[int, int]]
Initializer = Callable[[torch.Tensor], torch.Tensor]


class SubPixelConv1d(nn.Module):
    """Implements a SubPixel Convolution in 1d proposed in:
       https://arxiv.org/abs/1609.05158

    Attributes:
      in_channels:  Number of input channels.
      out_channels: Number of output channels.
      kernel_size:  Kernel size.
      stride:       Upsampling factor.
      padding:      Int or Int Tuple with the number of left/right padding samples
      dilation:     Kernel dilation.
      groups:       Number of groups in the convolution.
      bias:         If true, the convolution has bias.
      padding_mode: Padding mode in ['zeros', 'reflect', 'replicate' or 'circular'].

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: KernelSize1d,
        stride: int = 1,
        padding: Padding1d = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
      stride:       Upsampling factor.
      padding:      Int or Int Tuple with the number of left/right padding samples
      dilation:     Kernel dilation.
      groups:       Number of groups in the convolution.
      bias:         If true, the convolution has bias.
      padding_mode: Padding mode in ['zeros', 'reflect', 'replicate' or 'circular'].

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: KernelSize2d,
        stride: int = 1,
        padding: Padding2d = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies subpixel convolution 2d.

        Args:
          x: Input tensor with shape = (batch, in_channels, in_H, in_W)

        Returns:
          Output tensor with shape = (batch, out_channels, out_H, out_W)
        """
        x = self.conv(x)
        if self.stride == 1:
            return x

        return self.pixel_shuffle(x)


def ICNR2d(
    tensor: torch.Tensor,
    stride: int = 2,
    initializer: Initializer = nn.init.kaiming_normal,
) -> None:
    """Initialization method
    "Initialization to Convolution Nearest neighbours Resize (ICNR)"
    for subpixel convolutions described in
    "Andrew Aitken et al. (2017) Checkerboard artifact free sub-pixel convolution"
        https://arxiv.org/abs/1707.02937

    Args:
        tensor: torch.Tensor containing the conv weights
        stride: subpixel conv stride
        initializer: initializer to be used for sub-kernel initialization
    Examples:
        >>> conv = SubPixelConv2d(in_channels, out_channels, kernel_size=3, stride=upscale)
        >>> ICNR2d(conv.conv.weight, stride=upscale)

    """
    with torch.no_grad():
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        factor = stride ** 2
        if tensor.shape[0] % factor != 0:
            raise ValueError(
                f"tensor.shape[0] ({tensor.shape[0]}) must be divisible by stride**2 ({factor})"
            )
        new_shape = [tensor.shape[0] // factor] + list(tensor.shape[1:])
        subkernel = torch.zeros(
            new_shape, device=tensor.device, dtype=tensor.dtype
        )
        subkernel = initializer(subkernel)
        subkernel = subkernel.transpose(0, 1).contiguous()
        subkernel = subkernel.view(subkernel.shape[0], subkernel.shape[1], -1)

        kernel = subkernel.repeat(1, 1, stride ** 2)

        transposed_shape = [tensor.shape[1], tensor.shape[0]] + list(tensor.shape[2:])
        kernel = kernel.contiguous().view(transposed_shape).transpose(0, 1).contiguous()
        tensor.copy_(kernel)


def ICNR1d(
    tensor: torch.Tensor,
    stride: int = 2,
    initializer: Initializer = nn.init.kaiming_normal,
) -> None:
    """1d version of the initialization method
    "Initialization to Convolution Nearest neighbours Resize (ICNR)"
    for subpixel convolutions described in
    "Andrew Aitken et al. (2017) Checkerboard artifact free sub-pixel convolution"
        https://arxiv.org/abs/1707.02937

    Args:
        tensor: torch.Tensor containing the conv weights
        stride: subpixel conv stride
        initializer: initializer to be used for sub-kernel initialization
    Examples:
        >>> conv = SubPixelConv1d(in_channels, out_channels, kernel_size=3, stride=upscale)
        >>> ICNR1d(conv.conv.weight, stride=upscale)

    """
    with torch.no_grad():
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if tensor.shape[0] % stride != 0:
            raise ValueError(
                f"tensor.shape[0] ({tensor.shape[0]}) must be divisible by stride ({stride})"
            )
        new_shape = [tensor.shape[0] // stride] + list(tensor.shape[1:])
        subkernel = torch.zeros(
            new_shape, device=tensor.device, dtype=tensor.dtype
        )
        subkernel = initializer(subkernel)
        subkernel = subkernel.transpose(0, 1).contiguous()
        subkernel = subkernel.view(subkernel.shape[0], subkernel.shape[1], -1)

        kernel = subkernel.repeat(1, 1, stride)

        transposed_shape = (tensor.shape[1], tensor.shape[0], tensor.shape[2])
        kernel = kernel.contiguous().view(transposed_shape).transpose(0, 1).contiguous()
        tensor.copy_(kernel)
