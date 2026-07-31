"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn

from ..layers import ActivationFactory as AF
from .se_blocks import SEBlock1d


def _conv1(in_channels: int, out_channels: int, bias: bool = False) -> nn.Conv1d:
    """Build a 1x1 convolution layer.

    Args:
      in_channels: Number of input channels.
      out_channels: Number of output channels.
      bias: Whether to use a bias term.

    Returns:
      A 1x1 ``nn.Conv1d`` module.
    """
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)


def _dwconvk(
    channels: int, kernel_size: int, stride: int = 1, bias: bool = False
) -> nn.Conv1d:
    """Build a depth-wise 1D convolution with symmetric padding.

    Args:
      channels: Number of channels.
      kernel_size: Convolution kernel size.
      stride: Convolution stride.
      bias: Whether to use a bias term.

    Returns:
      A depth-wise ``nn.Conv1d`` module.
    """
    return nn.Conv1d(
        channels,
        channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size - 1) // 2,
        groups=channels,
        bias=bias,
        padding_mode="zeros",
    )


def _make_downsample(in_channels: int, out_channels: int, stride: int) -> nn.Conv1d:
    """Build the residual downsampling projection.

    Args:
      in_channels: Number of input channels.
      out_channels: Number of output channels.
      stride: Downsampling stride.

    Returns:
      A 1x1 ``nn.Conv1d`` module used to match residual dimensions.
    """
    return _conv1(in_channels, out_channels, stride, bias=True)


class ConformerConvBlock(nn.Module):
    """Convolutional block for conformer introduced at
        https://arxiv.org/pdf/2005.08100.pdf

        This includes some optional extra features
        not included in the original paper:
           - Squeeze-Excitation after depthwise-conv
           - Allows downsampling in time dimension
           - Allows choosing activation and layer normalization type

    Attributes:
       num_channels : number of input/output channels
       kernel_size: kernel_size for depth-wise conv
       stride: stride for depth-wise conv
       activation: activation specification accepted by ``ActivationFactory``.
       norm_layer: normalization layer constructor, if ``None`` it uses
                   ``BatchNorm1d``.
       dropout_rate: dropout rate
       se_r:         Squeeze-Excitation compression ratio,
                     if None it doesn't use Squeeze-Excitation
    """

    def __init__(
        self,
        num_channels: int,
        kernel_size: int,
        stride: int = 1,
        activation: Union[str, Dict[str, Any]] = "swish",
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout_rate: float = 0,
        se_r: Optional[int] = None,
    ) -> None:
        """Initialize the convolutional conformer block.

        Args:
          num_channels: Input and output channel dimension.
          kernel_size: Kernel size for the depth-wise convolution.
          stride: Stride for the depth-wise convolution and residual path.
          activation: Activation specification accepted by ``ActivationFactory``.
          norm_layer: Normalization layer constructor. Defaults to
            ``nn.BatchNorm1d``.
          dropout_rate: Dropout probability applied after the projection.
          se_r: Optional squeeze-excitation reduction ratio.
        """
        super().__init__()
        self.num_channels = (num_channels,)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout_rate = dropout_rate
        self.act = AF.create(activation)
        self.se_r = se_r
        self.has_se = se_r is not None and se_r > 1

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.layer_norm = nn.LayerNorm(num_channels)
        # expansion phase
        self.conv_exp = _conv1(num_channels, 2 * num_channels, bias=True)

        # depthwise conv phase
        self.conv_dw = _dwconvk(num_channels, kernel_size, stride=stride, bias=False)
        self.norm_dw = norm_layer(num_channels, momentum=0.01, eps=1e-3)
        if self.has_se:
            self.se_layer = SEBlock1d(num_channels, se_r, activation)

        # final projection
        self.conv_proj = _conv1(num_channels, num_channels, bias=True)
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

        # when input and output dimensions are different, we adapt the dimensions using conv1x1
        self.downsample = None
        if stride != 1:
            self.downsample = _make_downsample(num_channels, num_channels, stride)

        self.context = stride * (kernel_size - 1) // 2

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the conformer convolution block.

        Args:
          x: Input tensor with shape ``(batch, channels, time)``.
          x_mask: Mask indicating the valid frames in the sequence with
                  shape = (batch, 1, time) or (batch, time)

        Returns:
          Tensor with shape ``(batch, channels, time_out)``.
        """
        residual = x

        # layer norm
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)

        # expansion + glu
        x = self.conv_exp(x)
        x = nn.functional.glu(x, dim=1)

        # depthwide conv phase
        x = self.act(self.norm_dw(self.conv_dw(x)))
        if self.has_se:
            x = self.se_layer(x, x_mask=x_mask)

        # final projection
        x = self.conv_proj(x)
        if self.dropout_rate > 0:
            x = self.dropout(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        return x
