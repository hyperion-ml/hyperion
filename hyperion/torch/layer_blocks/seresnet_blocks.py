"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.nn import BatchNorm2d, Conv2d, Dropout2d, Linear

from ..layers import ActivationFactory as AF
from .resnet_blocks import ResNetBasicBlock, ResNetBNBlock
from .se_blocks import CFwSEBlock2d, FwSEBlock2d, SEBlock2d, TSEBlock2d

ActivationType = Union[nn.Module, Dict[str, Any]]
VALID_SE_TYPES = {"t-se", "cw-se", "fw-se", "cfw-se"}


def _require_num_feats(num_feats: Optional[int], se_type: str) -> int:
    if num_feats is None:
        raise ValueError(
            f"num_feats must be provided when se_type='{se_type}' or freq_pos_enc=True"
        )
    return num_feats


def _validate_se_type(se_type: str) -> str:
    if se_type not in VALID_SE_TYPES:
        raise ValueError(
            f"invalid se_type='{se_type}', expected one of {sorted(VALID_SE_TYPES)}"
        )
    return se_type


class SEResNetBasicBlock(ResNetBasicBlock):
    """Squeeze-excitation ResNet basic block.

    Attributes:
      in_channels:       input channels.
      channels:          output channels.
      activation:        Non-linear activation object, string of configuration dictionary.
      stride:            downsampling stride of the convs.
      dropout_rate:      dropout rate.
      groups:            number of groups in the convolutions.
      dilation:          dilation factor of the conv. kernels.
      norm_layer:        normalization layer constructor, if None BatchNorm2d is used.
      norm_before:       if True, normalization layer is before the activation, after otherwise.
      se_r:              squeeze-excitation compression ratio.
      se_type:           type of squeeze excitation in [t-se, cw-se, fw-se, cfw-se]
      freq_pos_enc:      use frequency wise positional encoder.
      num_feats:         Number of features in dimension 2, needed if
                         se_type in [t-se, fw-se, cfw-se] or freq_pos_enc=True.
      time_se:           (legacy deprecated) If true, use t-se
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        activation: ActivationType = {"name": "relu", "inplace": True},
        stride: int = 1,
        dropout_rate: float = 0,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        norm_before: bool = True,
        se_r: int = 16,
        se_type: str = "cw-se",
        freq_pos_enc: bool = False,
        num_feats: Optional[int] = None,
        time_se: bool = False,
    ) -> None:
        """Create a squeeze-excitation ResNet basic block.

        Args:
          in_channels: input channels.
          channels: output channels.
          activation: activation specification.
          stride: downsampling stride.
          dropout_rate: dropout probability.
          groups: number of convolution groups.
          dilation: convolution dilation.
          norm_layer: normalization layer constructor, if None BatchNorm2d is used.
          norm_before: if True normalization is before activation, otherwise after.
          se_r: squeeze-excitation reduction ratio.
          se_type: squeeze-excitation variant.
          freq_pos_enc: if True, add frequency positional encoding.
          num_feats: number of feature bins required by positional encoding.
          time_se: legacy alias for ``se_type='t-se'``.
        """
        super().__init__(
            in_channels,
            channels,
            activation=activation,
            stride=stride,
            dropout_rate=dropout_rate,
            groups=groups,
            dilation=dilation,
            norm_layer=norm_layer,
            norm_before=norm_before,
            freq_pos_enc=freq_pos_enc,
            num_feats=num_feats,
        )

        if time_se:
            se_type = "t-se"
        se_type = _validate_se_type(se_type)

        if se_type == "t-se":
            num_feats = _require_num_feats(num_feats, se_type)
            self.se_layer = TSEBlock2d(channels, num_feats, se_r, activation)
        elif se_type == "cw-se":
            self.se_layer = SEBlock2d(channels, se_r, activation)
        elif se_type == "fw-se":
            num_feats = _require_num_feats(num_feats, se_type)
            self.se_layer = FwSEBlock2d(num_feats, se_r, activation)
        elif se_type == "cfw-se":
            num_feats = _require_num_feats(num_feats, se_type)
            self.se_layer = CFwSEBlock2d(channels, num_feats, se_r, activation)

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the squeeze-excitation residual block.

        Args:
          x: input tensor with shape=(batch, in_channels, height, width).
          x_mask: Binary mask indicating which spatial dimensions are valid of
                  shape=(batch, time), (batch, 1, time), (batch, height, width)

        Returns:
          Tensor with shape=(batch, out_channels, height, width).
        """
        residual = x

        if self.pos_enc is not None:
            x = self.pos_enc(x)

        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)

        x = self.act1(x)

        if not self.norm_before:
            x = self.bn1(x)

        x = self.conv2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.norm_before:
            x = self.bn2(x)
            x = self.se_layer(x, x_mask=x_mask)
            x += residual
            x = self.act2(x)
        else:
            x = self.act2(x)
            x = self.bn2(x)
            x = self.se_layer(x, x_mask=x_mask)
            x += residual

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class SEResNetBNBlock(ResNetBNBlock):
    """Squeeze-excitation ResNet bottleneck block.

    Attributes:
      in_channels:       input channels.
      channels:          channels in bottleneck layer when width_factor=1.
      activation:        Non-linear activation object, string of configuration dictionary.
      stride:            downsampling stride of the convs.
      dropout_rate:      dropout rate.
      groups:            number of groups in the convolutions.
      dilation:          dilation factor of the conv. kernels.
      norm_layer:        normalization layer constructor, if None BatchNorm2d is used.
      norm_before:       if True, normalization layer is before the activation, after otherwise.
      se_r=None:         squeeze-excitation compression ratio.
      se_type:           type of squeeze excitation in [t-se, cw-se, fw-se, cfw-se]
      freq_pos_enc:      use frequency wise positional encoder.
      num_feats:         Number of features in dimension 2, needed if
                         se_type in [t-se, fw-se, cfw-se] or freq_pos_enc=True.
      time_se:           (legacy deprecated) If true, use t-se
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        activation: ActivationType = {"name": "relu", "inplace": True},
        stride: int = 1,
        dropout_rate: float = 0,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        norm_before: bool = True,
        se_r: int = 16,
        se_type: str = "cw-se",
        freq_pos_enc: bool = False,
        num_feats: Optional[int] = None,
        time_se: bool = False,
    ) -> None:
        """Create a squeeze-excitation ResNet bottleneck block.

        Args:
          in_channels: input channels.
          channels: bottleneck channels.
          activation: activation specification.
          stride: downsampling stride.
          dropout_rate: dropout probability.
          groups: number of convolution groups.
          dilation: convolution dilation.
          norm_layer: normalization layer constructor, if None BatchNorm2d is used.
          norm_before: if True normalization is before activation, otherwise after.
          se_r: squeeze-excitation reduction ratio.
          se_type: squeeze-excitation variant.
          freq_pos_enc: if True, add frequency positional encoding.
          num_feats: number of feature bins required by positional encoding.
          time_se: legacy alias for ``se_type='t-se'``.
        """
        super().__init__(
            in_channels,
            channels,
            activation=activation,
            stride=stride,
            dropout_rate=dropout_rate,
            groups=groups,
            dilation=dilation,
            norm_layer=norm_layer,
            norm_before=norm_before,
            freq_pos_enc=freq_pos_enc,
            num_feats=num_feats,
        )

        if time_se:
            se_type = "t-se"
        se_type = _validate_se_type(se_type)

        se_channels = channels * self.expansion
        if se_type == "t-se":
            num_feats = _require_num_feats(num_feats, se_type)
            self.se_layer = TSEBlock2d(se_channels, num_feats, se_r, activation)
        elif se_type == "cw-se":
            self.se_layer = SEBlock2d(se_channels, se_r, activation)
        elif se_type == "fw-se":
            num_feats = _require_num_feats(num_feats, se_type)
            self.se_layer = FwSEBlock2d(num_feats, se_r, activation)
        elif se_type == "cfw-se":
            num_feats = _require_num_feats(num_feats, se_type)
            self.se_layer = CFwSEBlock2d(se_channels, num_feats, se_r, activation)

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the squeeze-excitation bottleneck residual block.

        Args:
          x: input tensor with shape=(batch, in_channels, height, width).
          x_mask: Binary mask indicating which spatial dimensions are valid of
                  shape=(batch, time), (batch, 1, time), (batch, height, width)

        Returns:
          Tensor with shape=(batch, out_channels, height, width).
        """
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.pos_enc is not None:
            x = self.pos_enc(x)

        x = self.conv1(x)
        if self.norm_before:
            x = self.bn1(x)
        x = self.act1(x)
        if not self.norm_before:
            x = self.bn1(x)

        x = self.conv2(x)
        if self.norm_before:
            x = self.bn2(x)
        x = self.act2(x)
        if not self.norm_before:
            x = self.bn2(x)

        x = self.conv3(x)
        if self.norm_before:
            x = self.bn3(x)
            x = self.se_layer(x, x_mask=x_mask)
            x += residual
            x = self.act3(x)
        else:
            x = self.act3(x)
            x = self.bn3(x)
            x = self.se_layer(x, x_mask=x_mask)
            x += residual

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x
