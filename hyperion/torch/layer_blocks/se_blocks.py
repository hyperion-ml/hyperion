"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn
from torch.nn import Conv1d, Conv2d
from typing import Any, Optional

from ..layers import ActivationFactory as AF


class SEBlock2d(nn.Module):
    """Squeeze-excitation block 2d
        from https://arxiv.org/abs/1709.01507.

    Attributes:
      num_channels:      input/output channels.
      r:                 Squeeze-excitation compression ratio.
      activation:        Non-linear activation object, string of configuration dictionary.

    """

    def __init__(
        self,
        num_channels: int,
        r: int = 16,
        activation: Any = {"name": "relu", "inplace": True},
    ):
        """Initialize the 2d squeeze-excitation block.

        Args:
          num_channels: Number of input and output channels.
          r: Squeeze-excitation compression ratio.
          activation: Non-linear activation specification.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_channels, int(num_channels / r), kernel_size=1, bias=False
        )
        self.act = AF.create(activation)
        self.conv2 = nn.Conv2d(
            int(num_channels / r), num_channels, kernel_size=1, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def _standardize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Normalize the mask shape for 2d operations.

        Args:
          mask: Input mask tensor.

        Returns:
          Mask reshaped to a broadcastable 2d layout.
        """
        if mask.dim() == 2:
            return mask.view(mask.size(0), 1, 1, mask.size(-1))

        if mask.dim() == 3:
            return mask.unsqueeze(1)

        return mask

    def compute_scale_logits(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the scale before the sigmoid.

        Args:
          x: input tensor with shape = (batch, channels, height, width).
          x_mask: Binary mask indicating which spatial dimensions are valid of
                  shape=(batch, time), (batch, 1, time), (batch, height, width)

        Returns:
          Tensor with shape = (batch, channels, height, width).
        """
        if x_mask is None:
            z = torch.mean(x, dim=(2, 3), keepdim=True)
        else:
            x_mask = self._standardize_mask(x_mask)
            total = torch.mean(x_mask, dim=(2, 3), keepdim=True)
            z = torch.mean(x * x_mask, dim=(2, 3), keepdim=True) / total

        return self.conv2(self.act(self.conv1(z)))

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function.

        Args:
          x: input tensor with shape = (batch, channels, height, width).
          x_mask: Binary mask indicating which spatial dimensions are valid of
                  shape=(batch, time), (batch, 1, time), (batch, height, width)

        Returns:
          Tensor with shape = (batch, channels, height, width).
        """
        scale_logits = self.compute_scale_logits(x, x_mask)
        scale = self.sigmoid(scale_logits)
        y = scale * x
        return y


class TSEBlock2d(nn.Module):
    """From https://arxiv.org/abs/1709.01507
    Modified to do pooling only in time dimension.

    Attributes:
      num_channels:      input/output channels.
      num_feats:         Number of features in dimension 2.
      r:                 Squeeze-excitation compression ratio.
      activation:        Non-linear activation object, string of configuration dictionary.

    """

    def __init__(
        self,
        num_channels: int,
        num_feats: int,
        r: int = 16,
        activation: Any = {"name": "relu", "inplace": True},
    ):
        """Initialize the time-only squeeze-excitation block.

        Args:
          num_channels: Number of input channels.
          num_feats: Number of features in the frequency dimension.
          r: Squeeze-excitation compression ratio.
          activation: Non-linear activation specification.
        """
        super().__init__()
        self.num_channels_1d = num_channels * num_feats
        self.conv1 = nn.Conv2d(
            self.num_channels_1d,
            int(self.num_channels_1d / r),
            kernel_size=1,
            bias=False,
        )
        self.act = AF.create(activation)
        self.conv2 = nn.Conv2d(
            int(self.num_channels_1d / r),
            self.num_channels_1d,
            kernel_size=1,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def _standardize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Normalize the mask shape for time-only pooling.

        Args:
          mask: Input mask tensor.

        Returns:
          Mask reshaped to a broadcastable 4d layout.
        """
        if mask.dim() == 2:
            return mask.view(mask.size(0), 1, 1, mask.size(-1))

        if mask.dim() == 3:
            return mask.unsqueeze(1)

        return mask

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function.

        Args:
          x: input tensor with shape = (batch, channels, height, width).
          x_mask: Binary mask indicating which spatial dimensions are valid of
                  shape=(batch, time), (batch, 1, time), (batch, height, width)

        Returns:
          Tensor with shape = (batch, channels, height, width).
        """
        num_feats = x.shape[2]
        num_channels = x.shape[1]
        if x_mask is None:
            z = torch.mean(x, dim=-1, keepdim=True)
        else:
            x_mask = self._standardize_mask(x_mask)
            total = torch.mean(x_mask, dim=-1, keepdim=True)
            z = torch.mean(x * x_mask, dim=-1, keepdim=True) / total

        z = z.view(-1, self.num_channels_1d, 1, 1)
        scale = self.sigmoid(self.conv2(self.act(self.conv1(z))))
        scale = scale.view(-1, num_channels, num_feats, 1)
        y = scale * x
        return y


class FwSEBlock2d(SEBlock2d):
    """frequency-wise Squeeze-excitation block 2d

    Attributes:
      num_feats:      input/output features.
      r:                 Squeeze-excitation compression ratio.
      activation:        Non-linear activation object, string of configuration dictionary.

    """

    def __init__(
        self,
        num_feats: int,
        r: int = 16,
        activation: Any = {"name": "relu", "inplace": True},
    ):
        """Initialize the frequency-wise squeeze-excitation block.

        Args:
          num_feats: Number of input and output features.
          r: Squeeze-excitation compression ratio.
          activation: Non-linear activation specification.
        """
        super().__init__(num_feats, r, activation)

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function.

        Args:
          x: input tensor with shape = (batch, channels, height, width).
          x_mask: Binary mask indicating which spatial dimensions are valid of
                  shape=(batch, time), (batch, 1, time)
        Returns:
          Tensor with shape = (batch, channels, height, width).
        """
        x = x.transpose(1, 2)
        y = super().forward(x, x_mask)
        y = y.transpose(1, 2).contiguous()
        return y


class CFwSEBlock2d(nn.Module):
    """2-d channel and frequency wise squeeze-excitation block

    Attributes:
      num_channels:      input/output channels.
      num_feats:         Number of features in dimension 2.
      r:                 Squeeze-excitation compression ratio.
      activation:        Non-linear activation object, string of configuration dictionary.

    """

    def __init__(
        self,
        num_channels: int,
        num_feats: int,
        r: int = 16,
        activation: Any = {"name": "relu", "inplace": True},
    ):
        """Initialize the combined channel and frequency SE block.

        Args:
          num_channels: Number of input and output channels.
          num_feats: Number of features in the frequency dimension.
          r: Squeeze-excitation compression ratio.
          activation: Non-linear activation specification.
        """
        super().__init__()
        self.cw_se = SEBlock2d(num_channels, r, activation)
        # the bottlenet features will have at least dimension 4
        if num_feats // r < 4:
            r = num_feats // 4

        self.fw_se = SEBlock2d(num_feats, r, activation)

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function.

        Args:
          x: input tensor with shape = (batch, channels, height, width).
          x_mask: Binary mask indicating which spatial dimensions are valid of
                  shape=(batch, time), (batch, 1, time)
        Returns:
          Tensor with shape = (batch, channels, height, width).
        """
        cw_scale_logits = self.cw_se.compute_scale_logits(x, x_mask)
        fw_scale_logits = self.fw_se.compute_scale_logits(
            x.transpose(1, 2), x_mask
        ).transpose(1, 2)
        scale_logits = cw_scale_logits + fw_scale_logits
        scale = torch.sigmoid(scale_logits)
        y = scale * x
        return y


class SEBlock1d(nn.Module):
    """1d Squeeze Excitation version of
    https://arxiv.org/abs/1709.01507

    Attributes:
      num_channels:      input/output channels.
      r:                 Squeeze-excitation compression ratio.
      activation:        Non-linear activation object, string of configuration dictionary.
    """

    def __init__(
        self,
        num_channels: int,
        r: int = 16,
        activation: Any = {"name": "relu", "inplace": True},
    ):
        """Initialize the 1d squeeze-excitation block.

        Args:
          num_channels: Number of input and output channels.
          r: Squeeze-excitation compression ratio.
          activation: Non-linear activation specification.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(
            num_channels, int(num_channels / r), kernel_size=1, bias=False
        )
        self.act = AF.create(activation)
        self.conv2 = nn.Conv1d(
            int(num_channels / r), num_channels, kernel_size=1, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def _standardize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Normalize the mask shape for 1d operations.

        Args:
          mask: Input mask tensor.

        Returns:
          Mask reshaped to a broadcastable 3d layout.
        """
        if mask.dim() == 2:
            return mask.unsqueeze(1)

        return mask

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function.

        Args:
          x: input tensor with shape = (batch, channels, time).
          x_mask: Binary mask indicating which spatial dimensions are valid of
                  shape=(batch, time), (batch, 1, time)

        Returns:
          Tensor with shape = (batch, channels, time).
        """
        if x_mask is None:
            z = torch.mean(x, dim=2, keepdim=True)
        else:
            x_mask = self._standardize_mask(x_mask)
            total = torch.mean(x_mask, dim=-1, keepdim=True)
            z = torch.mean(x * x_mask, dim=-1, keepdim=True) / total

        scale = self.sigmoid(self.conv2(self.act(self.conv1(z))))
        y = scale * x
        return y


# aliases to mantein backwards compatibility
SEBlock2D = SEBlock2d
TSEBlock2D = TSEBlock2d
