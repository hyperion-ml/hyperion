"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from ..layers import ActivationFactory as AF


class TransformerConv2dSubsampler(nn.Module):
    """Convolutional 2D subsampling block for transformer encoders.

    Attributes:
      stride: Total stride of the subsampler.
      time_dim: Index of the time dimension in the input tensor.
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        hid_act: Any,
        stride: int = 4,
        pos_enc: Optional[nn.Module] = None,
        time_dim: int = 1,
    ) -> None:
        """Initializes the 2D convolutional transformer subsampler.

        Args:
          in_feats: Input feature dimension.
          out_feats: Output feature dimension (transformer model dimension).
          hid_act: Hidden activation specification accepted by ``ActivationFactory``.
          stride: Total temporal subsampling stride (supported values: 1, 2, 4).
          pos_enc: Optional positional encoding module appended after projection.
          time_dim: Index of the time dimension in the input tensor.

        Returns:
          None.
        """
        super().__init__()
        if time_dim not in (1, 2, -1):
            raise ValueError(
                f"invalid time_dim={time_dim}, expected one of (1, 2, -1)"
            )
        self.time_dim = time_dim
        hid_act = AF.create(hid_act)
        self.stride = stride
        if stride == 4:
            stride_1 = 2
            stride_2 = 2
            hid_feats = out_feats * (((in_feats - 1) // 2 - 1) // 2)
        elif stride == 2:
            stride_1 = 2
            stride_2 = 1
            hid_feats = out_feats * ((in_feats - 1) // 2 - 2)
        elif stride == 1:
            stride_1 = 1
            stride_2 = 1
            hid_feats = out_feats * (in_feats - 4)
        else:
            raise NotImplementedError(
                f"Valid TransformerConv2dSubsampler stride==1,2,4 !={stride}"
            )

        self.conv = nn.Sequential(
            nn.Conv2d(1, out_feats, 3, stride_1, padding=(0, 1)),
            hid_act,
            nn.Conv2d(out_feats, out_feats, 3, stride_2, padding=(0, 1)),
            hid_act,
        )

        linear = nn.Linear(hid_feats, out_feats)
        if pos_enc is None:
            self.out = linear
        else:
            self.out = nn.Sequential(linear, pos_enc)

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward function.

        Args:
          x: Input tensor. If ``time_dim == 1``, expected shape is
            ``(batch, time, feat)``; otherwise shape is ``(batch, feat, time)``.
          x_mask: Optional attention/sequence mask with time on the last axis.

        Returns:
          Tuple containing:
          output tensor of shape ``(batch, subsampled_time, out_feats)`` and
          the subsampled mask (or ``None`` if ``x_mask`` is ``None``).
        """
        if self.time_dim == 1:
            x = x.transpose(1, 2)

        x = x.unsqueeze(1)  # (b, c, f, t)
        x = self.conv(x)
        b, c, f, t = x.size()
        x = self.out(x.contiguous().view(b, c * f, t).transpose(1, 2))
        if x_mask is None:
            return x, None

        return x, x_mask[..., :: self.stride]


class TransformerConv1dSubsampler(nn.Module):
    """Convolutional 1D subsampling block for transformer encoders.

    Attributes:
      stride: Total stride of the subsampler.
      time_dim: Index of the time dimension in the input tensor.
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        hid_act: Any,
        stride: int = 4,
        pos_enc: Optional[nn.Module] = None,
        time_dim: int = 1,
    ) -> None:
        """Initializes the 1D convolutional transformer subsampler.

        Args:
          in_feats: Input feature dimension.
          out_feats: Output feature dimension (transformer model dimension).
          hid_act: Hidden activation specification accepted by ``ActivationFactory``.
          stride: Total temporal subsampling stride (supported values: 1, 2, 4).
          pos_enc: Optional positional encoding module appended after projection.
          time_dim: Index of the time dimension in the input tensor.

        Returns:
          None.
        """
        super().__init__()
        if time_dim not in (1, 2, -1):
            raise ValueError(
                f"invalid time_dim={time_dim}, expected one of (1, 2, -1)"
            )
        self.time_dim = time_dim
        hid_act = AF.create(hid_act)
        self.stride = stride
        if stride == 4:
            stride_1 = 2
            stride_2 = 2
        elif stride == 2:
            stride_1 = 2
            stride_2 = 1
        elif stride == 1:
            stride_1 = 1
            stride_2 = 1
        else:
            raise NotImplementedError(
                f"Valid TransformerConv1dSubsampler stride==1,2,4 !={stride}"
            )

        self.conv = nn.Sequential(
            nn.Conv1d(in_feats, out_feats, 3, stride_1, padding=1),
            hid_act,
            nn.Conv1d(out_feats, out_feats, 3, stride_2, padding=1),
            hid_act,
        )

        linear = nn.Linear(out_feats, out_feats)
        if pos_enc is None:
            self.out = linear
        else:
            self.out = nn.Sequential(linear, pos_enc)

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward function.

        Args:
          x: Input tensor. If ``time_dim == 1``, expected shape is
            ``(batch, time, feat)``; otherwise shape is ``(batch, feat, time)``.
          x_mask: Optional attention/sequence mask with time on the last axis.

        Returns:
          Tuple containing:
          output tensor of shape ``(batch, subsampled_time, out_feats)`` and
          the subsampled mask (or ``None`` if ``x_mask`` is ``None``).
        """
        if self.time_dim == 1:
            x = x.transpose(1, 2)

        x = self.conv(x)
        x = self.out(x.transpose(1, 2))
        if x_mask is None:
            return x, None

        return x, x_mask[..., :: self.stride]
