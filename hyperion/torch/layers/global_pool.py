"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf

from ..utils import seq_lengths_to_mask

SQRT_EPS = 1e-5
N_EPS = 1e-6
WindowArg = int | float
Shape = tuple[int, ...]


def _conv1(in_channels: int, out_channels: int, bias: bool = False) -> nn.Conv1d:
    """Builds a point-wise 1-D convolution layer.

    Args:
      in_channels: Number of input channels.
      out_channels: Number of output channels.
      bias: If ``True``, adds a learnable bias.

    Returns:
      A ``Conv1d`` layer with kernel size 1.
    """
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)


class _GlobalPool1d(nn.Module):
    """Abstract base class for global pooling layers on 1-D sequences.

    Attributes:
      dim: Pooling dimension.
      keepdim: If ``True``, preserves the pooled dimension in the output.
      size_multiplier: Output feature multiplier for subclasses.
    """

    def __init__(self, dim: int = -1, keepdim: bool = False) -> None:
        """Initializes the base global pooling layer.

        Args:
          dim: Pooling dimension.
          keepdim: If ``True``, preserves the pooled dimension in the output.

        Returns:
          ``None``.
        """
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.size_multiplier = 1

    def _standardize_weights(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Standardizes frame weights for element-wise multiplication with ``x``.

        Args:
          x: Input tensor.
          x_lengths: Optional sequence lengths used to build a mask when
            ``weights`` is ``None``.
          weights: Optional frame weights.

        Returns:
          A tensor broadcastable to ``x`` or ``None`` when no weighting is needed.
        """
        if weights is None:
            time_dim = self.dim if self.dim >= 0 else x.dim() + self.dim
            return seq_lengths_to_mask(
                x_lengths, x.size(self.dim), dtype=x.dtype, time_dim=time_dim
            )

        if weights.dim() == x.dim():
            return weights

        assert weights.dim() == 2
        shape = x.dim() * [1]
        shape[0] = weights.shape[0]
        shape[self.dim] = weights.shape[1]
        return weights.view(tuple(shape))

    def get_config(self) -> dict[str, Any]:
        """Returns a serializable layer configuration.

        Returns:
          Dictionary with pooling settings.
        """
        config = {"dim": self.dim, "keepdim": self.keepdim}
        return config

    def forward_slidwin(
        self, x: torch.Tensor, win_length: WindowArg, win_shift: WindowArg
    ) -> torch.Tensor:
        """Applies sliding-window pooling.

        Args:
          x: Input tensor.
          win_length: Window length in frames.
          win_shift: Window shift in frames.

        Returns:
          Tensor with pooled window statistics.
        """
        raise NotImplementedError()

    def _slidwin_pad(
        self,
        x: torch.Tensor,
        win_length: WindowArg,
        win_shift: WindowArg,
        snip_edges: bool,
    ) -> tuple[torch.Tensor, int]:
        """Pads ``x`` for sliding-window pooling.

        Args:
          x: Input tensor.
          win_length: Window length in frames.
          win_shift: Window shift in frames.
          snip_edges: If ``True``, drops incomplete windows at sequence edges.

        Returns:
          Tuple with padded tensor and number of output frames.
        """
        if snip_edges:
            num_frames = int(
                math.floor((x.size(-1) - win_length + win_shift) / win_shift)
            )
            if num_frames < 1:
                raise ValueError(
                    f"num_frames must be >= 1, got {num_frames} "
                    f"(length={x.size(-1)}, win_length={win_length}, "
                    f"win_shift={win_shift}, snip_edges={snip_edges})"
                )
            return nnf.pad(x, (1, 0), mode="constant"), num_frames

        assert (
            win_length >= win_shift
        ), "if win_length < win_shift snip-edges should be false"

        num_frames = int(round(x.size(-1) / win_shift))
        if num_frames < 1:
            raise ValueError(
                f"num_frames must be >= 1, got {num_frames} "
                f"(length={x.size(-1)}, win_length={win_length}, "
                f"win_shift={win_shift}, snip_edges={snip_edges})"
            )
        len_x = (num_frames - 1) * win_shift + win_length
        dlen_x = round(len_x - x.size(-1))
        pad_left = int(math.floor((win_length - win_shift) / 2))
        pad_right = int(dlen_x - pad_left)

        return nnf.pad(x, (pad_left + 1, pad_right), mode="reflect"), num_frames


class GlobalAvgPool1d(_GlobalPool1d):
    """Global average pooling in 1-D.

    Attributes:
      dim: Pooling dimension.
      keepdim: If ``True``, preserves the pooled dimension in the output.
      size_multiplier: Output feature multiplier, equal to ``1``.
    """

    def __init__(self, dim: int = -1, keepdim: bool = False) -> None:
        """Initializes global average pooling.

        Args:
          dim: Pooling dimension.
          keepdim: If ``True``, preserves the pooled dimension in the output.

        Returns:
          ``None``.
        """
        super().__init__(dim, keepdim)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Applies (optionally weighted) global average pooling.

        Args:
          x: Input tensor.
          x_lengths: Lengths in the pooling dimension. Used only when
            ``weights`` is ``None``.
          weights: Optional per-frame weights with shape ``(batch, time)`` or a
            shape already broadcastable to ``x``.

        Returns:
          Pooled tensor with the pooling dimension removed or preserved depending
          on ``keepdim``.
        """
        weights = self._standardize_weights(x, x_lengths, weights)
        if weights is None:
            y = torch.mean(x, dim=self.dim, keepdim=self.keepdim)
            return y

        xbar = torch.mean(weights * x, dim=self.dim, keepdim=self.keepdim)
        wbar = torch.mean(weights, dim=self.dim, keepdim=self.keepdim)
        return xbar / wbar

    def forward_slidwin(
        self,
        x: torch.Tensor,
        win_length: WindowArg,
        win_shift: WindowArg,
        snip_edges: bool = False,
    ) -> torch.Tensor:
        """Computes average pooling over sliding windows.

        Args:
          x: Input tensor.
          win_length: Window length in frames.
          win_shift: Window shift in frames.
          snip_edges: If ``True``, drops incomplete windows.

        Returns:
          Tensor of per-window means.
        """
        if isinstance(win_shift, int) and isinstance(win_length, int):
            return self._forward_slidwin_int(
                x, win_length, win_shift, snip_edges=snip_edges
            )

        # the window length and/or shift are floats
        return self._forward_slidwin_float(
            x, win_length, win_shift, snip_edges=snip_edges
        )

    def _pre_slidwin(
        self,
        x: torch.Tensor,
        win_length: WindowArg,
        win_shift: WindowArg,
        snip_edges: bool,
    ) -> tuple[torch.Tensor, Shape]:
        """Prepares cumulative-sum state for sliding-window mean computation.

        Args:
          x: Input tensor.
          win_length: Window length in frames.
          win_shift: Window shift in frames.
          snip_edges: If ``True``, drops incomplete windows.

        Returns:
          Tuple with flattened cumulative sums and output shape.
        """
        if self.dim != -1:
            x = x.transpose(self.dim, -1)

        x, num_frames = self._slidwin_pad(x, win_length, win_shift, snip_edges)
        out_shape = *x.shape[:-1], num_frames
        c_x = torch.cumsum(x, dim=-1).view(-1, x.shape[-1])
        return c_x, out_shape

    def _post_slidwin(self, m_x: torch.Tensor, x_shape: Shape) -> torch.Tensor:
        """Reshapes sliding-window output back to the original layout.

        Args:
          m_x: Flattened per-window means.
          x_shape: Target output shape.

        Returns:
          Tensor with dimensions restored to the expected layout.
        """
        m_x = m_x.view(x_shape)

        if self.dim != -1:
            m_x = m_x.transpose(self.dim, -1).contiguous()

        return m_x

    def _forward_slidwin_int(
        self, x: torch.Tensor, win_length: int, win_shift: int, snip_edges: bool
    ) -> torch.Tensor:
        """Sliding-window mean using integer frame boundaries.

        Args:
          x: Input tensor.
          win_length: Integer window length.
          win_shift: Integer window shift.
          snip_edges: If ``True``, drops incomplete windows.

        Returns:
          Tensor of per-window means.
        """
        c_x, out_shape = self._pre_slidwin(x, win_length, win_shift, snip_edges)
        num_frames = out_shape[-1]
        end = c_x[:, win_length : win_length + num_frames * win_shift : win_shift]
        start = c_x[:, : num_frames * win_shift : win_shift]
        m_x = (end - start) / win_length

        m_x = self._post_slidwin(m_x, out_shape)
        return m_x

    def _forward_slidwin_float(
        self,
        x: torch.Tensor,
        win_length: WindowArg,
        win_shift: WindowArg,
        snip_edges: bool,
    ) -> torch.Tensor:
        """Sliding-window mean using rounded float boundaries.

        Args:
          x: Input tensor.
          win_length: Window length (float or int).
          win_shift: Window shift (float or int).
          snip_edges: If ``True``, drops incomplete windows.

        Returns:
          Tensor of per-window means.
        """
        c_x, out_shape = self._pre_slidwin(x, win_length, win_shift, snip_edges)

        num_frames = out_shape[-1]
        m_x = torch.zeros(
            (c_x.shape[0], num_frames), dtype=c_x.dtype, device=c_x.device
        )
        k = 0
        for i in range(num_frames):
            k1 = int(round(k))
            k2 = int(round(k + win_length))
            m_x[:, i] = (c_x[:, k2] - c_x[:, k1]) / (k2 - k1)
            k += win_shift

        m_x = self._post_slidwin(m_x, out_shape)
        return m_x


class GlobalMeanStdPool1d(_GlobalPool1d):
    """Global mean and standard-deviation pooling in 1-D.

    Attributes:
      dim: Pooling dimension.
      keepdim: If ``True``, preserves the pooled dimension in the output.
      size_multiplier: Output feature multiplier, equal to ``2``.
    """

    def __init__(self, dim: int = -1, keepdim: bool = False) -> None:
        """Initializes global mean/std pooling.

        Args:
          dim: Pooling dimension.
          keepdim: If ``True``, preserves the pooled dimension in the output.

        Returns:
          ``None``.
        """
        super().__init__(dim, keepdim)
        self.size_multiplier = 2

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Applies (optionally weighted) global mean-std pooling.

        Args:
          x: Input tensor.
          x_lengths: Lengths in the pooling dimension. Used only when
            ``weights`` is ``None``.
          weights: Optional per-frame weights with shape ``(batch, time)`` or a
            shape already broadcastable to ``x``.

        Returns:
          Tensor containing concatenated mean and standard deviation statistics.
        """
        weights = self._standardize_weights(x, x_lengths, weights)
        if weights is None:
            mu = torch.mean(x, dim=self.dim, keepdim=True)
            delta = x - mu
            mu.squeeze_(dim=self.dim)

            # this can produce slightly negative variance when relu6 saturates in all time steps
            # add 1e-5 for stability
            s = torch.sqrt(
                torch.mean(delta.float() ** 2, dim=self.dim, keepdim=False).clamp(
                    min=SQRT_EPS
                )
            ).type_as(mu)

            mus = torch.cat((mu, s), dim=1)
            if self.keepdim:
                mus.unsqueeze_(dim=self.dim)

            return mus

        xbar = torch.mean(weights * x, dim=self.dim, keepdim=True)
        wbar = torch.mean(weights, dim=self.dim, keepdim=True)
        mu = xbar / wbar
        delta = (x - mu).float()
        var = torch.mean(weights * delta**2, dim=self.dim, keepdim=True) / wbar
        s = torch.sqrt(var.clamp(min=SQRT_EPS)).type_as(mu)
        mu = mu.squeeze(self.dim)
        s = s.squeeze(self.dim)
        mus = torch.cat((mu, s), dim=1)
        if self.keepdim:
            mus.unsqueeze_(dim=self.dim)

        return mus

    def forward_slidwin(
        self,
        x: torch.Tensor,
        win_length: WindowArg,
        win_shift: WindowArg,
        snip_edges: bool = False,
    ) -> torch.Tensor:
        """Computes mean-std pooling over sliding windows.

        Args:
          x: Input tensor.
          win_length: Window length in frames.
          win_shift: Window shift in frames.
          snip_edges: If ``True``, drops incomplete windows.

        Returns:
          Tensor of per-window concatenated mean/std statistics.
        """
        if isinstance(win_shift, int) and isinstance(win_length, int):
            return self._forward_slidwin_int(x, win_length, win_shift, snip_edges)

        # the window length and/or shift are floats
        return self._forward_slidwin_float(x, win_length, win_shift, snip_edges)

    def _pre_slidwin(
        self,
        x: torch.Tensor,
        win_length: WindowArg,
        win_shift: WindowArg,
        snip_edges: bool,
    ) -> tuple[torch.Tensor, Shape]:
        """Prepares padded input for sliding-window mean/std computation.

        Args:
          x: Input tensor.
          win_length: Window length in frames.
          win_shift: Window shift in frames.
          snip_edges: If ``True``, drops incomplete windows.

        Returns:
          Tuple with padded input tensor and output shape.
        """
        if self.dim != -1:
            x = x.transpose(self.dim, -1)

        x, num_frames = self._slidwin_pad(x, win_length, win_shift, snip_edges)
        out_shape = *x.shape[:-1], num_frames
        return x, out_shape

    def _post_slidwin(
        self, m_x: torch.Tensor, s_x: torch.Tensor, out_shape: Shape
    ) -> torch.Tensor:
        """Formats sliding-window mean and std outputs.

        Args:
          m_x: Flattened per-window means.
          s_x: Flattened per-window standard deviations.
          out_shape: Target output shape for each statistic tensor.

        Returns:
          Tensor with concatenated mean/std in channel dimension.
        """
        m_x = m_x.view(out_shape)
        s_x = s_x.view(out_shape)
        mus = torch.cat((m_x, s_x), dim=1)
        if self.dim != -1:
            mus = mus.transpose(self.dim, -1).contiguous()

        return mus

    def _forward_slidwin_int(
        self, x: torch.Tensor, win_length: int, win_shift: int, snip_edges: bool
    ) -> torch.Tensor:
        """Sliding-window mean/std using integer frame boundaries.

        Args:
          x: Input tensor.
          win_length: Integer window length.
          win_shift: Integer window shift.
          snip_edges: If ``True``, drops incomplete windows.

        Returns:
          Tensor of per-window concatenated mean/std statistics.
        """
        x, out_shape = self._pre_slidwin(x, win_length, win_shift, snip_edges)
        num_frames = out_shape[-1]
        c_x = torch.cumsum(x, dim=-1).view(-1, x.shape[-1])
        end = c_x[:, win_length : win_length + num_frames * win_shift : win_shift]
        start = c_x[:, : num_frames * win_shift : win_shift]
        m_x = (end - start) / win_length
        del c_x

        c_x2 = torch.cumsum(x**2, dim=-1).view(-1, x.shape[-1])
        end2 = c_x2[:, win_length : win_length + num_frames * win_shift : win_shift]
        start2 = c_x2[:, : num_frames * win_shift : win_shift]
        m_x2 = (end2 - start2) / win_length
        del c_x2
        s_x = torch.sqrt((m_x2 - m_x**2).clamp(min=SQRT_EPS))

        mus = self._post_slidwin(m_x, s_x, out_shape)
        return mus

    def _forward_slidwin_float(
        self,
        x: torch.Tensor,
        win_length: WindowArg,
        win_shift: WindowArg,
        snip_edges: bool,
    ) -> torch.Tensor:
        """Sliding-window mean/std using rounded float boundaries.

        Args:
          x: Input tensor.
          win_length: Window length (float or int).
          win_shift: Window shift (float or int).
          snip_edges: If ``True``, drops incomplete windows.

        Returns:
          Tensor of per-window concatenated mean/std statistics.
        """
        x, out_shape = self._pre_slidwin(x, win_length, win_shift, snip_edges)
        num_frames = out_shape[-1]
        c_x = torch.cumsum(x, dim=-1).view(-1, x.shape[-1])
        c_x2 = torch.cumsum(x**2, dim=-1).view(-1, x.shape[-1])

        # xx = x.view(-1, x.shape[-1])
        # print(xx.shape[1])
        # print(torch.max(torch.sum(xx==0, dim=1)))

        m_x = torch.zeros(
            (c_x.shape[0], num_frames), dtype=c_x.dtype, device=c_x.device
        )
        m_x2 = torch.zeros_like(m_x)

        k = 0
        # max_delta = 0
        # max_delta2 = 0
        for i in range(num_frames):
            k1 = int(round(k))
            k2 = int(round(k + win_length))
            m_x[:, i] = (c_x[:, k2] - c_x[:, k1]) / (k2 - k1)
            m_x2[:, i] = (c_x2[:, k2] - c_x2[:, k1]) / (k2 - k1)
            # for j in range(m_x.shape[0]):
            #     m_x_2 = torch.mean(xx[j,k1+1:k2+1])
            #     m_x2_2 = torch.mean(xx[j,k1+1:k2+1]**2)
            #     delta = torch.abs(m_x_2 - m_x[j,i]).item()
            #     delta2 = torch.abs(m_x2_2 - m_x2[j,i]).item()
            #     if (delta > max_delta or delta2 > max_delta2) and (delta>1e-3 or delta2>1e-3):
            #         max_delta = delta
            #         max_delta2 = delta2
            #         print('mx', delta, m_x[j,i], m_x_2)
            #         print('mx2', delta2, m_x2[j,i], m_x2_2)
            #         import sys
            #         sys.stdout.flush()
            #     # if m_x[j,i]**2 > m_x2[j,i]:
            #     #     print('nan')
            #     #     print('mx', m_x[j,i], m_x_2)
            #     #     print('mx2', m_x2[j,i], m_x2_2)
            #     #     print(c_x[j,k2])
            #     #     print(c_x[j,k1])
            #     #     print(c_x2[j,k2])
            #     #     print(c_x2[j,k1])
            #     #     print(xx[j,k1+1:k2+1])
            #     #     raise Exception()

            k += win_shift

        var_x = (m_x2 - m_x**2).clamp(min=SQRT_EPS)
        s_x = torch.sqrt(var_x)
        # idx = torch.isnan(s_x) #.any(dim=1)
        # if torch.sum(idx) > 0:
        #     print('sx-nan', s_x[idx])
        #     print('mx-nan', m_x[idx])
        #     print('mx2-nan', m_x2[idx])
        #     print('var-nan', m_x2[idx]-m_x[idx]**2)
        #     #print('cx2-nan', c_x2[idx])
        #     raise Exception()

        mus = self._post_slidwin(m_x, s_x, out_shape)
        return mus

    # def _forward_slidwin_int(self, x, win_length,  win_shift):
    #     num_frames = int((x.shape[self.dim] - win_length + 2*window_shift -1)/win_shift)
    #     pad_right = win_shift * (num_frames - 1) + win_length

    #     if self.dim != -1:
    #         # put pool dim at the end to do the padding
    #         x = x.transpose(self.dim, -1)

    #     xx = nnf.pad(x, (1, pad_right), mode='reflect')
    #     c_x = torch.cumsum(xx, dim=self.dim).transpose(0, -1)

    #     m_x = (c_x[win_shift:] - c_x[:-win_shift]).transpose(0, self.dim)/win_length

    #     c_x = torch.cumsum(xx**2, dim=-1).transpose(0, -1)
    #     m_x2 = (c_x[win_shift:] - c_x[:-win_shift]).transpose(0, self.dim)/win_length
    #     s_x = torch.sqrt(m_x2 - m_x**2).clamp(min=1e-5)
    #     if self.dim == -1:
    #         return torch.cat((m_x, s_x), dim=-2)

    #     return torch.cat((m_x, s_x), dim=-1)

    # def _forward_slidwin_float(self, x, win_shift, win_length):
    #     num_frames = int((x.shape[self.dim] - win_length + 2*window_shift -1)/win_shift)
    #     pad_right = win_shift * (num_frames - 1) + win_length
    #     if self.dim != -1:
    #         x = x.transpose(self.dim, -1)

    #     xx = nnf.pad(x, (1, pad_right), mode='reflect')
    #     c_x = torch.cumsum(xx, dim=-1).transpose(0, -1)
    #     c_x2 = torch.cumsum(xx**2, dim=-1).transpose(0, -1)
    #     m_x = []
    #     m_x2 = []
    #     k = 0
    #     for i in range(num_frames):
    #         k1 = int(math.round(k))
    #         k2 = int(math.round(k+win_length))
    #         w = (k2-k1)
    #         m_x.append((c_x[k2]-c_x[k1])/w)
    #         m_x2.append((c_x2[k2]-c_x2[k1])/w)
    #         k += win_shift

    #     m_x = m_x.cat(tuple(y), dim=0).transpose(0, self.dim).contiguous()
    #     m_x2 = m_x2.cat(tuple(y), dim=0).transpose(0, self.dim).contiguous()
    #     s_x = torch.sqrt(m_x2 - m_x**2).clamp(min=1e-5)
    #     if self.dim == -1:
    #         return torch.cat((m_x, s_x), dim=-2)

    #     return torch.cat((m_x, s_x), dim=-1)


class GlobalMeanLogVarPool1d(_GlobalPool1d):
    """Global mean and log-variance pooling in 1-D.

    Attributes:
      dim: Pooling dimension.
      keepdim: If ``True``, preserves the pooled dimension in the output.
      size_multiplier: Output feature multiplier, equal to ``2``.
    """

    def __init__(self, dim: int = -1, keepdim: bool = False) -> None:
        """Initializes global mean/log-variance pooling.

        Args:
          dim: Pooling dimension.
          keepdim: If ``True``, preserves the pooled dimension in the output.

        Returns:
          ``None``.
        """
        super().__init__(dim, keepdim)
        self.size_multiplier = 2

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Applies (optionally weighted) global mean/log-variance pooling.

        Args:
          x: Input tensor.
          x_lengths: Lengths of the input sequences in the pooling dimension.
            Used only when ``weights`` is not given.
          weights: Weights for weighted pooling with shape=(batch, max_length)
                   or (batch,..., max_length,...) with shape matching the one
                   of the input tensor.

        Returns:
          Tensor containing concatenated mean and log-variance statistics.
        """
        weights = self._standardize_weights(x, x_lengths, weights)
        if weights is None:
            mu = torch.mean(x, dim=self.dim, keepdim=True)
            x2bar = torch.mean(x**2, dim=self.dim, keepdim=True)
            logvar = torch.log((x2bar - mu * mu).clamp(min=SQRT_EPS))
            mu = mu.squeeze(self.dim)
            logvar = logvar.squeeze(self.dim)
            mulv = torch.cat((mu, logvar), dim=1)
            if self.keepdim:
                mulv = mulv.unsqueeze(self.dim)
            return mulv

        xbar = torch.mean(weights * x, dim=self.dim, keepdim=True)
        wbar = torch.mean(weights, dim=self.dim, keepdim=True)
        mu = xbar / wbar
        x2bar = torch.mean(weights * x**2, dim=self.dim, keepdim=True) / wbar
        var = (x2bar - mu * mu).clamp(min=SQRT_EPS)
        logvar = torch.log(var)

        mu = mu.squeeze(self.dim)
        logvar = logvar.squeeze(self.dim)
        mulv = torch.cat((mu, logvar), dim=1)
        if self.keepdim:
            mulv = mulv.unsqueeze(self.dim)

        return mulv


class LDEPool1d(_GlobalPool1d):
    """Learnable dictionary encoder pooling in 1d.
    It only works for 3-D tensors.

    Attributes:
      in_feats: Input feature dimension.
      num_comp: Number of dictionary components.
      dist_pow: Power used by the distance metric.
      use_bias: Whether additive posterior bias is enabled.
      dim: Pooling dimension.
      keepdim: If ``True``, preserves the pooled dimension in the output.
      size_multiplier: Output feature multiplier, equal to ``num_comp``.
    """

    def __init__(
        self,
        in_feats: int,
        num_comp: int = 64,
        dist_pow: int = 2,
        use_bias: bool = False,
        dim: int = -1,
        keepdim: bool = False,
    ) -> None:
        """Initializes dictionary-encoder pooling.

        Args:
          in_feats: Input feature dimension.
          num_comp: Number of dictionary components.
          dist_pow: Distance exponent (1 for Euclidean norm, 2 for squared norm).
          use_bias: If ``True``, uses a learnable additive bias in posterior
            logits.
          dim: Pooling dimension.
          keepdim: If ``True``, preserves the pooled dimension in the output.

        Returns:
          ``None``.
        """
        super().__init__(dim, keepdim)
        self.mu = nn.Parameter(torch.randn((num_comp, in_feats)))
        self.prec = nn.Parameter(torch.ones((num_comp,)))
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros((num_comp,)))
        else:
            self.bias = 0

        self.dist_pow = dist_pow
        self.dist_f: Callable[[torch.Tensor], torch.Tensor]
        if dist_pow == 1:
            self.dist_f = lambda x: torch.norm(x, p=2, dim=-1)
        else:
            self.dist_f = lambda x: torch.sum(x**2, dim=-1)

        self.size_multiplier = num_comp

    @property
    def num_comp(self) -> int:
        return self.mu.shape[0]

    @property
    def in_feats(self) -> int:
        return self.mu.shape[1]

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        s = "{}(in_feats={}, num_comp={}, dist_pow={}, use_bias={}, dim={}, keepdim={})".format(
            self.__class__.__name__,
            self.mu.shape[1],
            self.mu.shape[0],
            self.dist_pow,
            self.use_bias,
            self.dim,
            self.keepdim,
        )
        return s

    def _standardize_weights(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Standardizes frame weights to shape ``(batch, time)``.

        Args:
          x: Input tensor.
          x_lengths: Optional sequence lengths for mask generation.
          weights: Optional frame weights.

        Returns:
          Standardized weights tensor or ``None``.
        """
        if weights is None:
            time_dim = self.dim if self.dim >= 0 else x.dim() + self.dim
            return seq_lengths_to_mask(
                x_lengths, x.size(self.dim), dtype=x.dtype, time_dim=time_dim
            )

        if weights.dim() == x.dim():
            if self.dim != 1 and self.dim != -2:
                weights = weights.transpose(1, self.dim)

            # LDE uses frame-level posteriors r with shape (batch, time, num_comp),
            # so weights must collapse to (batch, time).
            if weights.size(2) != 1:
                raise ValueError(
                    "LDEPool1d expects frame-level weights with shape (batch, time) "
                    "or (batch, time, 1) after standardization."
                )
            return weights.squeeze(2)

        assert weights.dim() == 2
        return weights

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Applies dictionary-encoder pooling to the input.

        Args:
          x: Input tensor of shape=(batch, time, feat_dim) or (batch, feat_dim, time).
          x_lengths: Lengths of the input sequences in the pooling dimension.
                     x_lengths is only used if weights is not given.
          weights: Weights for weighted pooling with shape=(batch, max_length)
                   or (batch,..., max_length,...) with shape matching the one
                   of the input tensor.

        Returns:
          Pooled representation of shape ``(batch, num_comp * in_feats)`` or with
          an extra singleton dimension when ``keepdim=True``.
        """
        assert x.dim() == 3, "LDEPool1d only works for 3-D tensors"
        weights = self._standardize_weights(x, x_lengths, weights)
        if self.dim != 1 and self.dim != -2:
            x = x.transpose(1, self.dim)  # (batch, time, feat_dim)

        x = torch.unsqueeze(x, dim=2)  # (batch, time, 1, feat_dim)
        delta = x - self.mu  # (batch, time, num_comp, feat_dim)
        dist = self.dist_f(delta)  # (batch, time, num_comp)

        llk = -self.prec**2 * dist + self.bias
        r = nnf.softmax(llk, dim=-1)  # (batch, time, num_comp)
        if weights is not None:
            r *= weights

        r = torch.unsqueeze(r, dim=-1)  # (batch, time, num_comp, 1)
        N = torch.sum(r, dim=1) + N_EPS  # (batch, num_comp, 1)
        F = torch.sum(r * delta, dim=1)  # (batch, num_comp, feat_dim)
        pool = F / N  # (batch, num_comp, feat_dim)
        pool = pool.contiguous().view(-1, self.num_comp * self.in_feats)
        # (batch, num_comp * feat_dim)
        if self.keepdim:
            if self.dim == 1 or self.dim == -2:
                pool = pool.unsqueeze(1)
            else:
                pool = pool.unsqueeze(-1)

        return pool

    def get_config(self) -> dict[str, Any]:
        """Returns a serializable layer configuration.

        Returns:
          Dictionary with constructor-equivalent arguments.
        """
        config = {
            "in_feats": self.in_feats,
            "num_comp": self.num_comp,
            "dist_pow": self.dist_pow,
            "use_bias": self.use_bias,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ScaledDotProdAttV1Pool1d(_GlobalPool1d):
    """Scaled dot-product attention pooling in 1-D.

    The attention weights are obtained from scaled inner products between
    feature frames and learned query parameters. This class expects 3-D input.

    Attributes:
      in_feats: Input feature dimension.
      num_heads: Number of attention heads.
      d_k: Key/query dimension per head.
      d_v: Value dimension per head.
      bin_attn: Whether binary sigmoid attention is used.
      dim: Pooling dimension.
      keepdim: If ``True``, preserves the pooled dimension in the output.
      attn: Most recent attention weights.
      size_multiplier: Output feature multiplier, equal to
        ``num_heads * d_v / in_feats``.
    """

    def __init__(
        self,
        in_feats: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        bin_attn: bool = False,
        dim: int = -1,
        keepdim: bool = False,
    ) -> None:
        """Initializes scaled dot-product attention pooling.

        Args:
          in_feats: Input feature dimension.
          num_heads: Number of attention heads.
          d_k: Key/query dimension per head.
          d_v: Value dimension per head.
          bin_attn: If ``True``, applies sigmoid-based binary attention instead
            of softmax attention.
          dim: Pooling dimension.
          keepdim: If ``True``, preserves the pooled dimension in the output.

        Returns:
          ``None``.
        """
        super().__init__(dim, keepdim)

        self.d_v = d_v
        self.d_k = d_k
        self.num_heads = num_heads
        self.bin_attn = bin_attn
        self.q = nn.Parameter(torch.Tensor(1, num_heads, 1, d_k))
        nn.init.orthogonal_(self.q)
        if self.bin_attn:
            self.bias = nn.Parameter(torch.zeros((1, num_heads, 1, 1)))

        self.linear_k = nn.Linear(in_feats, num_heads * d_k)
        self.linear_v = nn.Linear(in_feats, num_heads * d_v)
        self.attn: torch.Tensor | None = None
        self.size_multiplier = num_heads * d_v / in_feats

    @property
    def in_feats(self) -> int:
        return self.linear_v.in_features

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        s = "{}(in_feats={}, num_heads={}, d_k={}, d_v={}, bin_attn={}, dim={}, keepdim={})".format(
            self.__class__.__name__,
            self.in_feats,
            self.num_heads,
            self.d_k,
            self.d_v,
            self.bin_attn,
            self.dim,
            self.keepdim,
        )
        return s

    def _standardize_weights(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Standardizes frame weights to shape ``(batch, time)``.

        Args:
          x: Input tensor.
          x_lengths: Optional sequence lengths for mask generation.
          weights: Optional frame weights.

        Returns:
          Standardized weights tensor or ``None``.
        """
        if weights is None:
            return seq_lengths_to_mask(
                x_lengths, x.size(self.dim), dtype=x.dtype, time_dim=2
            )

        if weights.dim() == x.dim():
            if self.dim != 1 and self.dim != -2:
                weights = weights.transpose(1, self.dim)

            # Attention masking is time-wise, so weights must be (batch, time).
            if weights.size(2) != 1:
                raise ValueError(
                    "ScaledDotProdAttV1Pool1d expects frame-level weights with "
                    "shape (batch, time) or (batch, time, 1) after standardization."
                )
            return weights.squeeze(2)

        assert weights.dim() == 2
        return weights

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Applies attention pooling to the input.

        Args:
          x: Input tensor of shape=(batch, time, feat_dim) or (batch, feat_dim, time).
          x_lengths: Lengths of the input sequences in the pooling dimension.
                     x_lengths is only used if weights is not given.
          weights: Weights for weighted pooling with shape=(batch, max_length)
                   or (batch,..., max_length,...) with shape matching the one
                   of the input tensor. In this implementation only binary
                   weights are allowed.

        Returns:
          Tensor of attended pooled features with shape
          ``(batch, num_heads * d_v)`` or with an extra singleton dimension when
          ``keepdim=True``.
        """
        weights = self._standardize_weights(x, x_lengths, weights)
        batch_size = x.size(0)
        if self.dim == 2 or self.dim == -1:
            x = x.transpose(1, self.dim)

        k = self.linear_k(x).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.linear_v(x).view(batch_size, -1, self.num_heads, self.d_v)
        k = k.transpose(1, 2)  # (batch, head, time, d_k)
        v = v.transpose(1, 2)  # (batch, head, time, d_v)

        scores = torch.matmul(self.q, k.transpose(-2, -1)) / math.sqrt(
            self.d_k
        )  # (batch, head, 1, time)
        if self.bin_attn:
            # use binary attention.
            scores = torch.sigmoid(scores + self.bias)

        # scores = scores.squeeze(dim=-1)                    # (batch, head, time)
        if weights is not None:
            mask = weights.view(batch_size, 1, 1, -1).eq(0)  # (batch, 1, 1, time)
            if self.bin_attn:
                scores = scores.masked_fill(mask, 0.0)
                self.attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-9)
            else:
                if scores.dtype == torch.half:
                    min_value = -65504
                else:
                    min_value = -1e200
                scores = scores.masked_fill(mask, min_value)
                self.attn = torch.softmax(scores, dim=-1).masked_fill(
                    mask, 0.0
                )  # (batch, head, 1, time)
        else:
            if self.bin_attn:
                self.attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-9)
            else:
                self.attn = torch.softmax(scores, dim=-1)  # (batch, head, 1, time)

        x = torch.matmul(self.attn, v)  # (batch, head, 1, d_v)
        if self.keepdim:
            if self.dim == 1 or self.dim == -2:
                x = x.view(
                    batch_size, 1, self.num_heads * self.d_v
                )  # (batch, 1, d_model)
            else:
                x = x.view(
                    batch_size, self.num_heads * self.d_v, 1
                )  # (batch, d_model, 1)
        else:
            x = x.view(batch_size, self.num_heads * self.d_v)  # (batch, d_model)
        return x

    def get_config(self) -> dict[str, Any]:
        """Returns a serializable layer configuration.

        Returns:
          Dictionary with constructor-equivalent arguments.
        """
        config = {
            "in_feats": self.in_feats,
            "num_heads": self.num_heads,
            "d_k": self.d_k,
            "d_v": self.d_v,
            "bin_attn": self.bin_attn,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GlobalChWiseAttMeanStdPool1d(_GlobalPool1d):
    """Attentive mean + stddev pooling for each channel.
    This class only works on 3-D tensors.

    Attributes:
      in_feats: Input feature dimension.
      inner_feats: Feature dimension in the hidden layer of the content based attention.
      bin_attn: Whether binary sigmoid attention is used.
      use_global_context: If True, concat global stats pooling to the input features to
                          compute the attention.
      norm_layer: Normalization layer object, if None, it used BatchNorm1d.
      dim: Pooling dimension.
      keepdim: If ``True``, preserves the pooled dimension in the output.
      size_multiplier: Output feature multiplier, equal to ``2``.
    """

    def __init__(
        self,
        in_feats: int,
        inner_feats: int = 128,
        bin_attn: bool = False,
        use_global_context: bool = True,
        norm_layer: Callable[[int], nn.Module] | None = None,
        dim: int = -1,
        keepdim: bool = False,
    ) -> None:
        """Initializes channel-wise attentive mean/std pooling.

        Args:
          in_feats: Input feature dimension.
          inner_feats: Hidden feature dimension used to predict attention
            weights.
          bin_attn: If ``True``, applies sigmoid-based binary attention instead
            of softmax attention.
          use_global_context: If ``True``, uses global mean/std context when
            computing attention logits.
          norm_layer: Normalization layer constructor. Defaults to
            ``BatchNorm1d``.
          dim: Pooling dimension.
          keepdim: If ``True``, preserves the pooled dimension in the output.

        Returns:
          ``None``.
        """
        super().__init__(dim, keepdim)
        self.size_multiplier = 2
        self.in_feats = in_feats
        self.inner_feats = inner_feats
        self.bin_attn = bin_attn

        self.use_global_context = use_global_context
        self.conv1 = _conv1(in_feats, inner_feats)
        if use_global_context:
            self.lin_global = nn.Linear(2 * in_feats, inner_feats, bias=False)
        # torch.autograd.set_detect_anomaly(True)
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.norm_layer = norm_layer(inner_feats)
        self.activation = nn.Tanh()
        self.conv2 = _conv1(inner_feats, in_feats, bias=True)
        self.stats_pool = GlobalMeanStdPool1d(dim=-1)
        if self.bin_attn:
            self.bias = nn.Parameter(torch.zeros((1, in_feats, 1)))

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        s = "{}(in_feats={}, inner_feats={}, use_global_context={}, bin_attn={}, dim={}, keepdim={})".format(
            self.__class__.__name__,
            self.in_feats,
            self.inner_feats,
            self.use_global_context,
            self.bin_attn,
            self.dim,
            self.keepdim,
        )
        return s

    def _standardize_weights(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Standardizes frame weights for channel-wise attentive pooling.

        Args:
          x: Input tensor.
          x_lengths: Optional sequence lengths for mask generation.
          weights: Optional frame weights.

        Returns:
          A tensor broadcastable to ``x`` or ``None``.
        """
        if weights is None:
            time_dim = self.dim if self.dim >= 0 else x.dim() + self.dim
            return seq_lengths_to_mask(
                x_lengths,
                x.size(self.dim),
                dtype=x.dtype,
                time_dim=time_dim,
            )

        if weights.dim() == x.dim():
            return weights.transpose(self.dim, -1)

        assert weights.dim() == 2
        shape = x.dim() * [1]
        shape[0] = weights.shape[0]
        shape[-1] = weights.shape[1]
        return weights.view(tuple(shape))

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Applies channel-wise attentive mean/std pooling.

        Args:
          x: Input tensor of shape=(batch, time, feat_dim) or (batch, feat_dim, time).
          x_lengths: Lengths of the input sequences in the pooling dimension.
                     x_lengths is only used if weights is not given.
          weights: Weights for weighted pooling with shape=(batch, max_length)
                   or (batch,..., max_length,...) with shape matching the one
                   of the input tensor.

        Returns:
          Tensor containing attentive mean/std statistics per channel. The output
          keeps or removes the pooled dimension according to ``keepdim``.
        """
        assert x.dim() == 3, "Input should be a 3d tensor"
        if self.dim == 1 or self.dim == -2:
            x = x.transpose(1, self.dim)

        # x = (batch, feat_dim, time)
        weights = self._standardize_weights(x, x_lengths, weights)  # (batch, 1,  time)
        x_inner = self.conv1(x)  # (batch, inner_dim, time)
        if not torch.all(torch.isfinite(x)):
            logging.warning("non-finite x-avg=%f", torch.mean(x))
        if not torch.all(torch.isfinite(x_inner)):
            logging.warning("non-finite x-inner-avg=%f", torch.mean(x_inner))
        # is_nan = torch.isnan(x_inner)
        # if torch.any(is_nan):
        #     logging.warning(
        #         f"""xinner-nan={torch.sum(is_nan)}
        #     xinner-batch-nan={torch.sum(is_nan, dim=(1,2))}
        #     xinner-channel-nan={torch.sum(is_nan, dim=(0,2))}
        #     xinner-time-nan={torch.sum(is_nan, dim=(0,1))}
        #     xinner-shape={x_inner.shape}"""
        #     )
        if self.use_global_context:
            global_mus = self.stats_pool(x, weights=weights)
            x_inner = x_inner + self.lin_global(global_mus).unsqueeze(-1)

            if not torch.all(torch.isfinite(global_mus)):
                logging.warning("non-finite global-mus-avg=%f", torch.mean(global_mus))
        if not torch.all(torch.isfinite(x_inner)):
            logging.warning("non-finite x-inner-avg=%f", torch.mean(x_inner))
        # attn = self.conv2(
        #     self.activation(self.norm_layer(x_inner))
        # )  # (batch, feat_dim, time)
        a1 = self.norm_layer(x_inner)
        a2 = self.activation(a1)
        attn = self.conv2(a2)
        if not torch.all(torch.isfinite(attn)):
            logging.warning(
                "non-finite attn-avg=%f %f %f %f %f %f %s %s",
                torch.mean(attn),
                torch.mean(a1),
                torch.mean(a2),
                torch.mean(x_inner),
                torch.max(x_inner),
                torch.min(x_inner),
                str(x_inner.dtype),
                str(attn.dtype),
            )
        if self.bin_attn:
            attn = torch.sigmoid(attn + self.bias).clamp(min=N_EPS)
        else:
            if weights is not None:
                if attn.dtype == torch.half:
                    min_value = -65504
                else:
                    min_value = -1e20
                mask = weights.eq(0)
                attn = attn.masked_fill(mask, min_value)

            if not torch.all(torch.isfinite(attn)):
                logging.warning("non-finite attn-avg=%f", torch.mean(attn))
            attn = nnf.softmax(attn, dim=-1)

        if not torch.all(torch.isfinite(attn)):
            logging.warning("non-finite attn-avg=%f", torch.mean(attn))

        if weights is not None:
            attn = attn * weights

        mus = self.stats_pool(x, weights=attn)
        if not torch.all(torch.isfinite(mus)):
            logging.warning("non-finite mus-avg=%f", torch.mean(mus))

        if self.keepdim:
            mus = mus.unsqueeze(self.dim)

        return mus

    def get_config(self) -> dict[str, Any]:
        """Returns a serializable layer configuration.

        Returns:
          Dictionary with constructor-equivalent arguments.
        """
        config = {
            "in_feats": self.in_feats,
            "inner_feats": self.inner_feats,
            "use_global_context": self.use_global_context,
            "bin_attn": self.bin_attn,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
