"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from ..layers.attention_v1 import *
from .transformer_feedforward import *


class TransformerEncoderBlockV1(nn.Module):
    """Building block for transformer encoder.

    Attributes:
      num_feats: Input/output feature dimension (d_model).
      self_attn: Attention module or string in ['scaled-dot-prod-v1', 'local-scaled-dot-prod-v1'].
      num_heads: Number of heads.
      feed_forward: position-wise feed-forward nn.Module or string in ['linear', 'conv1dx2', 'conv1d-linear']
      d_ff: Dimension of middle layer in the feed-forward block.
      ff_kernel_size: Kernel size for convolutional feed-forward variants.
      ff_act: Feed-forward hidden activation.
      ff_dropout_rate: Dropout rate for feed-forward and residual branches.
      att_context: Maximum context range for local attention.
      att_dropout_rate: Dropout rate for attention block.
      rel_pos_enc: If True, use relative positional encodings; otherwise absolute encodings.
      causal_pos_enc: If True, use causal positional encodings (when rel_pos_enc=True), assuming
                      query q_i only attends to key k_j when j <= i.
      norm_before: If True, use layer norm before each sublayer; otherwise after.
      concat_after: If True, concatenate attention input and output and apply linear transform,
                    i.e., y = x + linear(concat(x, att(x))); if False, y = x + att(x).

    """

    def __init__(
        self,
        num_feats: int,
        self_attn: Union[str, nn.Module],
        num_heads: int,
        feed_forward: Union[str, nn.Module],
        d_ff: int,
        ff_kernel_size: int,
        ff_act: str = "relu6",
        ff_dropout_rate: float = 0,
        att_context: int = 25,
        att_dropout_rate: float = 0,
        rel_pos_enc: bool = False,
        causal_pos_enc: bool = False,
        norm_before: bool = True,
        concat_after: bool = False,
    ) -> None:
        """Initializes a transformer encoder block with attention and feed-forward sublayers.

        Args:
          num_feats: Input/output feature dimension (d_model).
          self_attn: Attention module or attention type string.
          num_heads: Number of attention heads.
          feed_forward: Feed-forward module or feed-forward type string.
          d_ff: Hidden dimension in the feed-forward block.
          ff_kernel_size: Kernel size for convolutional feed-forward variants.
          ff_act: Activation function in the feed-forward block.
          ff_dropout_rate: Dropout probability applied after attention and feed-forward.
          att_context: Local attention context size.
          att_dropout_rate: Dropout probability inside the attention module.
          rel_pos_enc: Whether to use relative positional encodings.
          causal_pos_enc: Whether positional encodings are causal (for autoregressive masking).
          norm_before: Whether to apply layer normalization before each sublayer.
          concat_after: Whether to concatenate attention input/output before projection.
        """

        super().__init__()
        if isinstance(self_attn, str):
            self.self_attn = self._make_att(
                self_attn,
                num_feats,
                num_heads,
                att_context,
                att_dropout_rate,
                rel_pos_enc,
                causal_pos_enc,
            )
        else:
            self.self_attn = self_attn

        if isinstance(feed_forward, str):
            self.feed_forward = self._make_ff(
                feed_forward, num_feats, d_ff, ff_kernel_size, ff_act, ff_dropout_rate
            )
        else:
            self.feed_forward = feed_forward

        self.norm1 = nn.LayerNorm(num_feats)
        self.norm2 = nn.LayerNorm(num_feats)
        self.dropout_rate = ff_dropout_rate
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)

        self.norm_before = norm_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(num_feats + num_feats, num_feats)

    @staticmethod
    def _make_att(
        att_type: str,
        num_feats: int,
        num_heads: int,
        context: int,
        dropout_rate: float,
        rel_pos_enc: bool,
        causal_pos_enc: bool,
    ) -> nn.Module:
        """Creates multihead attention block from att_type string

        Args:
           att_type: String in ['scaled-dot-prod-v1', 'local-scaled-dot-prod-v1'].
           num_feats: Input/output feature dimension (d_model).
           num_heads: Number of heads.
           context: Maximum left/right context for local attention.
           dropout_rate: Dropout rate for attention block.
           rel_pos_enc: If True, use relative positional encodings; otherwise absolute encodings.
           causal_pos_enc: If True, use causal positional encodings (when rel_pos_enc=True), assuming
                           query q_i only attends to key k_j when j <= i.

        Returns:
           Attention nn.Module
        """

        assert num_feats % num_heads == 0
        d_k = num_feats // num_heads

        if att_type == "scaled-dot-prod-v1":
            if rel_pos_enc:
                return ScaledDotProdAttRelPosEncV1(
                    num_feats,
                    num_feats,
                    num_heads,
                    d_k,
                    d_k,
                    causal_pos_enc,
                    dropout_rate,
                )

            return ScaledDotProdAttV1(
                num_feats,
                num_feats,
                num_heads,
                d_k,
                d_k,
                dropout_rate,
            )

        if att_type == "local-scaled-dot-prod-v1":
            if rel_pos_enc:
                return LocalScaledDotProdAttRelPosEncV1(
                    num_feats,
                    num_feats,
                    num_heads,
                    d_k,
                    d_k,
                    context,
                    causal_pos_enc,
                    dropout_rate,
                )

            return LocalScaledDotProdAttV1(
                num_feats,
                num_feats,
                num_heads,
                d_k,
                d_k,
                context,
                dropout_rate,
            )
        raise ValueError(f"unknown att_type={att_type}")

    @staticmethod
    def _make_ff(
        ff_type: str,
        num_feats: int,
        hid_feats: int,
        kernel_size: int,
        activation: str,
        dropout_rate: float,
    ) -> nn.Module:
        """Creates position-wise feed forward block from ff_type string

        Args:
          ff_type: string in ['linear', 'conv1dx2', 'conv1d-linear']
          num_feats: input/output feat. dimension (aka d_model)
          hid_feats: dimension of middle layer in feed_forward block
          kernel_size: kernel size for convolutional versions of ff block
          dropout_rate: dropout rate for ff block
          activation: activation function for ff block

        Returns:
          Position-wise feed-forward nn.Module

        """
        if ff_type == "linear":
            return PositionwiseFeedForward(
                num_feats, hid_feats, activation, dropout_rate, time_dim=1
            )

        if ff_type == "conv1dx2":
            return Conv1dx2(
                num_feats, hid_feats, kernel_size, activation, dropout_rate, time_dim=1
            )

        if ff_type == "conv1d-linear":
            return Conv1dLinear(
                num_feats, hid_feats, kernel_size, activation, dropout_rate, time_dim=1
            )
        raise ValueError(f"unknown ff_type={ff_type}")

    def forward(
        self, x: Tensor, pos_emb: Optional[Tensor] = None, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass function

        Args:
          x: Input tensor of shape (batch, time, num_feats).
          pos_emb: Positional embeddings of shape (batch, time2, num_feats) as
            R_{L-1}, ..., R_0 when using relative positional encoding; otherwise None.
          mask: Optional mask for valid time steps with shape (batch, time).

        Returns:
          Tuple of:
            Tensor with output features of shape (batch, time, num_feats).
            Optional mask propagated from the input.
        """
        residual = x
        if self.norm_before:
            x = self.norm1(x)

        if pos_emb is None:
            x_att = self.self_attn(x, x, x, mask=mask)
        else:
            x_att = self.self_attn(x, x, x, pos_emb=pos_emb, mask=mask)
        if self.concat_after:
            x = torch.cat((x, x_att), dim=-1)
            x = self.concat_linear(x)
        else:
            x = x_att

        if self.dropout_rate > 0:
            x = self.dropout(x)

        x = residual + x
        if not self.norm_before:
            x = self.norm1(x)

        residual = x
        if self.norm_before:
            x = self.norm2(x)

        x = self.feed_forward(x)
        if self.dropout_rate > 0:
            x = self.dropout(x)

        x = residual + x
        if not self.norm_before:
            x = self.norm2(x)

        return x, mask
