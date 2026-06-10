"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

#

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Optional, Tuple, Union

from ..layers.attention_v1 import *
from .conformer_conv import ConformerConvBlock
from .transformer_feedforward import *


class ConformerEncoderBlockV1(nn.Module):
    """Building block for conformer encoder introduced in
       https://arxiv.org/pdf/2005.08100.pdf

        This includes some optional extra features
        not included in the original paper:
           - Choose local-attention (attending only to close frames
             instead of all the frames in the sequence)
           - Choose number of conv blocks
           - Squeeze-Excitation after depthwise-conv
           - Allows downsampling in time dimension
           - Allows choosing activation and layer normalization type
        We call this Conformer+

    Attributes:
      num_feats: input/output feat. dimension (aka d_model)
      self_attn: attention module in ['scaled-dot-prod-att-v1', 'local-scaled-dot-prod-att-v1']
      num_heads: number of heads
      conv_repeats: number of conv blocks
      conv_kernel_size: kernel size for conv blocks
      conv_stride: stride for depth-wise conv in first conv block
      feed_forward: position-wise feed-forward string in ['linear', 'conv1dx2', 'conv1d-linear']
      d_ff: dimension of middle layer in feed_forward block
      ff_kernel_size: kernel size for convolutional versions of ff block
      hid_act: ff and conv block hidden activation
      dropout_rate: dropout rate for ff and conv blocks
      att_context: maximum context range for local attention
      att_dropout_rate: dropout rate for attention block
      causal_pos_enc: if True, use causal positional encodings (when rel_pos_enc=True), it assumes
                      that query q_i only attends to key k_j when j<=i
      conv_norm_layer: norm layer constructor for conv block,
                       if None it uses BatchNorm1d.
      se_r:         Squeeze-Excitation compression ratio,
                    if None it doesn't use Squeeze-Excitation
      ff_macaron: if True, it uses macaron-net style ff layers, otherwise transformer style.
      out_lnorm: if True, use LayerNorm at the output as in the conformer paper,
                 we think that this layer is redundant and put it to False by default
      concat_after: if True, if concats attention input and output and apply linear transform, i.e.,
                             y = x + linear(concat(x, att(x)))
                    if False, y = x + att(x)

    """

    def __init__(
        self,
        num_feats: int,
        self_attn: Any,
        num_heads: int,
        conv_repeats: int = 1,
        conv_kernel_size: int = 31,
        conv_stride: int = 1,
        feed_forward: str = "linear",
        d_ff: int = 2048,
        ff_kernel_size: int = 3,
        hid_act: Union[str, Dict[str, Any]] = "swish",
        dropout_rate: float = 0,
        att_context: int = 25,
        att_dropout_rate: float = 0,
        pos_enc_type: str = "rel",
        causal_pos_enc: bool = False,
        conv_norm_layer: Optional[Callable[..., nn.Module]] = None,
        se_r: Optional[int] = None,
        ff_macaron: bool = True,
        out_lnorm: bool = False,
        concat_after: bool = False,
    ) -> None:
        """Initialize the conformer encoder block.

        Args:
          num_feats: Input/output feature dimension.
          self_attn: Self-attention constructor or module config.
          num_heads: Number of attention heads.
          conv_repeats: Number of convolution sub-blocks.
          conv_kernel_size: Kernel size for the convolution block.
          conv_stride: Stride for the first convolution block.
          feed_forward: Feed-forward block type.
          d_ff: Hidden dimension of the feed-forward block.
          ff_kernel_size: Kernel size for convolutional feed-forward variants.
          hid_act: Hidden activation specification accepted by
            ``ActivationFactory``.
          dropout_rate: Dropout probability for the block.
          att_context: Local attention context size.
          att_dropout_rate: Attention dropout probability.
          pos_enc_type: Positional encoding type.
          causal_pos_enc: Whether to use causal relative positions.
          conv_norm_layer: Convolution normalization layer constructor.
          se_r: Optional squeeze-excitation reduction ratio.
          ff_macaron: Whether to use macaron-style feed-forward blocks.
          out_lnorm: Whether to apply an output layer norm.
          concat_after: Whether to concatenate attention input/output.
        """
        super().__init__()
        self.self_attn = self._make_att(
            self_attn,
            num_feats,
            num_heads,
            att_context,
            att_dropout_rate,
            pos_enc_type,
            causal_pos_enc,
        )

        self.ff_scale = 1
        self.ff_macaron = ff_macaron
        if ff_macaron:
            self.ff_scale = 0.5
            self.feed_forward_macaron = self._make_ff(
                feed_forward, num_feats, d_ff, ff_kernel_size, hid_act, dropout_rate
            )
            self.norm_ff_macaron = nn.LayerNorm(num_feats)

        self.feed_forward = self._make_ff(
            feed_forward, num_feats, d_ff, ff_kernel_size, hid_act, dropout_rate
        )

        conv_blocks = []
        for i in range(conv_repeats):
            block_i = ConformerConvBlock(
                num_feats,
                conv_kernel_size,
                conv_stride,
                activation=hid_act,
                norm_layer=conv_norm_layer,
                dropout_rate=dropout_rate,
                se_r=se_r,
            )
            conv_stride = 1
            conv_blocks.append(block_i)

        self.conv_blocks = nn.ModuleList(conv_blocks)

        self.norm_att = nn.LayerNorm(num_feats)
        self.norm_ff = nn.LayerNorm(num_feats)
        self.out_lnorm = out_lnorm
        if out_lnorm:
            self.norm_out = nn.LayerNorm(num_feats)
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)

        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(num_feats + num_feats, num_feats)

    def change_attn_dropout(self, att_dropout_rate: float) -> None:
        """Update the attention dropout rate.

        Args:
          att_dropout_rate: New attention dropout probability.
        """
        attn = self.self_attn
        if hasattr(attn, "dropout_rate"):
            attn.dropout_rate = att_dropout_rate
            attn.dropout.p = att_dropout_rate

    @staticmethod
    def _make_att(
        att_type: str,
        num_feats: int,
        num_heads: int,
        context: int,
        dropout_rate: float,
        pos_enc_type: str,
        causal_pos_enc: bool,
    ) -> Any:
        """Creates multihead attention block from att_type string

      Args:
           att_type: string in ['scaled-dot-prod-v1', 'local-scaled-dot-prod-v1', 'block-scaled-dot-prod-v1']
           num_feats: input/output feat. dimension (aka d_model)
           num_heads: number of heads
           context: block attention receptive field
           dropout_rate: dropout rate for attention block
           pos_enc_type: type of positional encoder
           causal_pos_enc: if True, use causal positional encodings (when rel_pos_enc=True), it assumes
                           that query q_i only attends to key k_j when j<=i

        Returns:
           Attention nn.Module
        """

        assert num_feats % num_heads == 0
        d_k = num_feats // num_heads

        if att_type == "scaled-dot-prod-v1":
            if pos_enc_type == "rel":
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
            if pos_enc_type == "rel":
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

        if att_type == "block-scaled-dot-prod-v1":
            if pos_enc_type == "rel":
                return BlockScaledDotProdAttRelPosEncV1(
                    num_feats,
                    num_feats,
                    num_heads,
                    d_k,
                    d_k,
                    context,
                    causal_pos_enc,
                    dropout_rate,
                )

            return BlockScaledDotProdAttV1(
                num_feats,
                num_feats,
                num_heads,
                d_k,
                d_k,
                context,
                dropout_rate,
            )

        raise ValueError(f"unknown attention type: {att_type}")

    @staticmethod
    def _make_ff(
        ff_type: str,
        num_feats: int,
        hid_feats: int,
        kernel_size: int,
        activation: Union[str, Dict[str, Any]],
        dropout_rate: float,
    ) -> Any:
        """Creates position-wise feed forward block from ff_type string

        Args:
          ff_type: string in ['linear', 'conv1dx2', 'conv1d-linear']
          num_feats: input/output feat. dimension (aka d_model)
          hid_feats: dimension of middle layer in feed_forward block
          kernel_size: kernel size for convolutional versions of ff block
          dropout_rate: dropout rate for ff block
          activation: activation specification accepted by ``ActivationFactory``.

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

        raise ValueError(f"unknown feed-forward type: {ff_type}")

    def _forward_ff_macaron(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the macaron feed-forward residual branch.

        Args:
          x: Input tensor with shape ``(batch, time, features)``.

        Returns:
          Tensor with the same shape as ``x``.
        """
        residual = x
        x = self.norm_ff_macaron(x)
        x = self.feed_forward_macaron(x)
        if self.dropout_rate > 0:
            x = self.dropout(x)

        x = residual + self.ff_scale * x
        return x

    def _forward_self_attn(
        self,
        x: torch.Tensor,
        pos_emb: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply self-attention with residual connection.

        Args:
          x: Input tensor with shape ``(batch, time, features)``.
          pos_emb: Optional positional embedding tensor.
          mask: Optional valid-frame mask.

        Returns:
          Tensor with the same shape as ``x``.
        """
        residual = x
        x = self.norm_att(x)
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
        return x

    def _forward_convs(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution sub-blocks.

        Args:
          x: Input tensor with shape ``(batch, time, features)``.

        Returns:
          Tensor with the same shape as ``x``.
        """
        x = x.transpose(1, 2)
        for block in range(len(self.conv_blocks)):
            x = self.conv_blocks[block](x)

        x = x.transpose(1, 2)
        return x

    def _forward_ff(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the final feed-forward residual branch.

        Args:
          x: Input tensor with shape ``(batch, time, features)``.

        Returns:
          Tensor with the same shape as ``x``.
        """
        residual = x
        x = self.norm_ff(x)
        x = self.feed_forward(x)
        if self.dropout_rate > 0:
            x = self.dropout(x)

        x = residual + self.ff_scale * x
        return x

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass function

        Args:
          x: input tensor with size=(batch, time, num_feats)
          pos_emb: positional embedding size=(batch, time2, in_feats) as R_{L-1}, ..., R_0,
                   when using relative positional encoder, otherwise None
          mask: mask to indicate valid time steps for x (batch, time)

        Returns:
           Tensor with output features
           Tensor with mask
        """
        # macaron feed forward
        if self.ff_macaron:
            x = self._forward_ff_macaron(x)

        # multihead attention
        x = self._forward_self_attn(x, pos_emb, mask)

        # convolutional blocks
        x = self._forward_convs(x)

        # feed-forward block
        x = self._forward_ff(x)

        # output norm
        if self.out_lnorm:
            x = self.norm_out(x)

        return x, mask
