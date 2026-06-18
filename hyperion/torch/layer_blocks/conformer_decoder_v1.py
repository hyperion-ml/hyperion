"""
Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

#

from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..layers.attention_v1 import *
from .conformer_conv import ConformerConvBlock
from .conformer_encoder_v1 import ConformerEncoderBlockV1
from .transformer_feedforward import *


class ConformerDecoderBlockV1(ConformerEncoderBlockV1):
    """Building block for conformer decoder based on conformer encoder introduced in
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
        cross_attn: Any,
        num_heads: int,
        conv_repeats: int = 0,
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
        src_lnorm: bool = False,
        out_lnorm: bool = False,
        concat_after: bool = False,
    ) -> None:
        """Initialize the conformer decoder block.

        Args:
          num_feats: Input/output feature dimension.
          self_attn: Self-attention constructor or module config.
          cross_attn: Cross-attention constructor or module config.
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
          src_lnorm: Whether to layer-normalize the source branch.
          out_lnorm: Whether to apply an output layer norm.
          concat_after: Whether to concatenate attention input/output.
        """
        super().__init__(
            num_feats,
            self_attn,
            num_heads,
            conv_repeats=conv_repeats,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            feed_forward=feed_forward,
            d_ff=d_ff,
            ff_kernel_size=ff_kernel_size,
            hid_act=hid_act,
            dropout_rate=dropout_rate,
            att_context=att_context,
            att_dropout_rate=att_dropout_rate,
            pos_enc_type=pos_enc_type,
            causal_pos_enc=causal_pos_enc,
            conv_norm_layer=conv_norm_layer,
            se_r=se_r,
            ff_macaron=ff_macaron,
            out_lnorm=out_lnorm,
            concat_after=concat_after,
        )

        self.cross_attn = self._make_att(
            cross_attn,
            num_feats,
            num_heads,
            0,
            att_dropout_rate,
            "no",
            False,
        )

        self.norm_cross_att = nn.LayerNorm(num_feats)
        self.src_lnorm = src_lnorm
        if src_lnorm:
            self.norm_src = nn.LayerNorm(num_feats)

        if self.concat_after:
            self.cross_concat_linear = nn.Linear(num_feats + num_feats, num_feats)

    def _forward_self_attn(
        self,
        x: torch.Tensor,
        pos_emb: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply self-attention, optionally using the cached prefix.

        Args:
          x: Input tensor with shape ``(batch, time, features)``.
          pos_emb: Optional positional embedding tensor.
          mask: Optional valid-frame mask.
          cache: Optional cached previous self-attention state.

        Returns:
          Tensor with the same shape as ``x``.
        """
        residual = x
        x = self.norm_att(x)

        if cache is None:
            x_q = x
            mask_q = mask
            x_kv = x
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert_cache_shape = (x.size(0), x.size(1) - 1, x.size(2))
            assert (
                cache.shape == assert_cache_shape
            ), f"{cache.shape} != {assert_cache_shape}"
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask_q = None if mask is None else mask[:, -1:, :]
            x_kv = torch.cat((cache, x_q), dim=1)

        if pos_emb is None:
            x_att = self.self_attn(x_q, x_kv, x_kv, mask=mask_q)
        else:
            x_att = self.self_attn(x_q, x_kv, x_kv, pos_emb=pos_emb, mask=mask_q)

        if self.concat_after:
            x = torch.cat((x_q, x_att), dim=-1)
            x = self.concat_linear(x)
        else:
            x = x_att

        if self.dropout_rate > 0:
            x = self.dropout(x)

        x = residual + x
        return x

    def _forward_cross_attn(
        self,
        x: torch.Tensor,
        x_src: torch.Tensor,
        pos_emb: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply cross-attention with residual connection.

        Args:
          x: Decoder state tensor with shape ``(batch, time, features)``.
          x_src: Source tensor with shape ``(batch, time_src, features)``.
          pos_emb: Optional positional embedding tensor.
          mask: Optional source mask.

        Returns:
          Tensor with the same shape as ``x``.
        """
        residual = x
        x = self.norm_cross_att(x)
        if self.src_lnorm:
            x_src = self.norm_src(x_src)

        if pos_emb is None:
            x_att = self.cross_attn(x, x_src, x_src, mask=mask)
        else:
            x_att = self.cross_attn(x, x_src, x_src, pos_emb=pos_emb, mask=mask)

        if self.concat_after:
            x = torch.cat((x, x_att), dim=-1)
            x = self.cross_concat_linear(x)
        else:
            x = x_att

        if self.dropout_rate > 0:
            x = self.dropout(x)

        x = residual + x
        return x

    def forward(
        self,
        x: torch.Tensor,
        x_src: torch.Tensor,
        pos_emb: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        mask_src: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
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
        x = self._forward_self_attn(x, pos_emb, mask, cache=cache)
        x = self._forward_cross_attn(x, x_src, mask=mask_src)

        # convolutional blocks
        x = self._forward_convs(x)

        # feed-forward block
        x = self._forward_ff(x)

        # output norm
        if self.out_lnorm:
            x = self.norm_out(x)

        return x, mask
