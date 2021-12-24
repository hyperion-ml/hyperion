"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
#

import torch
import torch.nn as nn

from ..layers.attention import *
from .transformer_feedforward import *
from .conformer_conv import ConformerConvBlock


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
                      that query q_i only attents to key k_j when j<=i
      conv_norm_layer: norm layer constructor for conv block,
                       if None it uses BatchNorm
      se_r:         Squeeze-Excitation compression ratio,
                    if None it doesn't use Squeeze-Excitation
      ff_macaron: if True, it uses macaron-net style ff layers, otherwise transformer style.
      out_lnorm: if True, use LNorm layer at the output as in the conformer paper,
                 we think that this layer is redundant and put it to False by default
      concat_after: if True, if concats attention input and output and apply linear transform, i.e.,
                             y = x + linear(concat(x, att(x)))
                    if False, y = x + att(x)

    """

    def __init__(
        self,
        num_feats,
        self_attn,
        num_heads,
        conv_repeats=1,
        conv_kernel_size=31,
        conv_stride=1,
        feed_forward="linear",
        d_ff=2048,
        ff_kernel_size=3,
        hid_act="swish",
        dropout_rate=0,
        att_context=25,
        att_dropout_rate=0,
        pos_enc_type="rel",
        causal_pos_enc=False,
        conv_norm_layer=None,
        se_r=None,
        ff_macaron=True,
        out_lnorm=False,
        concat_after=False,
    ):

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

    @staticmethod
    def _make_att(
        att_type,
        num_feats,
        num_heads,
        context,
        dropout_rate,
        pos_enc_type,
        causal_pos_enc,
    ):
        """Creates multihead attention block from att_type string

        Args:
           att_type: string in ['scaled-dot-prod-att-v1', 'local-scaled-dot-prod-att-v1']
           num_feats: input/output feat. dimension (aka d_model)
           num_heads: number of heads
           dropout_rate: dropout rate for attention block
           pos_enc_type: type of positional encoder
           causal_pos_enc: if True, use causal positional encodings (when rel_pos_enc=True), it assumes
                           that query q_i only attents to key k_j when j<=i

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
                    time_dim=1,
                )

            return ScaledDotProdAttV1(
                num_feats, num_feats, num_heads, d_k, d_k, dropout_rate, time_dim=1
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
                    time_dim=1,
                )

            return LocalScaledDotProdAttV1(
                num_feats,
                num_feats,
                num_heads,
                d_k,
                d_k,
                context,
                dropout_rate,
                time_dim=1,
            )

    @staticmethod
    def _make_ff(ff_type, num_feats, hid_feats, kernel_size, activation, dropout_rate):
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

    def forward(self, x, pos_emb=None, mask=None):
        """Forward pass function

        Args:
          x: input tensor with size=(batch, time, num_feats)
          pos_emb: positional embedding size=(batch, time2, in_feats) as R_{L-1}, ..., R_0,
                   when using relative postional encoder, otherwise None
          mask: mask to indicate valid time steps for x (batch, time)

        Returns:
           Tensor with output features
           Tensor with mask
        """

        # macaron feed forward
        if self.ff_macaron:
            residual = x
            x = self.norm_ff_macaron(x)
            x = self.feed_forward_macaron(x)
            if self.dropout_rate > 0:
                x = self.dropout(x)
            x = residual + self.ff_scale * x

        # multihead attention
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

        # convolutional blocks
        x = x.transpose(1, 2)
        for block in range(len(self.conv_blocks)):
            x = self.conv_blocks[block](x)

        x = x.transpose(1, 2)

        # feed-forward block
        residual = x
        x = self.norm_ff(x)
        x = self.feed_forward(x)
        if self.dropout_rate > 0:
            x = self.dropout(x)

        x = residual + self.ff_scale * x

        # output norm
        if self.out_lnorm:
            x = self.norm_out(x)

        return x, mask
