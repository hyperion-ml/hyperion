"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn

from ..layers import ActivationFactory as AF


class TransformerConv2dSubsampler(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length) Tor transformer

    Attributes:
      in_feats: input feature dimension
      out_feats: Transformer d_model
      hid_act: activation layer object
      pos_enc: positional encoder layer
      time_dim: indicates which is the time dimension in the input tensor
    """

    def __init__(self, in_feats, out_feats, hid_act, pos_enc=None, time_dim=1):
        super().__init__()
        self.time_dim = time_dim
        hid_act = AF.create(hid_act)
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_feats, 3, 2, padding=(0, 1)),
            hid_act,
            nn.Conv2d(out_feats, out_feats, 3, 2, padding=(0, 1)),
            hid_act,
        )

        linear = nn.Linear(out_feats * (((in_feats - 1) // 2 - 1) // 2),
                           out_feats)
        if pos_enc is None:
            self.out = linear
        else:
            self.out = nn.Sequential(linear, pos_enc)

    def forward(self, x, x_mask=None):
        """Forward function.

        Args:
          x: input tensor with size=(batch, time, in_feats)
          x_mask: mask to indicate valid time steps for x (batch, time1, time2)

        Returns:
           Tensor with output features with shape = (batch, time//4, out_feats)
           Tensor with subsampled mask x4.
        """
        if self.time_dim == 1:
            x = x.transpose(1, 2)

        x = x.unsqueeze(1)  # (b, c, f, t)
        x = self.conv(x)
        b, c, f, t = x.size()
        x = self.out(x.contiguous().view(b, c * f, t).transpose(1, 2))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]
