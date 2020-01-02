"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import torch
import torch.nn as nn

from ..layers import PosEncoder

class TransformerConv2dSubsampler(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).
    :param int in_feats: input dim
    :param int out_feats: output dim
    :param flaot dropout_rate: dropout rate
    """

    def __init__(self, in_feats, out_feats, dropout_rate, time_dim=1):
        """Construct an Conv2dSubsampling object."""
        super(TransformerConv2dSubsampler, self).__init__()
        self.time_dim = time_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_feats, 3, 2),
            nn.ReLU6(),
            nn.Conv2d(out_feats, out_feats, 3, 2),
            nn.ReLU6()
        )
        self.out = nn.Sequential(
            nn.Linear(out_feats * (((in_feats - 1) // 2 - 1) // 2), out_feats),
            PosEncoder(out_feats, dropout_rate)
        )

    def forward(self, x, mask):
        """Subsample x.
        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        if self.time_dim == 1:
            x = x.transpose(1,2)

        x = x.unsqueeze(1)  # (b, c, f, t)
        x = self.conv(x)
        b, c, f, t = x.size()
        x = self.out(x.contiguous().view(b, c * f, t)).transpose(1,2)
        if mask is None:
            return x, None
        return x, mask[:, :, :-2:2][:, :, :-2:2]
