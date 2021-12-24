"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import math

import torch
from torch import nn


class PosEncoder(nn.Module):
    """Positional encoding.

    Attributes:
      num_feats: embedding dim
      dropout_rate: dropout rate
    """

    def __init__(self, num_feats, dropout_rate=0):
        super().__init__()
        self.num_feats = num_feats
        self.dropout_rate = dropout_rate
        self.xscale = math.sqrt(self.num_feats)
        if self.dropout_rate > 0:
            self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "{}(num_feats={}, dropout_rate={})".format(
            self.__class__.__name__, self.num_feats, self.dropout_rate
        )
        return s

    def _pe(self, x, relative=False):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return self.pe

        pe = torch.zeros(x.size(1), self.num_feats)
        if relative:
            # this is for relative positional encoders
            position = torch.arange(
                x.size(1) - 1, -1, -1, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.num_feats, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.num_feats)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)
        return self.pe

    def forward(self, x):
        """Add positional encoding.

        Args:
            x: Input with shape=(batch, time, C)

        Returns:
            x-scaled + pos-encoder
        """
        pe = self._pe(x)
        x = x * self.xscale + pe[:, : x.size(1)]
        if self.dropout_rate > 0:
            return self.dropout(x)
        return x


class RelPosEncoder(PosEncoder):
    """Relative Positional encoding as defined in
       https://arxiv.org/pdf/1901.02860.pdf

       It returns the input and the positional encoder separtely
       so they are mixed in the attention block later.

    Attributes:
      num_feats: embedding dim
      dropout_rate: dropout rate
    """

    def __init__(self, num_feats, dropout_rate=0):
        super().__init__(num_feats, dropout_rate)

    def forward(self, x):
        """Add positional encoding.

        Args:
            x: Input with shape=(batch, time, C)

        Returns:
            x-scaled, pos-encoding
        """

        pe = self._pe(x, relative=True)
        x = x * self.xscale
        # we want embedding  [R_L,..., R_0]
        # while in non relative we want [R_0, ..., R_L]
        pos_emb = self.pe[:, -x.size(1) :]
        # this pos_emb is matrix Q in
        # https://arxiv.org/pdf/1901.02860.pdf Appendix B
        # I think it should have been denoted as R,
        # probably a typo in the paper
        if self.dropout_rate > 0:
            x = self.dropout(x)
            pos_emb = self.dropout(pos_emb)

        return x, pos_emb


class NoPosEncoder(nn.Module):
    """This is a dummy class for the case where we
    deactivate the positional encoder

    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Identity map

        Args:
            x: Input with shape=(batch, time, C)

        Returns:
            x
        """
        return x
