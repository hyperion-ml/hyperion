"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnf


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super().__init__()
        self.interp = nnf.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def __repr__(self):
        s = "{}(scale_factor={}, mode={})".format(
            self.__class__.__name__,
            self.scale_factor,
            self.mode,
        )
        return s

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
