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

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
