"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np

import torch.nn as nn

from ..hyper_torch_model import HyperTorchModel


class NetArch(HyperTorchModel):
    def in_context(self):
        return 0

    def in_dim(self):
        return len(self.in_shape())

    def out_dim(self):
        return len(self.out_shape())

    def in_shape(self):
        raise NotImplementedError()

    def out_shape(self, in_shape=None):
        raise NotImplementedError()
