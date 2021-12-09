"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout2d


class Dropout1d(Dropout2d):
    def forward(self, inputs):
        x = torch.unsqueeze(inputs, dim=-2)
        x = F.dropout2d(x, self.p, self.training, self.inplace)
        return torch.squeeze(x, dim=-2)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "{}(p={})".format(self.__class__.__name__, self.p)
        return s


class DropConnect2d(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "{}(p={})".format(self.__class__.__name__, self.p)
        return s

    def forward(self, inputs):
        if not self.training:
            return inputs

        batch_size = inputs.shape[0]
        keep_prob = 1 - self.p
        random_tensor = (
            torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
            + keep_prob
        )
        binary_tensor = torch.floor(random_tensor)
        output = inputs / keep_prob * binary_tensor
        return output


class DropConnect1d(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "{}(p={})".format(self.__class__.__name__, self.p)
        return s

    def forward(self, inputs):
        if not self.training:
            return inputs

        batch_size = inputs.shape[0]
        keep_prob = 1 - self.p
        random_tensor = (
            torch.rand([batch_size, 1, 1], dtype=inputs.dtype, device=inputs.device)
            + keep_prob
        )
        binary_tensor = torch.floor(random_tensor)
        output = inputs / keep_prob * binary_tensor
        return output
