"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn


class TorchMetric(nn.Module):
    """Base class for metrics that cannot be
    objective functions
    """

    def __init__(self, weight=None, reduction="mean"):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
