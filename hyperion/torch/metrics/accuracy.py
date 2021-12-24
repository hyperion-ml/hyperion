"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch

from .metrics import TorchMetric
from .accuracy_functional import *


class CategoricalAccuracy(TorchMetric):
    def __init__(self, weight=None, reduction="mean"):
        super(CategoricalAccuracy, self).__init__(weight=weight, reduction=reduction)

    def forward(self, input, target):
        return categorical_accuracy(
            input, target, weight=self.weight, reduction=self.reduction
        )


class BinaryAccuracy(TorchMetric):
    def __init__(self, weight=None, reduction="mean", thr=0.5):
        super(BinaryAccuracy, self).__init__(weight=weight, reduction=reduction)
        self.thr = thr

    def forward(self, input, target):
        return binary_accuracy(
            input, target, weight=self.weight, reduction=self.reduction, thr=self.thr
        )


class BinaryAccuracyWithLogits(TorchMetric):
    def __init__(self, weight=None, reduction="mean", thr=0.0):
        super(BinaryAccuracyWithLogits, self).__init__(
            weight=weight, reduction=reduction
        )
        self.thr = thr

    def forward(self, input, target):
        return binary_accuracy_with_logits(
            input, target, weight=self.weight, reduction=self.reduction, thr=self.thr
        )
