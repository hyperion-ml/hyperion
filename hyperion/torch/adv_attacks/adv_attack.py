"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import torch
import torch.nn as nn

# from ..utils import TorchDataParallel


class AdvAttack(object):
    def __init__(self, model, loss=None, targeted=True, range_min=None, range_max=None):
        self.model = model
        if loss is None:
            loss = nn.CrossEntropyLoss()
        self.loss = loss
        self.range_min = range_min
        self.range_max = range_max
        self.targeted = targeted

    def to(self, device):
        self.model.to(device)

    @property
    def attack_info(self):
        return {"targeted": self.targeted}

    def generate(self, input, target):
        raise NotImplementedError()

    def _clamp(self, adv_ex):
        if self.range_min is not None and self.range_max is not None:
            adv_ex = torch.clamp(adv_ex, min=self.range_min, max=self.range_max)

        return adv_ex
