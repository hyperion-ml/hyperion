"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import torch

from .adv_attack import AdvAttack

class FGSMAttack(AdvAttack):

    def __init__(self, model, eps, loss=None, targeted=False, range_min=None, range_max=None):
        super(FGSMAttack, self).__init__(
            model, loss, targeted, range_min, range_max)
        self.eps = eps

    def generate(self, input, target):

        input.requires_grad = True
        output = self.model(input)
        loss = self.loss(output, target)
        self.model.zero_grad()
        loss.backward()
        dL_x = input.grad.data
        
        f = 1
        if self.targeted:
            f = -1

        adv_ex = input + f * self.eps * dL_x.sign()
        return self._clamp(adv_ex)

        
