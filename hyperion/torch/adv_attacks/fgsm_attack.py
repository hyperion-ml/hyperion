"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import torch

from .adv_attack import AdvAttack

class FGSMAttack(AdvAttack):

    def __init__(self, model, epsilon, loss=None, range_min=None, range_max=None):
        super(FGSMAttack, self).__init__(model, loss, range_min, range_max)
        self.epsilon = epsilon

    def generate(self, input, target):

        input.requires_grad = True
        output = self.model(input)
        loss = self.loss(output, target)

        self.model.zero_grad()
        loss.backward()
        dL_x = input.grad.data
        
        adv_ex = input + self.epsilon * dL_x.sign()
        return self._clamp(adv_ex)

        
