"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import torch

from .adv_attack import AdvAttack

class RandFGSMAttack(AdvAttack):

    def __init__(self, model, eps, alpha, loss=None, targeted=False, range_min=None, range_max=None):
        super(RandFGSMAttack, self).__init__(
            model, loss, targeted, range_min, range_max)

        assert alpha < eps, 'alpha({}) >= eps({})'.format(alpha, eps)
        self.eps = eps
        self.alpha = alpha


    def generate(self, input, target):

        x = input + self.alpha * torch.randn_like(input).sign()
        x.requires_grad = True

        output = self.model(x)
        loss = self.loss(output, target)
        self.model.zero_grad()
        loss.backward()
        dL_x = x.grad.data
        
        f = 1
        if self.targeted:
            f = -1

        adv_ex = x + f * (self.eps - self.alpha) * dL_x.sign()
        return self._clamp(adv_ex)

        
