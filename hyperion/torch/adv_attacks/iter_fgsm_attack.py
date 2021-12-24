"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import torch

from .adv_attack import AdvAttack


class IterFGSMAttack(AdvAttack):
    def __init__(
        self,
        model,
        eps,
        alpha,
        loss=None,
        targeted=False,
        range_min=None,
        range_max=None,
    ):
        super().__init__(model, loss, targeted, range_min, range_max)
        self.eps = eps
        self.alpha = alpha
        self.max_iter = int(1.25 * eps / alpha)

    @property
    def attack_info(self):
        info = super().attack_info
        new_info = {
            "eps": self.eps,
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "threat_model": "linf",
            "attack_type": "iter-fgsm",
        }
        info.update(new_info)
        return info

    def generate(self, input, target):

        f = 1
        if self.targeted:
            f = -1

        x = input
        for it in range(self.max_iter):
            x.detach_()
            x.requires_grad = True
            output = self.model(x)
            loss = self.loss(output, target)
            self.model.zero_grad()
            loss.backward()
            dL_x = x.grad.data
            x = x + f * self.alpha * dL_x.sign()
            x = input + torch.clamp(x - input, -self.eps, self.eps)

        return self._clamp(x)
