"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import torch

from .adv_attack import AdvAttack


class SNRFGSMAttack(AdvAttack):
    def __init__(
        self, model, snr, loss=None, targeted=False, range_min=None, range_max=None
    ):
        super().__init__(model, loss, targeted, range_min, range_max)
        self.snr = snr

    @property
    def attack_info(self):
        info = super().attack_info
        new_info = {"snr": self.snr, "threat_model": "snr", "attack_type": "snr-fgsm"}
        info.update(new_info)
        return info

    def generate(self, input, target):

        input.requires_grad = True
        output = self.model(input)
        loss = self.loss(output, target)

        self.model.zero_grad()
        loss.backward()
        dL_x = input.grad.data

        dim = tuple(i for i in range(1, input.dim()))
        P_x = 10 * torch.log10(torch.mean(input ** 2, dim=dim, keepdim=True))

        noise = dL_x.sign()
        P_n = 10 * torch.log10(torch.mean(noise ** 2, dim=dim, keepdim=True))

        snr_0 = P_x - P_n
        dsnr = self.snr - snr_0
        eps = 10 ** (-dsnr / 20)

        f = 1
        if self.targeted:
            f = -1

        adv_ex = input + f * eps * noise
        return self._clamp(adv_ex)
