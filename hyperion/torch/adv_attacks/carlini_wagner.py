"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .adv_attack import AdvAttack


class CarliniWagner(AdvAttack):
    def __init__(
        self,
        model,
        confidence=0.0,
        lr=1e-2,
        max_iter=10000,
        abort_early=True,
        initial_c=1e-3,
        norm_time=False,
        time_dim=None,
        use_snr=False,
        targeted=False,
        range_min=None,
        range_max=None,
    ):

        super().__init__(model, None, targeted, range_min, range_max)
        self.confidence = confidence
        self.lr = lr
        self.max_iter = max_iter
        self.abort_early = abort_early
        self.initial_c = initial_c
        self.is_binary = None
        self.box_scale = (self.range_max - self.range_min) / 2
        self.box_bias = (self.range_max + self.range_min) / 2
        self.norm_time = norm_time
        self.time_dim = time_dim
        self.use_snr = use_snr

    @property
    def attack_info(self):
        info = super().attack_info
        new_info = {
            "confidence": self.confidence,
            "lr": self.lr,
            "max_iter": self.max_iter,
            "abort_early": self.abort_early,
            "initial_c": self.initial_c,
            "norm_time": self.norm_time,
            "use_snr": self.use_snr,
        }
        info.update(new_info)
        return info

    @staticmethod
    def atanh(x, eps=1e-6):
        x = (1 - eps) * x
        return 0.5 * torch.log((1 + x) / (1 - x))

    def x_w(self, w):
        return self.box_scale * torch.tanh(w) + self.box_bias

    def w_x(self, x):
        return self.atanh((x - self.box_bias) / self.box_scale)

    def f(self, z, target):
        if self.is_binary:
            z_t = z.clone()
            z_t[target == 0] *= -1
            z_other = 0
        else:
            idx = torch.arange(0, z.shape[0], device=z.device)
            z_t = z[idx, target]
            z_clone = z.clone()
            z_clone[idx, target] = -1e10
            z_other = torch.max(z_clone, dim=-1)[0]

        if self.targeted:
            f = F.relu(z_other - z_t + self.confidence)  # max(0, z_other-z_target+k)
        else:
            f = F.relu(z_t - z_other + self.confidence)  # max(0, z_target-z_other+k)
        return f

    def generate(self, input, target):
        raise NotImplementedError()
