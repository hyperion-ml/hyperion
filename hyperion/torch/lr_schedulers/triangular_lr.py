"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""


import math
import logging

import torch

from .lr_scheduler import LRScheduler


class TriangularLR(LRScheduler):
    r"""Sets cyclid triangular learning rate schedule as proposed in
    .. Cyclical Learning Rates for Training Neural Networks:
    https://arxiv.org/abs/1506.01186

    .. math::
        \mathrm{cycle} = \mathrm{floor}(1 + \frac{T_{cur}}{T_{max}})
        x = \mathrm{abs}(2\frac{T_{cur}}{T_{max}}-2\mathrm{cycle}+1)
        \eta_t = \eta_{min} + (\eta_{max} - \eta_{min})\max(0, 1-x)

    """

    def __init__(
        self,
        optimizer,
        T,
        T_mul=1,
        min_lr=0,
        gamma=1,
        last_restart=0,
        num_restarts=0,
        epoch=0,
        step=0,
        update_lr_on_opt_step=False,
    ):

        super().__init__(optimizer, min_lr, 0, epoch, step, update_lr_on_opt_step)
        self.T = T
        self.T_mul = T_mul
        self.last_restart = last_restart
        self.num_restarts = num_restarts
        self.gamma = gamma

    def on_epoch_begin(self, epoch=None, epoch_updates=1, **kwargs):
        super().on_epoch_begin(epoch)
        if self.update_lr_on_opt_step:
            # T has to correspond to an integer number of epochs
            T = int(math.ceil(self.T / epoch_updates) * epoch_updates)
            if self.T != T:
                logging.info("readjusting triangular_lr T %d -> %d" % (self.T, T))
                self.T = T

    def get_lr(self, step):
        x = step - self.last_restart

        if x >= self.T:
            self.last_restart = step
            x = 0
            self.T *= self.T_mul
            self.num_restarts += 1
            logging.info(
                "triangular_lr warm-restart=%d T=%d" % (self.num_restarts, self.T)
            )

        alpha = self.gamma ** self.num_restarts
        x = math.abs(2 * x / self.T - 1)

        return [
            eta_min + (alpha * eta_max - eta_min) * math.max(0, 1 - x)
            for eta_max, eta_min in zip(self.base_lrs, self.min_lrs)
        ]
