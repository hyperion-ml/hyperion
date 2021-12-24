"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""


import math
import logging

import torch

from .lr_scheduler import LRScheduler


class CosineLR(LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(
        self,
        optimizer,
        T,
        T_mul=1,
        min_lr=0,
        warmup_steps=0,
        warm_restarts=False,
        gamma=1,
        last_restart=0,
        num_restarts=0,
        epoch=0,
        step=0,
        update_lr_on_opt_step=False,
    ):

        super(CosineLR, self).__init__(
            optimizer, min_lr, warmup_steps, epoch, step, update_lr_on_opt_step
        )
        self.T = T
        self.T_mul = T_mul
        self.warm_restarts = warm_restarts
        self.last_restart = last_restart
        self.num_restarts = num_restarts
        self.gamma = gamma

    def on_epoch_begin(self, epoch=None, epoch_updates=1, **kwargs):
        super(CosineLR, self).on_epoch_begin(epoch)
        if self.update_lr_on_opt_step:
            # T has to correspond to an integer number of epochs
            T = int(math.ceil(self.T / epoch_updates) * epoch_updates)
            if self.T != T:
                logging.info("readjusting cos_lr T %d -> %d" % (self.T, T))
                self.T = T

    def get_lr(self, step):
        x = step - self.last_restart
        # if x >= self.T and self.update_lr_on_opt_step and self.warm_restarts:
        #     #T has to be at least 1 epoch
        #     if self.epoch == 0:
        #         self.T = x + 1
        #         logging.info('readjusting cos_lr T to %d' % (self.T))
        # logging.info('cos-get-lr step=%d last=%d T=%d' % (step, self.last_restart, self.T))
        if x >= self.T:
            if self.warm_restarts:
                self.last_restart = step
                x = 0
                self.T *= self.T_mul
                self.num_restarts += 1
                logging.info(
                    "cos_lr warm-restart=%d T=%d" % (self.num_restarts, self.T)
                )
            else:
                return self.min_lrs

        alpha = self.gamma ** self.num_restarts
        r = math.pi / self.T

        return [
            eta_min + (alpha * eta_max - eta_min) * (1 + math.cos(r * x)) / 2
            for eta_max, eta_min in zip(self.base_lrs, self.min_lrs)
        ]

    # def epoch_end_step(self, metrics=None):
    #     if self.epoch==0 and self.update_lr_on_opt_step and self.warm_restarts:
    #         # assures that T period is equal to integer number of epochs
    #         self.T = math.ceil(self.T/self.step)*self.step
    #         logging.info('readjusting cos_lr T to %d' % (self.T))


class AdamCosineLR(CosineLR):
    def __init__(
        self,
        optimizer,
        T=1,
        T_mul=2,
        warmup_steps=0,
        warm_restarts=False,
        gamma=1,
        last_restart=0,
        num_restarts=0,
        epoch=-1,
        step=-1,
        update_lr_on_opt_step=False,
    ):
        super(AdamCosineLR, super).__init__(
            optimizer,
            T,
            T_mul,
            0,
            warmup_steps,
            warm_restarts,
            last_restart,
            num_restarts,
            gamma,
            epoch,
            step,
            update_lr_on_opt_step,
        )

    def get_lr(self, step):
        x = step - self.last_restart
        if x > self.T:
            if self.warm_restarts:
                self.last_restart = step
                x = 0
                self.T *= T_mul
                self.num_restarts += 1
            else:
                return self.min_lrs

        alpha = gamma ** self.num_restarts
        r = math.pi / self.T

        return [
            alpha * base_lr * 0.5 * (1 + math.cos(r * x)) for base_lr in self.base_lrs
        ]
