"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from __future__ import absolute_import

import torch

from .lr_scheduler import LRScheduler

class ExponentialLR(LRScheduler):
    """Exponential learning rate scheduler.
    """
    def __init__(self, optimizer, decay_rate, decay_steps, hold_steps,
                 min_lr=0, warmup_steps=0,
                 last_epoch=-1, last_batch=-1, update_lr_on_batch=False):
        super(ExponentialLR, self).__init__(
            optimizer, min_lr, warmup_steps,
            last_epoch, last_batch, update_lr_on_batch)
        self.decay_rate = decay_rate 
        self.decay_steps = decay_steps
        self.hold_steps = hold_steps


    def get_lr(self, step):
        if step < self.hold_steps:
            return self.base_lrs

        x = step - self.hold_steps
        return [max(
            min_lr,
            base_lr * self.decay_rate ** (x/self.decay_steps))
                for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)]
