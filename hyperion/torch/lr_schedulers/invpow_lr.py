"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""


import torch

from .lr_scheduler import LRScheduler


class InvPowLR(LRScheduler):
    """inverse power learning rate scheduler."""

    def __init__(
        self,
        optimizer,
        power=0.5,
        hold_steps=0,
        min_lr=0,
        warmup_steps=0,
        epoch=0,
        step=0,
        update_lr_on_opt_step=False,
    ):
        super(InvPowLR, self).__init__(
            optimizer, min_lr, warmup_steps, epoch, step, update_lr_on_opt_step
        )
        self.power = power
        self.hold_steps = max(hold_steps, self.warmup_steps)

    def get_lr(self, step):
        if step < self.hold_steps:
            return self.base_lrs

        x = step / self.hold_steps
        return [
            max(min_lr, base_lr * x ** (-self.power))
            for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)
        ]

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # we only load step and epoch so we can change the scheduler params during training
        self.step = state_dict["step"]
        self.epoch = state_dict["epoch"]
        # self.__dict__.update(state_dict)
