"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""


import torch

from .lr_scheduler import LRScheduler


class ExponentialLR(LRScheduler):
    """Exponential learning rate scheduler.

    Attributes:
      optimizer: Pytorch optimizer object.
      decay_rate: the lr is multiplied by `decay_rate` after `decay_ste.ps`
      decay_steps: number of decay steps.
      hold_steps: number of steps until the lr starts decaying.
      min_lr: minimum learning rate.
      warmup_steps: number of warm up steps to get the lr from 0 to the maximum lr.
      epoch: initial training training epoch, this is needed to restart the model
             training.
      step: initial training step, this is needed to restart the model training.
      update_lr_on_opt_step: if True, updates the lr each time we update the model,
        otherwise after each epoch.
    """

    def __init__(
        self,
        optimizer,
        decay_rate,
        decay_steps,
        hold_steps,
        min_lr=0,
        warmup_steps=0,
        epoch=0,
        step=0,
        update_lr_on_opt_step=False,
    ):
        super().__init__(
            optimizer, min_lr, warmup_steps, epoch, step, update_lr_on_opt_step
        )
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.hold_steps = max(hold_steps, self.warmup_steps)

    def get_lr(self, step):
        if step < self.hold_steps:
            return self.base_lrs

        x = step - self.hold_steps
        return [
            max(min_lr, base_lr * self.decay_rate ** (x / self.decay_steps))
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
