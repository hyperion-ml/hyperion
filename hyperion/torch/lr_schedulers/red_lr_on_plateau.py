"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from functools import partial

import torch
from torch._six import inf

from .lr_scheduler import LRScheduler


class ReduceLROnPlateau(LRScheduler):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Attributes:
     optimizer (Optimizer): optimizer.
     monitor: which metric to monitor.
     mode (str): One of `min`, `max`. In `min` mode, lr will
         be reduced when the quantity monitored has stopped
         decreasing; in `max` mode it will be reduced when the
         quantity monitored has stopped increasing. Default: 'min'.
     factor (float): Factor by which the learning rate will be
         reduced. new_lr = lr * factor. Default: 0.1.
     patience (int): Number of epochs with no improvement after
         which learning rate will be reduced. For example, if
         `patience = 2`, then we will ignore the first 2 epochs
         with no improvement, and will only decrease the LR after the
         3rd epoch if the loss still hasn't improved then.
         Default: 10.
     threshold (float): Threshold for measuring the new optimum,
         to only focus on significant changes. Default: 1e-4.
     threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
         dynamic_threshold = best * ( 1 + threshold ) in 'max'
         mode or best * ( 1 - threshold ) in `min` mode.
         In `abs` mode, dynamic_threshold = best + threshold in
         `max` mode or best - threshold in `min` mode. Default: 'rel'.
     cooldown (int): Number of epochs to wait before resuming
         normal operation after lr has been reduced. Default: 0.
     min_lr (float or list): A scalar or a list of scalars. A
         lower bound on the learning rate of all param groups
         or each group respectively. Default: 0.
     warmup_steps: number of warm up steps to get the lr from 0 to the maximum lr.
     eps (float): Minimal decay applied to lr. If the difference
         between new and old lr is smaller than eps, the update is
         ignored. Default: 1e-8.
    """

    def __init__(
        self,
        optimizer,
        monitor="val_loss",
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        warmup_steps=0,
        eps=1e-8,
    ):
        super().__init__(
            optimizer,
            min_lr,
            warmup_steps,
            epoch=0,
            step=0,
            update_lr_on_opt_step=False,
        )

        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.factor = factor

        self.monitor = monitor
        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def on_opt_step(self):
        self.step = self.step + 1
        if self.in_warmup:
            for param_group, lr in zip(
                self.optimizer.param_groups, self.get_warmup_lr()
            ):
                param_group["lr"] = lr
            return

    def on_epoch_begin(self, epoch=None):
        if epoch is not None:
            self.epoch = epoch

    def on_epoch_end(self, metrics=None):
        current = metrics[self.monitor]
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(self.epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self.epoch += 1

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr
                logging.info(
                    "Epoch {:5d}: reducing learning rate"
                    " of group {} to {:.4e}.".format(epoch, i, new_lr)
                )

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == "min" and threshold_mode == "rel":
            rel_epsilon = 1.0 - threshold
            return a < best * rel_epsilon

        elif mode == "min" and threshold_mode == "abs":
            return a < best - threshold

        elif mode == "max" and threshold_mode == "rel":
            rel_epsilon = threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        if mode == "min":
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(
            mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode
        )
