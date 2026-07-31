"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
from typing import Any, List, Optional

import torch.optim as optim

from .lr_scheduler import LRScheduler, MinLR


class CosineLR(LRScheduler):
    r"""Cosine annealing scheduler with optional warm restarts.

    The per-group learning rate follows:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    where ``T_cur`` is the index since the last restart and ``T_max`` is the
    current cycle length.

    Attributes:
      optimizer: Wrapped optimizer.
      T: Current cycle length.
      T_mul: Cycle-length multiplier after each restart.
      min_lrs: Per-parameter-group minimum learning rates.
      base_lrs: Per-parameter-group base learning rates.
      warmup_steps: Number of optimization steps used for linear warmup.
      warm_restarts: Whether warm restarts are enabled.
      gamma: Multiplicative factor applied to max LR after each restart.
      last_restart: Step index of the last restart.
      num_restarts: Number of completed restarts.
      epoch: Current epoch index.
      step: Current optimization-step index.
      update_lr_on_opt_step: Whether LR updates happen per optimizer step.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        T: int,
        T_mul: int = 1,
        min_lr: MinLR = 0,
        warmup_steps: int = 0,
        warm_restarts: bool = False,
        gamma: float = 1,
        last_restart: int = 0,
        num_restarts: int = 0,
        epoch: int = 0,
        step: int = 0,
        update_lr_on_opt_step: bool = True,
    ) -> None:
        """Initialize cosine scheduler.

        Args:
            optimizer: Wrapped optimizer.
            T: Initial cycle length.
            T_mul: Cycle-length multiplier after each restart.
            min_lr: Scalar or per-group lower LR bound.
            warmup_steps: Linear warmup duration in optimizer steps.
            warm_restarts: Whether to restart cycles after ``T``.
            gamma: Multiplicative factor applied to max LR after each restart.
            last_restart: Resume state for last restart step.
            num_restarts: Resume state for number of completed restarts.
            epoch: Initial epoch index.
            step: Initial optimization-step index.
            update_lr_on_opt_step: Whether to update LR every optimizer step.
        """
        if T <= 0:
            raise ValueError(f"T must be > 0, got {T}")
        if T_mul <= 0:
            raise ValueError(f"T_mul must be > 0, got {T_mul}")

        super().__init__(
            optimizer, min_lr, warmup_steps, epoch, step, update_lr_on_opt_step
        )
        self.T = T
        self.T_mul = T_mul
        self.warm_restarts = warm_restarts
        self.last_restart = last_restart
        self.num_restarts = num_restarts
        self.gamma = gamma

    def on_epoch_begin(
        self,
        epoch: Optional[int] = None,
        save_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Adjust cycle length to align to epoch boundaries when requested."""
        super().on_epoch_begin(epoch)
        if self.update_lr_on_opt_step and save_steps is not None:
            if save_steps <= 0:
                raise ValueError(f"save_steps must be > 0, got {save_steps}")
            # T has to correspond to an integer number of epochs
            T = int(math.ceil(self.T / save_steps) * save_steps)
            if self.T != T:
                logging.info("readjusting cos_lr T %d -> %d" % (self.T, T))
                self.T = T

    def get_lr(self, step: int) -> List[float]:
        """Return per-group cosine-annealed learning rates."""
        x = step - self.last_restart
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

        alpha = self.gamma**self.num_restarts
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
    """Cosine scheduler variant used with Adam-family optimizers.

    Attributes:
      optimizer: Wrapped optimizer.
      T: Current cycle length.
      T_mul: Cycle-length multiplier after each restart.
      min_lrs: Per-parameter-group minimum learning rates.
      base_lrs: Per-parameter-group base learning rates.
      warmup_steps: Number of optimization steps used for linear warmup.
      warm_restarts: Whether warm restarts are enabled.
      gamma: Multiplicative factor applied to max LR after each restart.
      last_restart: Step index of the last restart.
      num_restarts: Number of completed restarts.
      epoch: Current epoch index.
      step: Current optimization-step index.
      update_lr_on_opt_step: Whether LR updates happen per optimizer step.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        T: int = 1,
        T_mul: int = 2,
        warmup_steps: int = 0,
        warm_restarts: bool = False,
        gamma: float = 1,
        last_restart: int = 0,
        num_restarts: int = 0,
        epoch: int = 0,
        step: int = 0,
        update_lr_on_opt_step: bool = False,
    ) -> None:
        """Initialize Adam cosine scheduler."""
        super().__init__(
            optimizer=optimizer,
            T=T,
            T_mul=T_mul,
            min_lr=0,
            warmup_steps=warmup_steps,
            warm_restarts=warm_restarts,
            gamma=gamma,
            last_restart=last_restart,
            num_restarts=num_restarts,
            epoch=epoch,
            step=step,
            update_lr_on_opt_step=update_lr_on_opt_step,
        )

    def get_lr(self, step: int) -> List[float]:
        """Return per-group cosine-annealed learning rates."""
        x = step - self.last_restart
        if x > self.T:
            if self.warm_restarts:
                self.last_restart = step
                x = 0
                self.T *= self.T_mul
                self.num_restarts += 1
            else:
                return self.min_lrs

        alpha = self.gamma**self.num_restarts
        r = math.pi / self.T

        return [
            alpha * base_lr * 0.5 * (1 + math.cos(r * x)) for base_lr in self.base_lrs
        ]
