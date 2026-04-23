"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
from typing import Any, Dict, List, Optional

import torch.optim as optim

from .lr_scheduler import LRScheduler, MinLR


class TriangularLR(LRScheduler):
    r"""Cyclic triangular learning-rate scheduler.

    Implements the schedule proposed in:
    .. Cyclical Learning Rates for Training Neural Networks:
    https://arxiv.org/abs/1506.01186

    .. math::
        \mathrm{cycle} = \mathrm{floor}(1 + \frac{T_{cur}}{T_{max}})
        x = \mathrm{abs}(2\frac{T_{cur}}{T_{max}}-2\mathrm{cycle}+1)
        \eta_t = \eta_{min} + (\eta_{max} - \eta_{min})\max(0, 1-x)

    Attributes:
      optimizer: Wrapped optimizer.
      T: Current cycle length.
      T_mul: Cycle-length multiplier after each restart.
      min_lrs: Per-parameter-group minimum learning rates.
      base_lrs: Per-parameter-group base learning rates.
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
        gamma: float = 1,
        last_restart: int = 0,
        num_restarts: int = 0,
        epoch: int = 0,
        step: int = 0,
        update_lr_on_opt_step: bool = True,
    ) -> None:
        """Initialize triangular scheduler."""
        if T <= 0:
            raise ValueError(f"T must be > 0, got {T}")
        if T_mul <= 0:
            raise ValueError(f"T_mul must be > 0, got {T_mul}")
        super().__init__(optimizer, min_lr, 0, epoch, step, update_lr_on_opt_step)
        self.T = T
        self.T_mul = T_mul
        self.last_restart = last_restart
        self.num_restarts = num_restarts
        self.gamma = gamma

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state while preserving current ``gamma`` and ``T_mul``."""
        # we want to be able to change gamma and T_mul in the middle of training
        del state_dict["gamma"]
        del state_dict["T_mul"]
        super().load_state_dict(state_dict)

    def on_epoch_begin(
        self,
        epoch: Optional[int] = None,
        save_steps: int = 1,
        **kwargs: Any,
    ) -> None:
        """Optionally align cycle length to an integer number of epochs."""
        super().on_epoch_begin(epoch)
        if self.update_lr_on_opt_step and save_steps is not None:
            if save_steps <= 0:
                raise ValueError(f"save_steps must be > 0, got {save_steps}")
            # T has to correspond to an integer number of epochs
            T = int(math.ceil(self.T / save_steps) * save_steps)
            if self.T != T:
                logging.info("readjusting triangular_lr T %d -> %d", self.T, T)
                self.T = T

    def get_lr(self, step: int) -> List[float]:
        """Return per-group triangular-cycle learning rates."""
        x = step - self.last_restart

        if x >= self.T:
            self.last_restart = step
            x = 0
            self.T *= self.T_mul
            self.num_restarts += 1
            logging.info(
                "triangular_lr warm-restart=%d T=%d", self.num_restarts, self.T
            )

        alpha = self.gamma**self.num_restarts
        x = abs(2 * x / self.T - 1)

        return [
            eta_min + (alpha * eta_max - eta_min) * max(0, 1 - x)
            for eta_max, eta_min in zip(self.base_lrs, self.min_lrs)
        ]
