"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
from typing import List

import torch

from .wd_scheduler import InitialWD, WDScheduler


class CosineWD(WDScheduler):
    r"""Cosine warmup schedule for weight decay.

    For each parameter group, the weight decay transitions from ``initial_wd``
    to ``final_wd`` during warmup:

    .. math::
        wd_t = wd_{final} + (wd_{init} - wd_{final})
        \frac{1 + \cos(\pi t / T_{warmup})}{2}

    Attributes:
      optimizer: Wrapped optimizer.
      initial_wds: Per-parameter-group initial weight decays.
      final_wds: Per-parameter-group final/target weight decays.
      warmup_steps: Number of optimizer steps for cosine transition.
      epoch: Current epoch index.
      step: Current optimization-step index.
      update_wd_on_opt_step: Whether WD updates happen per optimizer step.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_wd: InitialWD = 1e-5,
        warmup_steps: int = 0,
        epoch: int = 0,
        step: int = 0,
        update_wd_on_opt_step: bool = False,
    ) -> None:
        """Initialize cosine weight-decay scheduler.

        Args:
            optimizer: Wrapped optimizer.
            initial_wd: Scalar or per-group initial weight decay.
            warmup_steps: Steps for cosine transition to final WDs.
            epoch: Initial epoch index.
            step: Initial optimization-step index.
            update_wd_on_opt_step: Whether to update WD every optimizer step.
        """
        super().__init__(
            optimizer, initial_wd, warmup_steps, epoch, step, update_wd_on_opt_step
        )

    def get_wd(self, step: int) -> List[float]:
        """Return per-group weight decays for the provided step."""
        if step >= self.warmup_steps:
            return self.final_wds

        r = math.pi / self.warmup_steps
        return [
            final_wd + (init_wd - final_wd) * (1 + math.cos(r * step)) / 2
            for init_wd, final_wd in zip(self.initial_wds, self.final_wds)
        ]
