"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math

import torch.optim as optim

from .invpow_lr import InvPowLR
from .lr_scheduler import MinLR


class NoamLR(InvPowLR):
    """Noam learning-rate schedule from *Attention Is All You Need*.

    This is Inverse Power Law decay scheduler with parameters that depend on
    the transformer hidden dimension.

    Attributes:
      optimizer: Wrapped optimizer.
      power: Exponent controlling inverse-power decay (inherited).
      hold_steps: Number of steps to hold base LR before decaying (inherited).
      min_lrs: Per-parameter-group minimum learning rates.
      base_lrs: Per-parameter-group base learning rates.
      warmup_steps: Number of optimization steps used for linear warmup.
      epoch: Current epoch index.
      step: Current optimization-step index.
      update_lr_on_opt_step: Whether LR updates happen per optimizer step.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        lr_factor: float = 1,
        min_lr: MinLR = 0,
        warmup_steps: int = 1,
        epoch: int = 0,
        step: int = 0,
    ) -> None:
        """Initialize Noam scheduler and rescale optimizer group LRs.

        Args:
            optimizer: Wrapped optimizer.
            d_model: Transformer hidden size.
            lr_factor: Global multiplier for Noam peak LR.
            min_lr: Scalar or per-group lower LR bound.
            warmup_steps: Warmup duration in optimizer steps.
            epoch: Initial epoch index.
            step: Initial optimization-step index.
        """
        if d_model is None or d_model <= 0:
            raise ValueError(f"d_model must be > 0 for NoamLR, got {d_model}")
        if warmup_steps <= 0:
            raise ValueError(f"warmup_steps must be > 0 for NoamLR, got {warmup_steps}")
        lr = lr_factor / math.sqrt(d_model * warmup_steps)
        logging.info("Noam lr=%f", lr)
        # we scale the lr taking account the relative
        # learning rates in the param_groups
        # in order to be able to have different lr for
        # different modules of the model
        max_lr = 0
        for group in optimizer.param_groups:
            max_lr = max(group["lr"], max_lr)
        if max_lr <= 0:
            raise ValueError(
                "NoamLR requires at least one optimizer param_group lr > 0"
            )
        for group in optimizer.param_groups:
            group["lr"] = lr * group["lr"] / max_lr
        super().__init__(
            optimizer,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            epoch=epoch,
            step=step,
            update_lr_on_opt_step=True,
        )
