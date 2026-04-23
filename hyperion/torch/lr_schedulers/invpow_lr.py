"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, List, Mapping

import torch.optim as optim

from .lr_scheduler import LRScheduler, MinLR


class InvPowLR(LRScheduler):
    """Inverse-power learning-rate decay scheduler.

    After ``hold_steps``, the learning rate decays proportionally to
    ``(step / hold_steps) ** (-power)``.

    Attributes:
      optimizer: Wrapped optimizer.
      power: Exponent controlling inverse-power decay.
      hold_steps: Number of steps to hold base LR before decaying.
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
        power: float = 0.5,
        hold_steps: int = 1,
        min_lr: MinLR = 0,
        warmup_steps: int = 0,
        epoch: int = 0,
        step: int = 0,
        update_lr_on_opt_step: bool = True,
    ) -> None:
        """Initialize inverse-power scheduler.

        Args:
            optimizer: Wrapped optimizer.
            power: Exponent controlling inverse-power decay.
            hold_steps: Steps to hold base LR before decay begins.
            min_lr: Scalar or per-group lower LR bound.
            warmup_steps: Linear warmup duration in optimizer steps.
            epoch: Initial epoch index.
            step: Initial optimization-step index.
            update_lr_on_opt_step: Whether to update LR every optimizer step.
        """
        super().__init__(
            optimizer, min_lr, warmup_steps, epoch, step, update_lr_on_opt_step
        )
        self.power = power
        if hold_steps <= 0:
            raise ValueError(f"hold_steps must be > 0, got {hold_steps}")
        self.hold_steps = max(hold_steps, self.warmup_steps)

    def get_lr(self, step: int) -> List[float]:
        """Return per-group learning rates for the provided step."""
        if step < self.hold_steps:
            return self.base_lrs

        x = step / self.hold_steps
        return [
            max(min_lr, base_lr * x ** (-self.power))
            for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)
        ]

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load resume counters while allowing scheduler hyperparameter changes.

        Args:
            state_dict: Serialized scheduler state.
        """
        # Only load counters to allow changing scheduler params mid-training.
        self.step = state_dict["step"]
        self.epoch = state_dict["epoch"]
