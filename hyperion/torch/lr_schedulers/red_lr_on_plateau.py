"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from math import inf
from functools import partial
from typing import Any, Callable, Mapping, Optional

import torch.optim as optim

from .lr_scheduler import LRScheduler, MinLR


class ReduceLROnPlateau(LRScheduler):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    This implementation mirrors PyTorch's ReduceLROnPlateau behavior while
    keeping compatibility with this project's scheduler interface.

    Attributes:
      optimizer: Wrapped optimizer.
      monitor: Metric key read at epoch end.
      mode: Optimization direction for the monitored metric (``"min"`` or ``"max"``).
      factor: Multiplicative LR reduction factor.
      patience: Number of bad epochs before reducing LR.
      threshold: Improvement threshold.
      threshold_mode: Threshold semantics (``"rel"`` or ``"abs"``).
      cooldown: Cooldown epochs after each LR reduction.
      cooldown_counter: Remaining epochs in cooldown.
      min_lrs: Per-parameter-group minimum learning rates.
      base_lrs: Per-parameter-group base learning rates.
      warmup_steps: Number of optimization steps used for linear warmup.
      best: Best monitored metric value seen so far.
      num_bad_epochs: Number of consecutive non-improving epochs.
      mode_worse: Worst possible metric baseline for the configured mode.
      eps: Minimum effective LR change for applying a reduction.
      epoch: Current epoch index.
      step: Current optimization-step index.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        monitor: str = "val_loss",
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: MinLR = 0,
        warmup_steps: int = 0,
        eps: float = 1e-8,
    ) -> None:
        """Initialize ReduceLROnPlateau scheduler.

        Args:
            optimizer: Wrapped optimizer.
            monitor: Metric key to read from epoch-end metrics.
            mode: ``"min"`` or ``"max"``.
            factor: Multiplicative LR reduction factor (< 1).
            patience: Number of bad epochs before reducing LR.
            threshold: Improvement threshold.
            threshold_mode: ``"rel"`` or ``"abs"`` threshold semantics.
            cooldown: Cooldown epochs after each LR reduction.
            min_lr: Scalar or per-group lower LR bound.
            warmup_steps: Linear warmup duration in optimizer steps.
            eps: Minimum applied LR change.
        """
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
        self.best: float = 0.0
        self.num_bad_epochs: int = 0
        self.mode_worse: float = 0.0  # worst value for selected mode
        self.is_better: Callable[[float, float], bool]
        self.eps = eps
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )
        self._reset()

    def _reset(self) -> None:
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def on_opt_step(self) -> None:
        """Advance one optimization step, applying warmup when active."""
        super().on_opt_step()

    def on_epoch_begin(self, epoch: Optional[int] = None, **kwargs: Any) -> None:
        """Optionally set epoch counter at epoch start."""
        if epoch is not None:
            self.epoch = epoch

    def on_epoch_end(self, metrics: Mapping[str, float]) -> None:
        """Evaluate monitored metric and reduce LR if plateaued.

        Args:
            metrics: Mapping containing the monitored metric key.
        """
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

    def _reduce_lr(self, epoch: int) -> None:
        """Apply LR reduction to optimizer parameter groups."""
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
    def in_cooldown(self) -> bool:
        """Whether the scheduler is in cooldown after a reduction."""
        return self.cooldown_counter > 0

    def _cmp(
        self, mode: str, threshold_mode: str, threshold: float, a: float, best: float
    ) -> bool:
        """Compare metric values according to mode/threshold configuration."""
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

    def _init_is_better(
        self, mode: str, threshold: float, threshold_mode: str
    ) -> None:
        """Initialize comparison baseline and comparator closure."""
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        if mode == "min":
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load scheduler state and rebuild comparator closure."""
        self.__dict__.update(state_dict)
        self._init_is_better(
            mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode
        )
