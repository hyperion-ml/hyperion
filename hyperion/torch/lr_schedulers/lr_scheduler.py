"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch.optim as optim

MinLR = Union[float, Sequence[float]]


class LRScheduler:
    """Base class for project learning-rate schedulers.

    This scheduler supports optional linear warmup from a near-zero value to
    each parameter group's base learning rate.

    Attributes:
      optimizer: Wrapped optimizer.
      min_lrs: Per-parameter-group minimum learning rates.
      base_lrs: Per-parameter-group base (initial) learning rates.
      warmup_steps: Number of optimization steps used for linear warmup.
      epoch: Current epoch index.
      step: Current optimization-step index.
      update_lr_on_opt_step: If ``True``, update LR on each optimizer step;
        otherwise update LR on epoch boundaries.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        min_lr: MinLR = 0,
        warmup_steps: int = 0,
        epoch: int = 0,
        step: int = 0,
        update_lr_on_opt_step: bool = True,
    ) -> None:
        """Initialize scheduler state.

        Args:
            optimizer: Wrapped optimizer.
            min_lr: Scalar or per-parameter-group lower bound.
            warmup_steps: Number of optimization steps used for linear warmup.
            epoch: Initial epoch index (for checkpoint resume).
            step: Initial optimization-step index (for checkpoint resume).
            update_lr_on_opt_step: If ``True``, update LR on each optimizer step;
                otherwise update LR on epoch boundaries.
        """
        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError("%s is not an Optimizer" % (type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(
                    "expected {} min_lrs, got {}".format(
                        len(optimizer.param_groups), len(min_lr)
                    )
                )
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        if epoch == 0:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified "
                        "in param_groups[{}] when resuming an optimizer".format(i)
                    )

        self.base_lrs = list(
            map(lambda group: group["initial_lr"], optimizer.param_groups)
        )
        self.warmup_steps = warmup_steps
        self.epoch = epoch
        self.step = step
        self.update_lr_on_opt_step = update_lr_on_opt_step

    @property
    def in_warmup(self) -> bool:
        """Whether the scheduler is currently in the warmup phase."""
        return self.step < self.warmup_steps

    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state for checkpointing.

        The optimizer object itself is excluded.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load scheduler state from :meth:`state_dict`.

        Args:
            state_dict: Serialized scheduler state.
        """
        self.__dict__.update(state_dict)

    def get_warmup_lr(self) -> List[float]:
        """Compute warmup learning rates for each parameter group."""
        x = self.step
        return [
            (base_lr - min(min_lr, 1e-8)) / self.warmup_steps * x + min(min_lr, 1e-8)
            for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)
        ]

    def get_lr(self, step: int) -> List[float]:
        """Compute learning rates for a given step/epoch index.

        Args:
            step: Current scheduler index (step or epoch depending on usage).
        """
        raise NotImplementedError

    def on_epoch_begin(self, epoch: Optional[int] = None, **kwargs: Any) -> None:
        """Update learning rates at epoch start when configured for epoch updates."""
        if epoch is not None:
            self.epoch = epoch

        if self.update_lr_on_opt_step:
            if self.in_warmup:
                for param_group, lr in zip(
                    self.optimizer.param_groups, self.get_warmup_lr()
                ):
                    param_group["lr"] = lr

            return

        for param_group, lr in zip(
            self.optimizer.param_groups, self.get_lr(self.epoch)
        ):
            param_group["lr"] = lr

    def on_epoch_end(self, metrics: Optional[Mapping[str, Any]] = None) -> None:
        """Advance epoch counter at epoch end."""
        self.epoch += 1

    def on_opt_step(self) -> None:
        """Update learning rates after an optimization step."""
        if self.in_warmup:
            for param_group, lr in zip(
                self.optimizer.param_groups, self.get_warmup_lr()
            ):
                param_group["lr"] = lr
            self.step += 1
            return

        if self.update_lr_on_opt_step:
            for param_group, lr in zip(
                self.optimizer.param_groups, self.get_lr(self.step)
            ):
                param_group["lr"] = lr

        self.step += 1
