"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch.optim as optim

InitialWD = Union[float, Sequence[float]]


class WDScheduler:
    """Base class for weight decay schedulers.

    This scheduler supports per-parameter-group weight decay scheduling either
    on optimizer steps or on epoch boundaries.

    Attributes:
      optimizer: Wrapped optimizer.
      initial_wds: Per-parameter-group initial weight decays.
      final_wds: Per-parameter-group final/target weight decays.
      warmup_steps: Number of warmup steps used by scheduler-specific policies.
      epoch: Current epoch index.
      step: Current optimization-step index.
      update_wd_on_opt_step: If ``True``, update WD on optimizer steps; otherwise
        on epoch boundaries.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        initial_wd: InitialWD = 1e-5,
        warmup_steps: int = 0,
        epoch: int = 0,
        step: int = 0,
        update_wd_on_opt_step: bool = False,
    ) -> None:
        """Initialize scheduler state.

        Args:
            optimizer: Wrapped optimizer.
            initial_wd: Scalar or per-group initial weight decay.
            warmup_steps: Number of warmup steps used by scheduler policy.
            epoch: Initial epoch index (for checkpoint resume).
            step: Initial optimization-step index (for checkpoint resume).
            update_wd_on_opt_step: Whether to update WD every optimizer step.
        """
        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError("%s is not an Optimizer" % (type(optimizer).__name__))
        self.optimizer: optim.Optimizer = optimizer

        if epoch == 0:
            for group in optimizer.param_groups:
                group.setdefault("final_wd", group["weight_decay"])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "final_wd" not in group:
                    raise KeyError(
                        "param 'final_wd' is not specified "
                        "in param_groups[{}] when resuming an optimizer".format(i)
                    )

        self.final_wds: List[float] = list(
            map(lambda group: group["final_wd"], optimizer.param_groups)
        )

        if isinstance(initial_wd, list) or isinstance(initial_wd, tuple):
            if len(initial_wd) != len(optimizer.param_groups):
                raise ValueError(
                    "expected {} initial_wds, got {}".format(
                        len(optimizer.param_groups), len(initial_wd)
                    )
                )
            self.initial_wds = list(initial_wd)
        else:
            max_wd = max([group["final_wd"] for group in optimizer.param_groups])
            if max_wd == 0:
                raise ValueError(
                    "max final_wd across optimizer param groups is 0; cannot scale initial_wd"
                )
            self.initial_wds = [
                initial_wd * group["final_wd"] / max_wd
                for group in optimizer.param_groups
            ]

        if epoch == 0:
            for group, wd in zip(optimizer.param_groups, self.initial_wds):
                group["weight_decay"] = wd

        self.warmup_steps: int = warmup_steps
        self.epoch: int = epoch
        self.step: int = step
        self.update_wd_on_opt_step: bool = update_wd_on_opt_step

    @property
    def in_warmup(self) -> bool:
        """Whether the scheduler is currently in warmup."""
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

    def get_wd(self, step: int) -> List[float]:
        """Compute per-group weight decays for a given step/epoch index.

        Args:
            step: Current scheduler index (step or epoch depending on usage).
        """
        raise NotImplementedError

    def on_epoch_begin(self, epoch: Optional[int] = None, **kwargs: Any) -> None:
        """Update WDs at epoch start when configured for epoch updates."""
        if epoch is not None:
            self.epoch = epoch

        if self.update_wd_on_opt_step:
            return

        for param_group, wd in zip(
            self.optimizer.param_groups, self.get_wd(self.epoch)
        ):
            param_group["weight_decay"] = wd

    def on_epoch_end(self, metrics: Optional[Mapping[str, Any]] = None) -> None:
        """Advance epoch counter at epoch end."""
        self.epoch += 1

    def on_opt_step(self) -> None:
        """Update WDs after an optimization step."""
        if self.update_wd_on_opt_step:
            for param_group, wd in zip(
                self.optimizer.param_groups, self.get_wd(self.step)
            ):
                param_group["weight_decay"] = wd

        self.step += 1
