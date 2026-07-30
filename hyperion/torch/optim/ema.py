"""
Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

import torch
from jsonargparse import ActionParser, ArgumentParser

ParamGroup = Iterable[torch.Tensor]
ParamsLike = Union[ParamGroup, Iterable[ParamGroup]]


class ExpMovingAvg:
    """Exponential moving average (EMA) updater for one or more parameter groups.

    The updater maintains target parameters (typically teacher parameters) as an
    EMA of source parameters (typically student parameters):
    ``p <- m * p + (1 - m) * p_new``.

    Optionally, momentum can be warmed up from ``init_momentum`` to
    ``momentum`` over ``warmup_steps`` using a cosine schedule.
    """

    def __init__(
        self,
        params: ParamsLike,
        init_momentum: float = 0.996,
        momentum: float = 0.996,
        warmup_steps: int = 0,
        global_step: int = 0,
    ) -> None:
        """Initialize EMA state.

        Args:
            params: Parameter iterable or iterable of parameter-group iterables.
                Target parameters to be updated by EMA.
            init_momentum: Initial momentum at step 0 (used only during warmup).
            momentum: Final momentum after warmup.
            warmup_steps: Number of warmup steps for momentum scheduling.
            global_step: Initial global step value.
        """
        self.params = self._normalize_param_groups(params)
        self.init_momentum = init_momentum
        self._momentum = momentum
        self.warmup_steps = warmup_steps
        self.global_step = global_step

    @staticmethod
    def _normalize_param_groups(params: ParamsLike) -> List[List[torch.Tensor]]:
        """Convert supported parameter inputs to ``list[list[Tensor]]``.

        Accepts either a single parameter iterable (flat) or an iterable of
        parameter iterables (already grouped).
        """
        params_list = list(params)
        if len(params_list) == 0:
            return []

        if isinstance(params_list[0], torch.Tensor):
            return [params_list]

        return [list(group) for group in params_list]

    def state_dict(self) -> Dict[str, int]:
        """Return serializable EMA state for checkpointing."""
        return {"global_step": self.global_step}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load EMA state from ``state_dict``.

        Args:
            state_dict: Mapping returned by :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    @property
    def momentum(self) -> float:
        """Current momentum value, including warmup scheduling."""
        if self.global_step >= self.warmup_steps:
            return self._momentum
        else:
            alpha = (1 + math.cos(self.global_step / self.warmup_steps * math.pi)) / 2
            return self.init_momentum * alpha + self._momentum * (1 - alpha)

    @torch.no_grad()
    def step(self, new_params: ParamsLike) -> None:
        """Apply one EMA update from ``new_params`` to stored parameters.

        Args:
            new_params: Source parameter iterable or iterable of parameter-group
                iterables with the same structure as ``params`` passed to
                :meth:`__init__`.
        """
        new_param_groups = self._normalize_param_groups(new_params)

        assert len(self.params) == len(new_param_groups)
        momentum = self.momentum
        for param_group, new_param_group in zip(self.params, new_param_groups):
            for p, p_new in zip(param_group, new_param_group):
                p.data.mul_(momentum).add_((1 - momentum) * p_new.data)

        self.global_step += 1

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Add EMA-specific CLI arguments to an argument parser.

        Args:
            parser: Destination argument parser.
            prefix: Optional namespace prefix used to group EMA options.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--init-momentum", default=0.996, type=float, help="initial momentum"
        )
        parser.add_argument(
            "--momentum", default=0.996, type=float, help="final momentum"
        )
        parser.add_argument(
            "--warmup-steps", default=0, type=int, help="momentum warmup steps"
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
