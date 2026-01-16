"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional, Sequence, Union

import torch
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import filter_func_args
from .cos_wd import CosineWD
from .wd_scheduler import WDScheduler


class WDSchedulerFactory:
    def create(
        optimizer: torch.optim.Optimizer,
        wdsch_type: str,
        initial_wd: Optional[Union[float, Sequence[float]]] = None,
        warmup_steps: int = 0,
        update_wd_on_opt_step: bool = True,
    ) -> Optional[WDScheduler]:
        """Creates a weight decay scheduler object.

        Args:
          optimizer: Pytorch optimizer object.
          wdsched_type: type of scheduler in ["none", "cos_wd"].
          initial_wd: inital value of weight decay
          warmup_steps: steps until reaching final weight decay
          update_wd_on_opt_step: if True, updates the wd each time we update the model,
                otherwise after each epoch.
        """

        if wdsch_type == "none":
            return None

        if wdsch_type == "cos_wd":
            return CosineWD(
                optimizer,
                initial_wd=initial_wd,
                warmup_steps=warmup_steps,
                update_wd_on_opt_step=update_wd_on_opt_step,
            )

        raise ValueError(f"invalid wdsch_type={wdsch_type}")

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        return filter_func_args(WDSchedulerFactory.create, kwargs)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--wdsch-type",
            type=str.lower,
            default="none",
            choices=[
                "none",
                "cos_wd",
            ],
            help=(
                "Weight decay scheduler type: none (no schedule) or cos_wd "
                "(cosine annealing)."
            ),
        )

        parser.add_argument(
            "--initial-wd",
            default=None,
            type=float,
            help=(
                "Initial weight decay value; if set, it should be lower than the "
                "final value defined in the optimizer param groups."
            ),
        )

        parser.add_argument(
            "--warmup-steps",
            default=0,
            type=int,
            help="Number of warmup steps to reach the final weight decay value.",
        )

        parser.add_argument(
            "--update-wd-on-opt-step",
            default=True,
            action=ActionYesNo,
            help=(
                "Update weight decay every optimizer step instead of once per epoch."
            ),
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
