"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
from typing import Sequence, Union

import torch

from .wd_scheduler import WDScheduler


class CosineWD(WDScheduler):
    r"""Set the weight decay of each parameter group using a cosine

    Attributes:
      optimizer: Pytorch optimizer object.
      initial_wd: initial value of the weight decay.
      warmup_steps: number of warm up steps to get the the weight decay to its final value.
      epoch: initial training training epoch, this is needed to restart the model
             training.
      step: initial training step, this is needed to restart the model training.
      update_wd_on_opt_step: if True, updates the weight decay each time we update the model,
        otherwise after each epoch.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_wd: Union[float, Sequence[float]] = 0,
        warmup_steps: int = 0,
        epoch: int = 0,
        step: int = 0,
        update_wd_on_opt_step: bool = False,
    ) -> None:
        super().__init__(
            optimizer, initial_wd, warmup_steps, epoch, step, update_wd_on_opt_step
        )

    def get_wd(self, step: int) -> Sequence[float]:
        if step >= self.warmup_steps:
            return self.final_wds

        r = math.pi / self.warmup_steps
        return [
            final_wd + (init_wd - final_wd) * (1 + math.cos(r * step)) / 2
            for init_wd, final_wd in zip(self.initial_wds, self.final_wds)
        ]
