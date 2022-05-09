"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import math
from turtle import up
import torch

from .invpow_lr import InvPowLR


class NoamLR(InvPowLR):
    """Optimizer used for Transformers in
    Attention is all You Need: https://arxiv.org/pdf/1706.03762.pdf

    This is Inverse Power Law decay scheduler with parameters that depend on
    the transformer hidden dimension.

    Attributes:

    """

    def __init__(
        self,
        optimizer,
        d_model,
        lr_factor=1,
        min_lr=0,
        warmup_steps=0,
        epoch=0,
        step=0,
    ):
        lr = lr_factor / math.sqrt(d_model * warmup_steps)
        print("noam_lr", lr, flush=True)
        for group in optimizer.param_groups:
            group["lr"] = lr
        super().__init__(
            optimizer,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            epoch=epoch,
            step=step,
            update_lr_on_opt_step=True,
        )
