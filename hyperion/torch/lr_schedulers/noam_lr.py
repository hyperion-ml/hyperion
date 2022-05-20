"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import math
import logging

# import torch

from .invpow_lr import InvPowLR


class NoamLR(InvPowLR):
    """Optimizer used for Transformers in
    Attention is all You Need: https://arxiv.org/pdf/1706.03762.pdf

    This is Inverse Power Law decay scheduler with parameters that depend on
    the transformer hidden dimension.

    Attributes:
      optimizer: Pytorch optimizer object.
      d_model: hidden dimension of transformer model.
      lr_factor: multiplies the Noam lr by this number.
      min_lr: minimum learning rate.
      warmup_steps: number of warm up steps to get the lr from 0 to the maximum lr.
      epoch: initial training training epoch, this is needed to restart the model
             training.
      step: initial training step, this is needed to restart the model training.

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
        logging.info("Noam lr=%f", lr)
        # we scale the lr taking account the relative
        # learning rates in the param_groups
        # in order to be able to have different lr for
        # different modules of the model
        max_lr = 0
        for group in optimizer.param_groups:
            max_lr = max(lr, max_lr)
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
