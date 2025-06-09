"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from ..layers import GatherDistributedFunction
from ..utils import ddp_get_rank


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss with multi-GPU support and optional memory banks.

    Args:
        temp (float): Target temperature scaling value for similarity logits.
                      This is the final value used after warmup.
        temp_warmup_steps (int): Number of steps over which to linearly warm up the temperature
                                 from `initial_temp` to `temp`.
        initial_temp (float): Initial temperature used at the start of training before warmup completes.
    """

    def __init__(
        self,
        temp: float = 0.07,
        temp_warmup_steps: int = 1000,
        initial_temp: float = 0.07,
        margin=0.3,
        margin_warmup_steps=0,
    ):
        super().__init__()
        self.temp = temp
        self.cur_temp = initial_temp
        self.temp_warmup_steps = temp_warmup_steps
        self.initial_temp = initial_temp
        self.margin = margin
        self.margin_warmup_steps = margin_warmup_steps
        self.cur_margin = 0.0

    def update_temp(self, step: int):
        if step < self.temp_warmup_steps:
            self.cur_temp = (
                self.initial_temp
                + (self.temp - self.initial_temp) * step / self.temp_warmup_steps
            )
            logging.info("updating contrastive losss temp=%.3f", self.cur_temp)
        else:
            self.cur_temp = self.temp

    def update_margin(self, step: int):
        """Updates the value of the margin.

        Args:
          step: value of current step.
        """

        if step < self.margin_warmup_steps:
            self.cur_margin = self.margin * step / self.margin_warmup_steps
            logging.info(
                "updating constrastive loss margin=%.2f",
                self.cur_margin,
            )
        else:
            if self.cur_margin != self.margin:
                self.cur_margin = self.margin
                logging.info(
                    "updating constrastive loss margin=%.2f",
                    self.cur_margin,
                )
            else:
                return

    def forward(
        self,
        z_pred: torch.Tensor,
        z_true: torch.Tensor,
        z_negatives: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Compute cross-modal contrastive loss with distributed gathering and optional memory banks.

        Args:
            z_pred (Tensor): Predicted embeddings, shape (B, D)
            z_true (Tensor): Ground Truth embeddings, shape (B, D)
            z_negatives (Optional[Tensor]): Additional image z_negatives (K_img, D)

        Returns:
            Tensor: Scalar contrastive loss.
        """
        B, D = z_pred.shape

        # Normalize local features
        z_pred = nn.functional.normalize(z_pred, dim=1)
        z_true = nn.functional.normalize(z_true, dim=1)

        # Gather across distributed GPUs
        z_true_all = GatherDistributedFunction.apply(z_true)  # (G*B, D)

        # Optionally append memory bank z_negatives
        if z_negatives is not None:
            z_true_all = torch.cat(
                [z_true_all, nn.functional.normalize(z_negatives, dim=1)], dim=0
            )

        # Similarity matrices
        logits = torch.matmul(z_pred, z_true_all.T)

        # Labels are position indices: correct match at local position + rank offset
        rank = ddp_get_rank()
        labels = torch.arange(B, device=z_pred.device) + rank * B

        if self.cur_margin > 0.0 and self.training:
            # Apply margin to logits
            logits = logits.clone()
            logits_m = logits - self.cur_margin
            idx_ = torch.arange(
                0, logits.shape[0], dtype=torch.long, device=z_pred.device
            )
            logits[idx_, labels] = logits_m[idx_, labels]

        logits = logits / self.cur_temp  # Scale by temperature

        # Cross-entropy losses
        loss = nn.functional.cross_entropy(logits, labels)
        if return_logits:
            return loss, logits

        return loss
