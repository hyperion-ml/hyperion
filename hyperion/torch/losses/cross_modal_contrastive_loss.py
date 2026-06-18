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


class CrossModalContrastiveLoss(nn.Module):
    """CLIP-style cross-modal contrastive loss.

    Attributes:
        temp: Target temperature used after warmup.
        cur_temp: Current temperature value.
        temp_warmup_steps: Number of steps for temperature warmup.
        initial_temp: Starting temperature value.
    """

    def __init__(
        self,
        temp: float = 0.07,
        temp_warmup_steps: int = 1000,
        initial_temp: float = 0.07,
    ) -> None:
        """Initializes the loss.

        Args:
            temp: Final temperature used after warmup.
            temp_warmup_steps: Number of steps over which to warm up the temperature.
            initial_temp: Temperature used at step zero.
        """
        super().__init__()
        self.temp = temp
        self.cur_temp = initial_temp
        self.temp_warmup_steps = temp_warmup_steps
        self.initial_temp = initial_temp

    def update_temp(self, step: int) -> None:
        """Updates the temperature schedule.

        Args:
            step: Current training step.
        """
        if step < self.temp_warmup_steps:
            self.cur_temp = (
                self.initial_temp
                + (self.temp - self.initial_temp) * step / self.temp_warmup_steps
            )
            logging.info("updating contrastive losss temp=%.3f", self.cur_temp)
        else:
            self.cur_temp = self.temp

    def forward(
        self,
        z_1: torch.Tensor,
        z_2: torch.Tensor,
        z_negatives_1: Optional[torch.Tensor] = None,
        z_negatives_2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes the symmetric cross-modal contrastive loss.

        Args:
            z_1: Mode 1 embeddings with shape ``(B, D)``.
            z_2: Mode 2 embeddings with shape ``(B, D)``.
            z_negatives_1: Optional additional mode 1 negatives with shape
                ``(K_1, D)``.
            z_negatives_2: Optional additional mode 2 negatives with shape
                ``(K_2, D)``.

        Returns:
            Scalar contrastive loss tensor.
        """
        B, D = z_1.shape

        # Normalize local features
        z_1 = nn.functional.normalize(z_1, dim=1)
        z_2 = nn.functional.normalize(z_2, dim=1)

        # Gather across distributed GPUs
        z_1_all = GatherDistributedFunction.apply(z_1)  # (G*B, D)
        z_2_all = GatherDistributedFunction.apply(z_2)  # (G*B, D)

        # Optionally append memory bank z_negatives
        if z_negatives_1 is not None:
            z_1_all = torch.cat(
                [z_1_all, nn.functional.normalize(z_negatives_1, dim=1)], dim=0
            )
        if z_negatives_2 is not None:
            z_2_all = torch.cat(
                [z_2_all, nn.functional.normalize(z_negatives_2, dim=1)], dim=0
            )

        # Similarity matrices
        logits_12 = torch.matmul(z_1, z_2_all.T) / self.cur_temp  # (B, G*B + K_txt)
        logits_21 = torch.matmul(z_2, z_1_all.T) / self.cur_temp  # (B, G*B + K_img)

        # Labels are position indices: correct match at local position + rank offset
        rank = ddp_get_rank()
        labels = torch.arange(B, device=z_1.device) + rank * B

        # Cross-entropy losses
        loss_12 = nn.functional.cross_entropy(logits_12, labels)
        loss_21 = nn.functional.cross_entropy(logits_21, labels)

        return (loss_12 + loss_21) / 2
