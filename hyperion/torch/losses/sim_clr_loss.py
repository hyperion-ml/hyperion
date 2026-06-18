"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from ..layers import GatherDistributedFunction


class SimCLRLoss(nn.Module):
    """Normalized temperature-scaled cross entropy loss for SimCLR.

    Attributes:
        temp: Target temperature used after warmup.
        cur_temp: Current temperature value.
        temp_warmup_steps: Number of steps for temperature warmup.
        initial_temp: Starting temperature value.
        num_views: Number of views per sample.
        grouped_views: Whether the input layout is grouped by sample.
    """

    def __init__(
        self,
        temp: float,
        temp_warmup_steps: int = 1000,
        initial_temp: float = 0.07,
        num_views: int = 2,
        grouped_views: bool = False,
    ) -> None:
        """Initializes the loss.

        Args:
            temp: Final temperature used after warmup.
            temp_warmup_steps: Number of steps over which to warm up the temperature.
            initial_temp: Temperature used at step zero.
            num_views: Number of augmented views per sample.
            grouped_views: If ``True``, input views are grouped per sample.
        """
        super().__init__()
        self.temp = temp
        self.cur_temp = initial_temp
        self.temp_warmup_steps = temp_warmup_steps
        self.initial_temp = initial_temp
        self.num_views = num_views
        self.grouped_views = grouped_views

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
        self, z: torch.Tensor, z_negatives: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes the SimCLR NT-Xent loss.

        Args:
            z: Representations of augmented views with shape
                ``(num_views * batch_size, projection_dim)``.
            z_negatives: Optional extra negatives with shape ``(K, projection_dim)``.

        Returns:
            Scalar loss tensor.
        """
        batch_size = z.shape[0] // self.num_views
        z = GatherDistributedFunction.apply(z)  # Gather across distributed ranks
        global_batch_size = z.shape[0] // self.num_views
        z = nn.functional.normalize(z, dim=1)
        num_gpus = global_batch_size // batch_size

        # Optionally normalize and append z_negatives
        if z_negatives is not None:
            z_negatives = nn.functional.normalize(z_negatives, dim=1)
            z_extended = torch.cat([z, z_negatives], dim=0)  # (N * B + K, D)
        else:
            z_extended = z

        similarity = torch.matmul(z, z_extended.T)  # shape: (NxB, NxB+K)
        sim = similarity / self.cur_temp

        # Remove self-similarity
        mask = torch.eye(sim.shape[0], dtype=torch.bool, device=z.device)
        if z_negatives is None:
            sim.masked_fill_(mask, -1e9)
        else:
            sim[:, : sim.shape[0]].masked_fill_(mask, -1e9)  # Keep z_negatives intact

        if self.grouped_views:
            # e.g., [x1_v1, x1_v2, x1_v3, x2_v1, x2_v2, x2_v3]
            labels = torch.arange(global_batch_size, device=z.device).repeat_interleave(
                self.num_views
            )
        else:
            ## e.g., [x1_v1, x2_v1, x3_v1, x1_v2, x2_v2, x3_v2]
            if num_gpus > 1:
                labels = torch.zeros(
                    (global_batch_size * self.num_views,),
                    device=z.device,
                    dtype=torch.long,
                )
                start = 0
                for i in range(global_batch_size // batch_size):
                    stop = start + batch_size * self.num_views
                    labels[start:stop] = torch.arange(
                        i * batch_size, i * batch_size + batch_size, device=z.device
                    ).repeat(self.num_views)
                    start = stop
            else:
                labels = torch.arange(batch_size, device=z.device).repeat(
                    self.num_views
                )

        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask.fill_diagonal_(False)  # remove (i, i)

        # Pad mask to match sim shape
        if z_negatives is not None:
            pad = torch.zeros(
                pos_mask.shape[0],
                z_negatives.shape[0],
                dtype=torch.bool,
                device=z.device,
            )
            pos_mask = torch.cat([pos_mask, pad], dim=1)  # (N*B, N*B+K)

        # For each i, compute log-softmax over all except i
        log_prob = nn.functional.log_softmax(sim, dim=1)

        # Only keep positives
        mean_log_prob_pos = (log_prob * pos_mask).sum(1) / pos_mask.sum(1)

        # Final loss
        loss = -mean_log_prob_pos.mean()
        return loss
