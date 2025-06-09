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
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent) used in SimCLR.

    This contrastive loss pulls together representations of different augmented
    views of the same sample and pushes apart those of different samples.

    Args:
        temp (float): Target temperature scaling value for similarity logits.
                      This is the final value used after warmup.
        temp_warmup_steps (int): Number of steps over which to linearly warm up the temperature
                                 from `initial_temp` to `temp`.
        initial_temp (float): Initial temperature used at the start of training before warmup completes.
        num_views (int): Number of augmented views per sample. Default is 2.
        grouped_views (bool): If True, assumes views are grouped per sample (e.g., [x1_view1, x1_view2, ..., xN_view1, xN_view2]).
                              If False, assumes tiled layout (e.g., [x1_view1, x2_view1, ..., xN_view1, x1_view2, ...]).
    """

    def __init__(
        self,
        temp: float,
        temp_warmup_steps: int = 1000,
        initial_temp: float = 0.07,
        num_views: int = 2,
        grouped_views: bool = False,
    ):
        super().__init__()
        self.temp = temp
        self.cur_temp = initial_temp
        self.temp_warmup_steps = temp_warmup_steps
        self.initial_temp = initial_temp
        self.num_views = num_views
        self.grouped_views = grouped_views

    def update_temp(self, step: int):
        if step < self.temp_warmup_steps:
            self.cur_temp = (
                self.initial_temp
                + (self.temp - self.initial_temp) * step / self.temp_warmup_steps
            )
            logging.info("updating contrastive losss temp=%.3f", self.cur_temp)
        else:
            self.cur_temp = self.temp

    def forward(self, z: torch.Tensor, z_negatives: Optional[torch.Tensor] = None):
        """
        Compute the SimCLR NT-Xent contrastive loss.

        Args:
            z (Tensor): Normalized representations of augmented views.
                        Shape: (num_views * batch_size, projection_dim)
            z_negatives_1 (Optional[Tensor]): Additional mode 1 z_negatives (K_img, D)

        Returns:
            Tensor: Scalar loss value.
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
        sim = similarity / self.temp

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
                        i * batch_size + batch_size, device=z.device
                    ).repeat(self.num_views)
                    start = stop
            else:
                labels = torch.arange(batch_size, device=z.device).repeat(
                    self.num_views
                )

        mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_pos.fill_diagonal_(False)  # remove (i, i)

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
        mean_log_prob_pos = (log_prob * mask_pos).sum(1) / mask_pos.sum(1)

        # Final loss
        loss = -mean_log_prob_pos.mean()
        return loss
