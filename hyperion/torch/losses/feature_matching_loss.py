"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for GAN training.

    Attributes:
        None: This module is stateless.
    """

    def __init__(self) -> None:
        """Initializes the stateless loss."""
        super().__init__()

    def forward(
        self,
        fmaps_generated: List[List[torch.Tensor]],
        fmaps_real: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """Computes the L1 distance between real and generated feature maps.

        Args:
            fmaps_generated: Discriminator feature maps for generated samples.
            fmaps_real: Discriminator feature maps for real samples.

        Returns:
            Scalar loss tensor.
        """
        loss = 0
        for fmap_r, fmap_g in zip(fmaps_real, fmaps_generated):
            for fmap_r_l, fmap_g_l in zip(fmap_r, fmap_g):
                fmap_r_l = fmap_r_l.float().detach()
                fmap_g_l = fmap_g_l.float()
                loss += torch.mean(torch.abs(fmap_g_l - fmap_r_l))

        return loss
