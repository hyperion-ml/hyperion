"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss for GANs.

    This loss computes the mean absolute error between the features of real and generated samples.
    It is used to stabilize GAN training by encouraging the generator to produce samples that
    match the statistics of real data.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        fmaps_generated: List[List[torch.Tensor]],
        fmaps_real: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute the feature matching loss.

        Args:
            fmaps_generated (list): List of feature maps from the discriminator for generated samples.
            fmaps_real (list): List of feature maps from the discriminator for real samples.

        Returns:
            torch.Tensor: Computed feature matching loss.
        """
        loss = 0
        for fmap_r, fmap_g in zip(fmaps_real, fmaps_generated):
            for fmap_r_l, fmap_g_l in zip(fmap_r, fmap_g):
                fmap_r_l = fmap_r_l.float().detach()
                fmap_g_l = fmap_g_l.float()
                loss += torch.mean(torch.abs(fmap_g_l - fmap_r_l))

        return loss
