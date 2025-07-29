"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class AudioDiscriminatorAdvLoss(nn.Module):
    """
    Adversarial audio discriminator loss for GANs.

    This loss computes the mean squared error between the discriminator's outputs for real and generated samples.
    It is used to train the discriminator to distinguish between real and generated audio samples.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, discrim_generated: List[torch.Tensor], discrim_real: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[float], List[float]]:
        """
        Compute the adversarial discriminator loss.

        Args:
            discrim_generated (list): List of discriminator outputs for generated samples.
            discrim_real (list): List of discriminator outputs for real samples.

        Returns:
            torch.Tensor: Computed adversarial discriminator loss.
        """
        loss = 0
        real_losses = []
        gen_losses = []
        for d_r, d_g in zip(discrim_real, discrim_generated):
            d_r = d_r.float()
            d_g = d_g.float()
            r_loss = torch.mean((1 - d_r) ** 2)
            g_loss = torch.mean(d_g**2)
            loss += r_loss + g_loss
            real_losses.append(r_loss.item())
            gen_losses.append(g_loss.item())

        return loss, gen_losses, real_losses


class AudioGeneratorAdvLoss(nn.Module):
    """
    Adversarial audio generator loss for GANs.

    This loss computes the mean squared error between the discriminator's outputs for generated samples.
    It is used to train the generator to produce samples that are indistinguishable from real audio samples.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, discrim_gen: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Compute the adversarial generator loss.

        Args:
            discrim_real (list): List of discriminator outputs for generated samples.

        Returns:
            torch.Tensor: Computed adversarial generator loss.
        """
        loss = 0
        gen_losses = []
        for d_g in discrim_gen:
            d_g = d_g.float()
            g_loss = torch.mean((1 - d_g) ** 2)
            gen_losses.append(g_loss.item())
            loss += g_loss

        return loss, gen_losses
