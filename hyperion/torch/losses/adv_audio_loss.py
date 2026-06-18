"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class AudioDiscriminatorAdvLoss(nn.Module):
    """Least-squares adversarial loss for an audio discriminator.

    Attributes:
        None: This module is stateless.
    """

    def __init__(self) -> None:
        """Initializes the stateless loss."""
        super().__init__()

    def forward(
        self, discrim_generated: List[torch.Tensor], discrim_real: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[float], List[float]]:
        """Computes the discriminator loss for real and generated samples.

        Args:
            discrim_generated: Discriminator outputs for generated samples.
            discrim_real: Discriminator outputs for real samples.

        Returns:
            Tuple containing the scalar loss tensor, per-layer generated losses,
            and per-layer real losses.
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
    """Least-squares adversarial loss for an audio generator.

    Attributes:
        None: This module is stateless.
    """

    def __init__(self) -> None:
        """Initializes the stateless loss."""
        super().__init__()

    def forward(
        self, discrim_gen: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[float]]:
        """Computes the generator loss for discriminator outputs.

        Args:
            discrim_gen: Discriminator outputs for generated samples.

        Returns:
            Tuple containing the scalar loss tensor and per-layer generator
            losses.
        """
        loss = 0
        gen_losses = []
        for d_g in discrim_gen:
            d_g = d_g.float()
            g_loss = torch.mean((1 - d_g) ** 2)
            gen_losses.append(g_loss.item())
            loss += g_loss

        return loss, gen_losses
