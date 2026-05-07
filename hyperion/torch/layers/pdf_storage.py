"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
#

from __future__ import annotations

from typing import Sequence

import torch
import torch.distributions as dist
import torch.nn as nn


class StdNormal(nn.Module):
    """Storage for a standard normal distribution ``N(0, I)``.

    Args:
        shape: Shape used to initialize the location and scale buffers.

    Attributes:
        shape: Shape of the location and scale tensors.
        loc: Location buffer of zeros.
        scale: Scale buffer of ones.
    """

    def __init__(self, shape: int | Sequence[int] | torch.Size) -> None:
        super().__init__()
        self.shape = (
            torch.Size((shape,)) if isinstance(shape, int) else torch.Size(shape)
        )
        self.register_buffer("loc", torch.zeros(self.shape))
        self.register_buffer("scale", torch.ones(self.shape))

    @property
    def pdf(self) -> dist.Normal:
        """Builds the normal distribution represented by this module.

        Returns:
            A ``torch.distributions.Normal`` with ``loc`` and ``scale`` buffers.
        """
        return dist.normal.Normal(self.loc, self.scale)

    def forward(self) -> dist.Normal:
        """Returns the stored normal distribution.

        Returns:
            A ``torch.distributions.Normal`` with ``loc`` and ``scale`` buffers.
        """
        return self.pdf
