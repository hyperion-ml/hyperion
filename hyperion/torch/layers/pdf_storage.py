"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
#

import torch
import torch.nn as nn
import torch.distributions as pdf


class StdNormal(nn.Module):
    """Storage for Standard Normal distribution parameters
    
    Attributes:
      shape: shape of the location/scale tensors.
    """

    def __init__(self, shape):
        super().__init__()
        self.register_buffer("loc", torch.zeros(shape))
        self.register_buffer("scale", torch.ones(shape))

    @property
    def pdf(self):
        """Probability density function for N(0,I)."""
        return pdf.normal.Normal(self.loc, self.scale)

    def forward(self):
        """Probability density function for N(0,I)."""
        return self.pdf
