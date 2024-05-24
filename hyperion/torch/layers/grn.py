"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from typing import Optional
import torch
import torch.nn as nn

class GRN2d(nn.Module):
    """ GRN (Global Response Normalization) layer for 2d feature maps

    Args:
      num_channels: number of input/output channels
      channels_last: it True, channels are in the last dimension, otherwise in dim 1
    """
    def __init__(self, num_channels, channels_last=False):
        super().__init__()
        if channels_last:
            self.gamma = nn.Parameter(torch.zeros(1, 1, 1, num_channels))
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, num_channels))
            self.norm_dims = (1,2)
            self.mean_dims = -1
        else:
            self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
            self.norm_dims = (2,3)
            self.mean_dims = 1

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor]=None):
        if x_mask is not None:
            x = x * x_mask

        Gx = torch.norm(x, p=2, dim=self.norm_dims, keepdim=True)
        Nx = Gx / (Gx.mean(dim=self.mean_dims, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
    
class GRN1d(nn.Module):
    """ GRN (Global Response Normalization) layer for 1d feature maps

    Args:
      num_channels: number of input/output channels
      channels_last: it True, channels are in the last dimension, otherwise in dim 1
    """
    def __init__(self, num_channels, channels_last=False):
        super().__init__()
        if channels_last:
            self.gamma = nn.Parameter(torch.zeros(1, 1, num_channels))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_channels))
            self.norm_dims = 1
            self.mean_dims = -1
        else:
            self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))
            self.norm_dims = -1
            self.mean_dims = 1

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor]=None):
        if x_mask is not None:
            x = x * x_mask

        Gx = torch.norm(x, p=2, dim=self.norm_dims, keepdim=True)
        Nx = Gx / (Gx.mean(dim=self.mean_dims, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x