"""
Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch import Tensor

ScaleFactor = int | float | tuple[int, ...] | tuple[float, ...]


class Interpolate(nn.Module):
    """Thin wrapper over ``torch.nn.functional.interpolate``.

    Args:
      scale_factor: Upsampling scale factor.
      mode: Algorithm used for upsampling.

    Attributes:
      scale_factor: Upsampling scale factor.
      mode: Algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'.

    Returns:
      Interpolate module instance.
    """

    def __init__(self, scale_factor: ScaleFactor, mode: str = "nearest") -> None:
        """Initializes interpolation parameters.

        Args:
          scale_factor: Upsampling scale factor.
          mode: Interpolation algorithm.

        Returns:
          None.
        """
        super().__init__()
        self.interp = nnf.interpolate
        if isinstance(scale_factor, (int, float)):
            self.scale_factor = float(scale_factor)
        else:
            self.scale_factor = tuple(float(s) for s in scale_factor)

        self.mode = mode

    def __repr__(self) -> str:
        s = "{}(scale_factor={}, mode={})".format(
            self.__class__.__name__,
            self.scale_factor,
            self.mode,
        )
        return s

    def forward(self, x: Tensor) -> Tensor:
        """Interpolates the input.

        Args:
          x: Input tensor.

        Returns:
          Interpolated tensor.
        """
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
