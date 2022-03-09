"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn


class LinBinCalibrator(nn.Module):
    """Linear score calibrator.
        Applies a scale and bias to a tensor.

    Attributes:
      a: Scale
      b: Bias
    """

    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        """Applies scale and bias to a tensor.

        Args:
          x: Input tensor.

        Returns:
          Calibrated tensor.
        """
        return self.a * x + self.b
