"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Optional, Sequence, Tuple

import torch.nn as nn

from ..hyper_torch_model import HyperTorchModel


class NetArch(HyperTorchModel):
    """Base class for network architecture modules.

    Subclasses are expected to describe their input/output tensor shapes.
    """

    def in_context(self) -> int:
        """Return left/right input context in frames.

        Returns:
            int: Required contextual frames. Defaults to 0.
        """
        return 0

    def in_dim(self) -> int:
        """Return the rank of the expected input shape.

        Returns:
            int: Number of dimensions in :meth:`in_shape`.
        """
        return len(self.in_shape())

    def out_dim(self) -> int:
        """Return the rank of the produced output shape.

        Returns:
            int: Number of dimensions in :meth:`out_shape`.
        """
        return len(self.out_shape())

    def in_shape(self) -> Tuple[int, ...]:
        """Return the expected input shape including the batch axis.

        Returns:
            Tuple[int, ...]: Input tensor shape specification.
        """
        raise NotImplementedError()

    def out_shape(self, in_shape: Optional[Sequence[int]] = None) -> Tuple[int, ...]:
        """Return the output shape including the batch axis.

        Args:
            in_shape: Optional input shape override used by dynamic architectures.

        Returns:
            Tuple[int, ...]: Output tensor shape specification.
        """
        raise NotImplementedError()
