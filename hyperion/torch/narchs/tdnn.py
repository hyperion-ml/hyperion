"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

import torch
import torch.nn as nn
from torch.nn import Linear

from ..layer_blocks import TDNNBlock
from ..layers import ActivationFactory as AF
from ..layers import NormLayer1dFactory as NLF
from .net_arch import NetArch


class TDNNV1(NetArch):
    """TDNN encoder with optional output projection.

    Attributes:
        num_blocks: Number of TDNN blocks.
        out_units: Output dimension for the final linear layer, or ``0`` when
            the network is used as a pure encoder.
        in_units: Input feature dimension.
        hid_units: Hidden layer width specification.
        kernel_size: Convolution kernel size specification.
        dilation: Convolution dilation specification.
        dilation_factor: Increment used when ``dilation`` is not a sequence.
        dropout_rate: Dropout probability applied inside each block.
        use_norm: Whether normalization layers are enabled.
        norm_before: Whether normalization is applied before activation.
        in_norm: Whether input normalization is enabled.
        pooling: Optional pooling mode applied before the output projection.
    """

    def __init__(
        self,
        num_blocks: int,
        in_units: int,
        hid_units: Any,
        out_units: int = 0,
        kernel_size: Any = 3,
        dilation: Any = 1,
        dilation_factor: int = 1,
        hid_act: Any = {"name": "relu", "inplace": True},
        out_act: Any = None,
        dropout_rate: float = 0,
        norm_layer: Optional[str] = None,
        use_norm: bool = True,
        norm_before: bool = True,
        in_norm: bool = True,
        pooling: Optional[str] = None,
    ) -> None:
        """Initializes the TDNN encoder.

        Args:
            num_blocks: Number of TDNN blocks.
            in_units: Input feature dimension.
            hid_units: Hidden layer width or per-block width sequence.
            out_units: Output dimension for the final linear layer. ``0`` keeps
                the module as an encoder.
            kernel_size: Kernel size or per-block kernel sizes.
            dilation: Dilation or per-block dilations.
            dilation_factor: Dilation increment used when ``dilation`` is a
                scalar.
            hid_act: Hidden activation specification.
            out_act: Output activation specification.
            dropout_rate: Dropout probability used in the TDNN blocks.
            norm_layer: Normalization layer name.
            use_norm: Whether to enable normalization layers.
            norm_before: Whether normalization happens before activation.
            in_norm: Whether input normalization is enabled.
            pooling: Optional pooling mode before the output projection.
        """

        super().__init__()

        self.num_blocks = num_blocks
        self.out_units = out_units
        self.in_units = in_units
        self.hid_units = hid_units
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dilation_factor = dilation_factor
        self.dropout_rate = dropout_rate
        self.use_norm = use_norm
        self.norm_before = norm_before
        self.in_norm = in_norm
        self.pooling = pooling

        if isinstance(hid_units, list):
            assert num_blocks == len(hid_units)
        else:
            hid_units = [hid_units for i in range(num_blocks)]

        units = [in_units] + hid_units

        if isinstance(kernel_size, list):
            assert num_blocks == len(kernel_size)
        else:
            kernel_size = [kernel_size for i in range(num_blocks)]

        if isinstance(dilation, list):
            assert num_blocks == len(dilation)
        else:
            dilation = [dilation_factor * i + dilation for i in range(num_blocks)]

        # past and future context
        self._context = int(
            np.sum(np.array(dilation) * (np.array(kernel_size) - 1) / 2)
        )

        self.norm_layer = norm_layer
        norm_groups = None
        if norm_layer == "group-norm":
            norm_groups = min(np.min(hid_units) // 2, 32)
        self._norm_layer = NLF.create(norm_layer, norm_groups)

        self.in_bn = None
        if self.in_norm:
            self.in_bn = self._norm_layer(in_units)

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                TDNNBlock(
                    units[i],
                    units[i + 1],
                    kernel_size=kernel_size[i],
                    dilation=dilation[i],
                    activation=hid_act,
                    dropout_rate=dropout_rate,
                    norm_layer=self._norm_layer,
                    use_norm=use_norm,
                    norm_before=norm_before,
                )
            )

        self.blocks = nn.ModuleList(blocks)

        self.with_output = False
        if out_units == 0:
            self.out_act = None
            self.output = None
            return

        self.with_output = True
        self.out_act = AF.create(out_act)

        self.output = Linear(units[-1], out_units)

    @property
    def in_context(self) -> Tuple[int, int]:
        """Return the left and right temporal context.

        Returns:
            Tuple[int, int]: Required input context in frames.
        """
        return (self._context, self._context)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the network to an input tensor.

        Args:
            x: Input tensor with shape ``(batch, channels, frames)``.

        Returns:
            torch.Tensor: Encoded tensor. If ``out_units`` is non-zero, the
            output is the projected representation.
        """

        if self.in_norm:
            x = self.in_bn(x)

        for i in range(self.num_blocks):
            x = self.blocks[i](x)

        if self.with_output:
            if self.pooling is not None:
                if self.pooling == "mean":
                    x = torch.mean(x, dim=2)
                elif self.pooling == "max":
                    x = torch.max(x, dim=2)[0]
                else:
                    raise Exception("pooling=%s not implemented" % (self.pooling))
            else:
                x = torch.transpose(x, 1, 2)

            x = self.output(x)
            if self.out_act is not None:
                x = self.out_act(x)

        return x

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Return a serializable configuration dictionary.

        Args:
            no_class_name: If ``True``, omit the class name from the base
                configuration.

        Returns:
            Dict[str, Any]: Configuration dictionary for reconstruction.
        """

        out_act = AF.get_config(self.out_act)
        hid_act = AF.get_config(self.blocks[0].activation)

        config = {
            "num_blocks": self.num_blocks,
            "in_units": self.in_units,
            "hid_units": self.hid_units,
            "out_units": self.out_units,
            "kernel_size": self.kernel_size,
            "dilation": self.dilation,
            "dilation_factor": self.dilation_factor,
            "dropout_rate": self.dropout_rate,
            "norm_layer": self.norm_layer,
            "use_norm": self.use_norm,
            "norm_before": self.norm_before,
            "in_norm": self.in_norm,
            "out_act": out_act,
            "hid_act": hid_act,
            "pooling": self.pooling,
        }

        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))

    def in_shape(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Return the expected input shape.

        Returns:
            Tuple[Optional[int], Optional[int], Optional[int]]: Shape
            specification ``(batch, channels, frames)``.
        """
        return (None, self.in_units, None)

    def out_shape(
        self, in_shape: Optional[Sequence[int]] = None
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape for a given input shape.

        Args:
            in_shape: Optional input shape override.

        Returns:
            Tuple[Optional[int], ...]: Output shape specification.
        """
        if self.with_output:
            if self.pooling is None:
                if in_shape is None:
                    return (None, None, self.out_units)
                assert len(in_shape) == 3
                return (in_shape[0], in_shape[2], self.out_units)

            return (None, self.out_units)

        if isinstance(self.hid_units, list):
            out_units = self.hid_units[-1]
        else:
            out_units = self.hid_units

        if in_shape is None:
            return (None, out_units, None)

        assert len(in_shape) == 3
        return (in_shape[0], out_units, in_shape[2])
