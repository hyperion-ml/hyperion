"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.nn import Linear

from ..layer_blocks import FCBlock
from ..layers import ActivationFactory as AF
from ..layers import NormLayer1dFactory as NLF
from .net_arch import NetArch


class FCNetV1(NetArch):
    """Stack of fully connected blocks with an optional output head.

    Attributes:
        num_blocks: Number of hidden fully connected blocks.
        out_units: Output feature dimension for the final linear layer, or
            ``0`` when the module is used as an encoder.
        in_units: Input feature dimension.
        hid_units: Hidden feature specification, either a scalar or a per-block
            list.
        hid_act: Hidden activation specification.
        dropout_rate: Dropout probability applied inside each block.
        norm_layer: Normalization layer specification.
        _norm_layer: Normalization layer constructor returned by
            :class:`NormLayer1dFactory`.
        use_norm: Whether normalization is enabled in the hidden blocks.
        norm_before: Whether normalization is applied before activation.
        in_norm: Whether to normalize the input features.
        in_bn: Optional input normalization layer.
        blocks: Hidden fully connected blocks.
        with_output: Whether the network has an output projection.
        out_act: Optional output activation module.
        output: Output linear projection layer.
    """

    def __init__(
        self,
        num_blocks: int,
        in_units: int,
        hid_units: Any,
        out_units: int = 0,
        hid_act: Any = {"name": "relu", "inplace": True},
        out_act: Any = None,
        dropout_rate: float = 0,
        norm_layer: Any = None,
        use_norm: bool = True,
        norm_before: bool = False,
        in_norm: bool = False,
    ) -> None:
        """Initialize the fully connected network.

        Args:
            num_blocks: Number of hidden fully connected blocks.
            in_units: Input feature dimension.
            hid_units: Hidden feature specification, either a scalar or a
                per-block list.
            out_units: Output feature dimension for the final linear layer.
                ``0`` keeps the module as an encoder.
            hid_act: Hidden activation specification.
            out_act: Output activation specification.
            dropout_rate: Dropout probability applied inside each block.
            norm_layer: Normalization layer specification.
            use_norm: Whether normalization is enabled in the hidden blocks.
            norm_before: Whether normalization is applied before activation.
            in_norm: Whether to normalize the input features.
        """

        super().__init__()

        self.num_blocks = num_blocks
        self.out_units = out_units
        self.in_units = in_units
        self.hid_units = hid_units
        self.hid_act = hid_act
        self.dropout_rate = dropout_rate

        self.norm_layer = norm_layer
        self.use_norm = use_norm
        self.norm_before = norm_before
        self.in_norm = in_norm

        self._norm_layer = NLF.create(norm_layer) if (use_norm or in_norm) else None

        if self.in_norm:
            self.in_bn = self._norm_layer(in_units)

        if isinstance(hid_units, list):
            assert num_blocks == len(hid_units)
        else:
            hid_units = [hid_units for i in range(num_blocks)]

        units = [in_units] + hid_units
        blocks = []
        for i in range(1, num_blocks + 1):
            blocks.append(
                FCBlock(
                    units[i - 1],
                    units[i],
                    activation=hid_act,
                    dropout_rate=dropout_rate,
                    norm_layer=self._norm_layer,
                    use_norm=use_norm,
                    norm_before=norm_before,
                )
            )

        self.blocks = nn.ModuleList(blocks)

        self.with_output = False
        self.out_act = None
        self.output = None
        if out_units == 0:
            return

        self.with_output = True
        self.out_act = AF.create(out_act)

        self.output = Linear(units[-1], out_units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the network to an input tensor.

        Args:
            x: Input tensor with the last dimension equal to ``in_units``.

        Returns:
            torch.Tensor: Network output.
        """

        if self.in_norm:
            x = self.in_bn(x)

        for i in range(self.num_blocks):
            x = self.blocks[i](x)

        if self.with_output:
            x = self.output(x)
            if self.out_act is not None:
                x = self.out_act(x)

        return x

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Return a serializable configuration dictionary.

        Args:
            no_class_name: If ``True``, omit the base ``class_name`` entry.

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
            "dropout_rate": self.dropout_rate,
            "norm_layer": self.norm_layer,
            "use_norm": self.use_norm,
            "norm_before": self.norm_before,
            "in_norm": self.in_norm,
            "out_act": out_act,
            "hid_act": hid_act,
        }

        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))


class FCNetV2(NetArch):
    """Stack of fully connected blocks with a dedicated output block.

    Attributes:
        num_blocks: Number of hidden fully connected blocks.
        out_units: Output feature dimension for the final block.
        in_units: Input feature dimension.
        hid_units: Hidden feature specification, either a scalar or a per-block
            list.
        hid_act: Hidden activation specification.
        dropout_rate: Dropout probability applied inside each hidden block.
        norm_layer: Normalization layer specification.
        nom_layer: Backward-compatible alias of :attr:`norm_layer`.
        _norm_layer: Normalization layer constructor returned by
            :class:`NormLayer1dFactory`.
        use_norm: Whether normalization is enabled in the blocks.
        norm_before: Whether normalization is applied before activation.
        in_norm: Whether to normalize the input features.
        in_bn: Optional input normalization layer.
        blocks: Fully connected blocks, including the output block when
            ``out_units`` is non-zero.
        with_output: Whether the network has a final output block.
        out_act: Optional output activation module.
    """

    def __init__(
        self,
        num_blocks: int,
        in_units: int,
        hid_units: Any,
        out_units: int = 0,
        hid_act: Any = {"name": "relu", "inplace": True},
        out_act: Any = None,
        dropout_rate: float = 0,
        norm_layer: Any = None,
        use_norm: bool = True,
        norm_before: bool = True,
        in_norm: bool = False,
    ) -> None:
        """Initialize the fully connected network.

        Args:
            num_blocks: Number of hidden fully connected blocks.
            in_units: Input feature dimension.
            hid_units: Hidden feature specification, either a scalar or a
                per-block list.
            out_units: Output feature dimension for the final block. ``0``
                keeps the module as an encoder.
            hid_act: Hidden activation specification.
            out_act: Output activation specification.
            dropout_rate: Dropout probability applied inside each hidden
                block.
            norm_layer: Normalization layer specification.
            use_norm: Whether normalization is enabled in the blocks.
            norm_before: Whether normalization is applied before activation.
            in_norm: Whether to normalize the input features.
        """

        super().__init__()

        self.num_blocks = num_blocks
        self.out_units = out_units
        self.in_units = in_units
        self.hid_units = hid_units
        self.hid_act = hid_act
        self.dropout_rate = dropout_rate

        self.norm_layer = norm_layer
        self.nom_layer = norm_layer
        self.use_norm = use_norm
        self.norm_before = norm_before
        self.in_norm = in_norm

        self._norm_layer = NLF.create(norm_layer) if (use_norm or in_norm) else None

        if self.in_norm:
            self.in_bn = self._norm_layer(in_units)

        if isinstance(hid_units, list):
            assert num_blocks == len(hid_units)
        else:
            hid_units = [hid_units for i in range(num_blocks)]

        units = [in_units] + hid_units
        blocks = []
        for i in range(1, num_blocks + 1):
            blocks.append(
                FCBlock(
                    units[i - 1],
                    units[i],
                    activation=hid_act,
                    dropout_rate=dropout_rate,
                    norm_layer=self._norm_layer,
                    use_norm=use_norm,
                    norm_before=norm_before,
                )
            )

        self.with_output = out_units != 0
        self.out_act = None
        if self.with_output:
            self.out_act = AF.create(out_act)
            blocks.append(
                FCBlock(
                    units[-1],
                    out_units,
                    activation=out_act,
                    norm_layer=self._norm_layer,
                    use_norm=use_norm,
                    norm_before=norm_before,
                )
            )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the network to an input tensor.

        Args:
            x: Input tensor with the last dimension equal to ``in_units``.

        Returns:
            torch.Tensor: Network output.
        """

        if self.in_norm:
            x = self.in_bn(x)

        for i in range(self.num_blocks):
            x = self.blocks[i](x)

        if self.with_output:
            x = self.blocks[self.num_blocks](x)

        return x

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Return a serializable configuration dictionary.

        Args:
            no_class_name: If ``True``, omit the base ``class_name`` entry.

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
            "dropout_rate": self.dropout_rate,
            "norm_layer": self.norm_layer,
            "use_norm": self.use_norm,
            "norm_before": self.norm_before,
            "in_norm": self.in_norm,
            "out_act": out_act,
            "hid_act": hid_act,
        }

        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))
