"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np

import torch
import torch.nn as nn
from torch.nn import Conv1d, Linear

from ..layers import ActivationFactory as AF
from ..layers import NormLayer1dFactory as NLF
from ..layer_blocks import ETDNNBlock
from .net_arch import NetArch


class ETDNNV1(NetArch):
    def __init__(
        self,
        num_blocks,
        in_units,
        hid_units,
        out_units=0,
        kernel_size=3,
        dilation=1,
        dilation_factor=1,
        hid_act={"name": "relu", "inplace": True},
        out_act=None,
        dropout_rate=0,
        norm_layer=None,
        use_norm=True,
        norm_before=True,
        in_norm=True,
        pooling=None,
    ):

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

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                ETDNNBlock(
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
    def in_context(self):
        return (self._context, self._context)

    def forward(self, x):

        for i in range(self.num_blocks):
            x = self.blocks[i](x)

        if self.with_output:
            if self.pooling is not None:
                if self.pooling == "mean":
                    x = torch.mean(x, dim=2)
                elif self.pooling == "max":
                    x = torch.max(x, dim=2)
                else:
                    raise Exception("pooling=%s not implemented" % (self.pooling))
            else:
                x = torch.transpose(x, 1, 2)

            x = self.output(x)
            if self.out_act is not None:
                x = self.out_act(x)

        return x

    def get_config(self):

        out_act = AF.get_config(self.out_act)
        hid_act = AF.get_config(self.blocks[0].activation1)

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

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def in_shape(self):
        return (None, self.in_units, None)

    def out_shape(self, in_shape=None):
        if self.with_output:
            return (None, self.out_units)

        if isinstance(self.hid_units, list):
            out_units = self.hid_units[-1]
        else:
            out_units = self.hid_units

        if in_shape is None:
            return (None, out_units, None)

        assert len(in_shape) == 3
        return (in_shape[0], out_units, in_shape[2])
