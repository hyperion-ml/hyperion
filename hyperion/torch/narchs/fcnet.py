"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch.nn as nn
from torch.nn import Linear, BatchNorm1d, Dropout

from ..layers import ActivationFactory as AF
from ..layers import NormLayer1dFactory as NLF
from .net_arch import NetArch
from ..layer_blocks import FCBlock


class FCNetV1(NetArch):
    def __init__(
        self,
        num_blocks,
        in_units,
        hid_units,
        out_units=0,
        hid_act={"name": "relu", "inplace": True},
        out_act=None,
        dropout_rate=0,
        norm_layer=None,
        use_norm=True,
        norm_before=False,
        in_norm=False,
    ):

        super().__init__()

        self.num_blocks = num_blocks
        self.out_units = out_units
        self.in_units = in_units
        self.hid_units = hid_units
        self.hid_act = hid_act
        self.dropout_rate = dropout_rate

        if use_norm:
            self._norm_layer = NLF.create(norm_layer)
        else:
            self._norm_layer = None

        self.use_norm = use_norm
        self.norm_before = norm_before
        self.in_norm = in_norm

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
        if out_units == 0:
            return

        self.with_output = True
        self.out_act = AF.create(out_act)

        self.output = Linear(units[-1], out_units)

    def forward(self, x):

        if self.in_norm:
            x = self.in_bn(x)

        for i in range(self.num_blocks):
            x = self.blocks[i](x)

        if self.with_output:
            x = self.output(x)
            if self.out_act is not None:
                x = self.out_act(x)

        return x

    def get_config(self):

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

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FCNetV2(NetArch):
    def __init__(
        self,
        num_blocks,
        in_units,
        hid_units,
        out_units=0,
        hid_act={"name": "relu6", "inplace": True},
        out_act=None,
        dropout_rate=0,
        norm_layer=None,
        use_norm=True,
        norm_before=True,
        in_norm=False,
    ):

        super().__init__()

        self.num_blocks = num_blocks
        self.out_units = out_units
        self.in_units = in_units
        self.hid_units = hid_units
        self.hid_act = hid_act
        self.dropout_rate = dropout_rate

        if use_norm:
            self._norm_layer = NLF.create(norm_layer)
        else:
            self._norm_layer = None

        self.nom_layer = norm_layer
        self.use_norm = use_norm
        self.norm_before = norm_before
        self.in_norm = in_norm

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

    def forward(self, x):

        if self.in_norm:
            x = self.in_bn(x)

        for i in range(self.num_blocks + 1):
            x = self.blocks[i](x)

        return x

    def get_config(self):

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

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
