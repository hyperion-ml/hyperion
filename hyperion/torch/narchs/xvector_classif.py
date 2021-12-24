"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np

import torch.nn as nn
from torch.nn import Linear, BatchNorm1d, Dropout

from ..layers import ActivationFactory as AF
from .net_arch import NetArch


class XVectorClassifV1(NetArch):
    def __init__(
        self,
        input_units,
        num_classes,
        embed_dim=512,
        num_hid_layers=2,
        hid_act="relu",
        outputs="logits",
        use_batchnorm=True,
        dropout_rate=0,
    ):

        super(XVectorClassifV1, self).__init__()
        assert num_hid_layers >= 1, "num_hid_layers (%d < 1)" % num_hid_layers

        self.num_hid_layers = num_hid_layers
        self.input_units = input_units
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        self.outputs = outputs

        if isinstance(hid_units, list):
            assert num_hid_layers == len(embed_dim)
        else:
            embed_dim = [embed_dim for i in range(num_hid_layers)]

        units = [input_units] + embed_dim

        # fully connected layers
        fc_layers = []
        for i in range(1, num_hid_layers + 1):
            fc_layers.append(Linear(units[i - 1], units[i]))

        self.fc_layers = nn.ModuleList(fc_layers)

        # hidden activations
        self.hid_acts = None
        if hid_act is not None:
            hid_acts = []
            for i in range(num_hid_layers):
                hid_act = AF.create(hid_act)
                hid_acts.append(hid_act)
            self.hid_acts = nn.ModuleList(hid_acts)

        # batch normalization
        self.batchnorm_layers = None
        if use_batchnorm:
            batchnorm_layers = []
            for i in range(num_hid_layers):
                batchnorm_layers.append(BatchNorm1d(units[i]))
            self.batchnorm_layers = nn.ModuleList(batchnorm_layers)

        # dropout
        self.dropout_layers = None
        if dropout_rate > 0:
            dropout_layers = []
            for i in range(num_hid_layers):
                dropout_layers.append(Dropout(dropout_rate))
            self.dropout_layers = nn.ModuleList(dropout_layers)

        # output layers
        self.logits_layer = Linear(units[-1], num_classes)

    def forward(self, x):

        for l in range(self.num_hid_layers):
            if self.use_batchnorm:
                x = self.batchnorm_layers[l](x)

            x = self.fc_layers[l](x)
            if self.hid_acts is not None:
                x = self.hid_acts[l](x)

            if self.dropout_rate > 0:
                x = self.dropout_layers[l](x)

        y = self.logits_layer(x)

        return y

    def extract_embed(self, x, embed_layers=0):

        if isinstance(embed_layers, int):
            embed_layers = [embed_layers]

        last_embed_layer = np.max(embed_layers)
        embed_layers = set(embed_layers)

        embed_list = []
        for l in range(self.num_hid_layers):
            if self.use_batchnorm:
                x = self.batchnorm_layers[l](x)

            x = self.fc_layers[l](x)
            if l in embed_layers:
                embed_list.append(x)

            if l == last_embed_layer:
                break

            if self.hid_acts is not None:
                x = self.hid_acts[l](x)

            if self.dropout_rate > 0:
                x = self.dropout_layers[l](x)

        y = torch.cat((embed_list), dim=-1)
        return y

    def get_config(self):

        if self.hid_acts is None:
            hid_act = None
        else:
            hid_act = AF.get_config(self.hid_acts[0])

        config = {
            "num_hid_layers": self.num_hid_layers,
            "num_classes": self.num_classes,
            "embed_dim": self.embed_dim,
            "input_units": self.input_units,
            "use_batchnorm": self.use_batchnorm,
            "dropout_rate": self.dropout_rate,
            "hid_act": hid_act,
        }

        base_config = super(XVectorClassifV1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
