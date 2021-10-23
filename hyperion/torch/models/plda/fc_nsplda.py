"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn

from ...layers import ActivationFactory as AF
from ...narchs import FCNetV2
from .splda import SPLDA


class FCNSPLDA(SPLDA):
    def __init__(
        self,
        in_feats,
        num_layers,
        hid_feats,
        x_dim,
        y_dim,
        mu=None,
        V=None,
        W=None,
        num_classes=0,
        x_ref=None,
        p_tar=0.05,
        margin_multi=0.3,
        margin_tar=0.3,
        margin_non=0.3,
        margin_warmup_epochs=10,
        adapt_margin=False,
        adapt_gamma=0.99,
        lnorm=False,
        hid_act={"name": "relu6", "inplace": True},
        dropout_rate=0,
        norm_layer="batch-norm",
        use_norm=True,
        var_floor=1e-5,
        prec_floor=1e-5,
    ):
        # dd = locals()
        # del dd["self"]
        # print(dd, flush=True)
        # in_feats = 139
        # num_layers = 3
        # hid_feats = 128
        # hid_act = "relu6"
        # dropout_rate = 0
        preprocessor = FCNetV2(
            num_layers,
            in_feats,
            hid_feats,
            x_dim,
            hid_act=hid_act,
            dropout_rate=dropout_rate,
            norm_layer=norm_layer,
            use_norm=use_norm,
            norm_before=True,
        )

        super().__init__(
            x_dim=x_dim,
            y_dim=y_dim,
            mu=mu,
            V=V,
            W=W,
            num_classes=num_classes,
            x_ref=x_ref,
            p_tar=p_tar,
            margin_multi=margin_multi,
            margin_tar=margin_tar,
            margin_non=margin_non,
            margin_warmup_epochs=margin_warmup_epochs,
            adapt_margin=adapt_margin,
            adapt_gamma=adapt_gamma,
            lnorm=lnorm,
            var_floor=var_floor,
            prec_floor=prec_floor,
            preprocessor=preprocessor,
        )

    @property
    def in_feats(self):
        return self.preprocessor.in_units

    @property
    def num_layers(self):
        return self.preprocessor.num_blocks

    @property
    def hid_feats(self):
        return self.preprocessor.hid_units

    @property
    def hid_act(self):
        hid_act = AF.get_config(self.preprocessor.blocks[0].activation)
        return hid_act

    @property
    def dropout_rate(self):
        return self.preprocessor.dropout_rate

    @property
    def norm_layer(self):
        return self.preprocessor.nom_layer

    @property
    def use_norm(self):
        return self.preprocessor.use_norm

    def get_config(self):
        config = {
            "in_feats": self.in_feats,
            "num_layers": self.num_layers,
            "hid_feats": self.hid_feats,
            "hid_act": self.hid_act,
            "dropout_rate": self.dropout_rate,
            "norm_layer": self.norm_layer,
            "use_norm": self.use_norm,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
