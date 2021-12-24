"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging

import torch
import torch.nn as nn

from ...torch_model import TorchModel
from ...narchs import TorchNALoader


class AE(TorchModel):
    """Basic Autoencoder class

    Attributes:
      encoder_net: NArch encoder network object
      decoder_net: NArch decoder network object
      z_dim: latent variable dimension (inferred from encoder_net output shape)

    """

    def __init__(self, encoder_net, decoder_net):
        super().__init__()
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net

        # infer input feat dimension from encoder network
        in_shape = encoder_net.in_shape()
        # number of dimension of input/output enc/dec tensors,
        # needed to connect the blocks
        self._enc_in_dim = len(in_shape)
        self._enc_out_dim = self.encoder_net.out_dim()
        self._dec_in_dim = self.decoder_net.in_dim()
        self._dec_out_dim = self.decoder_net.out_dim()

        # we assume conv nnets with channel in dimension 1
        self.in_channels = in_shape[1]
        self._enc_out_channels = self.encoder_net.out_shape()[1]
        self._dec_out_channels = self.decoder_net.out_shape()[1]

        self.z_dim = self._enc_out_channels

    def _pre_enc(self, x):
        if x.dim() == 3 and self._enc_in_dim == 4:
            return x.unsqueeze(1)

        return x

    def _post_enc(self, x):
        return x

    def _pre_dec(self, x):
        if self._enc_out_dim == 3 and self._dec_in_dim == 4:
            return x.unsqueeze(dim=1)

        if self._enc_out_dim == 4 and self._dec_in_dim == 3:
            return x.view(x.size(0), -1, x.size(-1))

        return x

    def forward(self, x, x_target=None, use_amp=False):
        if use_amp:
            with torch.cuda.amp.autocast():
                return self._forward(x, x_target)

        return self._forward(x, x_target)

    def _forward(self, x, x_target=None):
        if x_target is None:
            x_target = x

        xx = self._pre_enc(x)
        z = self.encoder_net(xx)

        zz = self._pre_dec(z)
        xhat = self.decoder_net(zz, target_shape=x_target.shape)
        return xhat

    def get_config(self):
        enc_cfg = self.encoder_net.get_config()
        dec_cfg = self.decoder_net.get_config()
        config = {"encoder_cfg": enc_cfg, "decoder_cfg": dec_cfg}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def load(cls, file_path=None, cfg=None, state_dict=None):
        cfg, state_dict = cls._load_cfg_state_dict(file_path, cfg, state_dict)

        encoder_net = TorchNALoader.load(cfg=cfg["encoder_cfg"])
        decoder_net = TorchNALoader.load(cfg=cfg["decoder_cfg"])
        for k in ("encoder_cfg", "decoder_cfg"):
            del cfg[k]

        model = cls(encoder_net, decoder_net, **cfg)
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return model
