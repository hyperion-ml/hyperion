"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging

import torch
import torch.nn as nn

from ...torch_model import TorchModel
from ...helpers import TorchNALoader

class AE(TorchModel):
    """Autoencoder class
    """

    def __init__(self, encoder_net, decoder_net):
        super(AE, self).__init__()
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net


    def forward(self, x):
        z = self.encoder_net(x)
        xhat = self.decoder_net(z)
        logging.info('x-shape={} z-shape={} x-hat-shape={}'.format(
            x.shape, z.shape, xhat.shape))
        return xhat


    def get_config(self):
        enc_cfg = self.encoder_net.get_config()
        dec_cfg = self.decoder_net.get_config()
        config = {'encoder_cfg': enc_cfg,
                  'decoder_cfg': dec_cfg }
        base_config = super(AE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    @classmethod
    def load(cls, file_path=None, cfg=None, state_dict=None):
        cfg, state_dict = cls._load_cfg_state_dict(
            file_path, cfg, state_dict)

        encoder_net = TorchNALoader.load(cfg=cfg['encoder_cfg'])
        decoder_net = TorchNALoader.load(cfg=cfg['decoder_cfg'])
        for k in ('encoder_cfg', 'decoder_cfg'):
            del cfg[k]
        
        model = cls(encoder_net, decoder_net, **cfg) 
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return model

        
