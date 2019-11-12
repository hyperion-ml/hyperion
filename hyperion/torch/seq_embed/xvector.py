"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn

from ..layers import GlobalPool1dFactory as PF
from ..narchs import ClassifHead
from ..torch_model import TorchModel
from ..helpers import TorchNALoader

class XVector(TorchModel):

    def __init__(encoder_net, pool_net, classif_net, preproc_net=None, embed_layer=0):

        self.encoder_net = encoder_net
        self.pool_cfg = pool_cfg
        if isinstance(pool_cfg, str):
            self.pool_net = PF.create(pool_cfg)
        else:
            self.pool_net = PF.create(**pool_cfg)

        self.classif_net = classif_net
        self.preproc_net = preproc_net
        self.embed_layer = embed_layer


    def forward(self, x):

        if self.preproc_net is not None:
            x = self.preproc_net(x)

        z = self.encoder_net(x)
        p = self.pool_net(z)
        y = self.classif_net(p)


    def extract_embed(self, x, chunk_length=0):

        if chunk_length == 0:
            if self.preproc_net is not None:
                x = self.preproc_net(x)
            z = self.encoder_net(x)
        else:
            raise NotImplementedError()

        p = self.pool_net(z)
        y = self.classif_net.extract_embed(p, self.embed_layers)


    def get_config(self):

        preproc_cfg = None if self.preproc_net is None else self.preproc_net.get_config()
        config = {'preproc_cfg': preproc_cfg,
                  'encoder_cfg': self.encoder_net.get_config(),
                  'pool_cfg': self.pool_net.get_config(),
                  'classif_cfg': self.classif_net.get_config(),
                  'embed_layers': self.embed_layers }
        
        return dict(list(base_config.items()) + list(config.items()))


    @classmethod
    def load(cls, file_path=None, cfg=None, state_dict=None):
        cfg, state_dict = TorchModel._load_cfg_state_dict(
            file_path, cfg, state_dict)

        preproc_net = None
        if 'preproc_cfg' in cfg:
            preproc_net = TorchNALoader.load(cfg=cfg['preproc_cfg'])
            del cfg['preproc_cfg']

        encoder_net = TorchNALoader.load(cfg=cfg['encoder_cfg'])
        classif_net = TorchNALoader.load(cfg=cfg['classif_cfg'])
        pool_cfg = cfg['pool_cfg']

        for k in ('encoder_cfg', 'classif_cfg', 'pool_cfg'):
            del cfg[k]

        return XVector(encoder_net, pool_cfg, classif_net,
                       preproc_net=preproc_net, **cfg)

    
        

    
        
    
        

                          


    
                
        
