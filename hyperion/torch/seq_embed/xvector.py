"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn


from ..layers import GlobalPool1dFactory as PF
from ..layers import _GlobalPool1d
from ..layer_blocks import TDNNBlock
from ..narchs import ClassifHead
from ..torch_model import TorchModel
from ..helpers import TorchNALoader

class XVector(TorchModel):

    def __init__(encoder_net, 
                 pool_net, pool_feats, 
                 num_classes, embed_dim=256,
                 num_embed_layers=1, 
                 hid_act={'name':'relu', 'inplace': True}, 
                 loss_type='arc-softmax',
                 s=64, margin=0.3, margin_inc_epochs=0,
                 use_norm=True, norm_before=True, 
                 dropout_rate=0,
                 embed_layer=0, 
                 in_feats=None, enc_feats=None, proj_feats=None):

        self.encoder_net = encoder_net

        self.pool_net = _make_pool_net(pool_net)

        self.proj = None:
        self.proj_feats = proj_feats
        if proj_feats is not None:
            self.proj = TDNNBlock(enc_feats, proj_feats, kernel_size=1, 
                                  activation=None, use_norm=use_norm)
            

        self.classif_net = ClassifHead(
            pool_feats, num_classes, embed_dim=embed_dim,
            num_embed_layers=num_embed_layers, 
            hid_act=hid_act,
            loss_type=loss_type,
            s=s, margin=margin, margin_inc_epochs=margin_inc_epochs,
            use_norm=use_norm, norm_before=norm_before, 
            dropout_rate=dropout_rate)

        self.hid_act = hid_act
        self.use_norm = use_norm
        self.norm_before = norm_before
        self.dropout_rate = dropout_rate
        self.embed_layer = embed_layer

    @property
    def pool_feats(self):
        return self.classif_net.in_feats

    @property
    def num_classes(self):
        return self.classif_net.num_classes

    @propetry
    def embed_dim(self):
        return self.classif_net.embed_dim

    @property
    def num_embed_layers(self):
        return self.classif_net.num_embed_layers

    @property
    def s(self):
        return self.classif_net.s

    @property
    def margin(self):
        return


    @staticmethod
    def _make_pool_net(pool_net):
        if isinstance(pool_net, str):
            return PF.create(pool_net)
        elif isinstance(pool_net, dict):
            return PF.create(**pool_net)
        elif isinstance(pool_net, _GlobalPool1d):
            pass pool_net
        else:
            raise Exception('Invalid pool_net argument')


    def forward(self, x, y=None):

        if self.encoder_net.in_dim() == 4 and x.dim() == 3:
            x = x.view(x.size(0), 1, x.size(1), x.size(2))

        x = self.encoder_net(x)

        if self.encoder_net.out_dim() == 4:
            x = x.view(x.size(0), -1, x.size(-1))

        if self.proj is not None:
            x = self.proj(x)
            
        p = self.pool_net(x)
        y = self.classif_net(p, y)


    def extract_embed(self, x, chunk_length=0, embed_layer=None):

        if embed_layer is None:
            embed_layer = self.embed_layer

        if self.encoder_net.in_dim() == 4 and x.dim() == 3:
            x = x.view(x.size(0), 1, x.size(1), x.size(2))

        if chunk_length == 0:
            x = self.encoder_net(x)
        else:
            raise NotImplementedError()

        if self.encoder_net.out_dim() == 4:
            x = x.view(x.size(0), -1, x.size(-1))

        if self.proj is not None:
            x = self.proj(x)

        p = self.pool_net(x)
        y = self.classif_net.extract_embed(p, embed_layer)


    def get_config(self):

        enc_cfg = self.encoder_net.get_config()
        pool_cfg = self.pool_net.get_config()

        config = {'encoder_cfg': enc_cfg,
                  'pool_net': pool_cfg,
                  'pool_feats': self.pool_feats, 
                  'num_classes': self.num_classes,
                  'embed_dim': self.embed_dim,
                  'num_embed_layers': self.num_embed_layers,
                  'hid_act': self.hid_act,
                  'loss_type': self.loss_type,
                  's': self.s,
                  'margin': self.margin,
                  'margin_inc_epochs': self.margin_inc_epochs,
                  'use_norm': self.use_norm,
                  'norm_before': self.norm_before,
                  'dropout_rate': self.dropout_rate,
                  'embed_layers': self.embed_layers }

        base_config = super(XVector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    @classmethod
    def load(cls, file_path=None, cfg=None, state_dict=None):
        cfg, state_dict = TorchModel._load_cfg_state_dict(
            file_path, cfg, state_dict)

        # preproc_net = None
        # if 'preproc_cfg' in cfg:
        #     preproc_net = TorchNALoader.load(cfg=cfg['preproc_cfg'])
        #     del cfg['preproc_cfg']

        encoder_net = TorchNALoader.load(cfg=cfg['encoder_cfg'])

        for k in ('encoder_cfg')
            del cfg[k]
        
        model = XVector(encoder_net, **cfg) 
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return model


    
        

            
        
    
        
    
        

                          


    
                
        
