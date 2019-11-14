"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn

from .xvector import XVector
from ..narchs import TDNNV1, ETDNNV1, ResETDNNV1


class TDNNXVector(XVector):

    def __init__(self, tdnn_type, num_enc_blocks, 
                 in_feats, enc_hid_units, enc_expand_units=None,
                 kernel_size=3, dilation=1, dilation_factor=1,
                 pool_net, pool_feats, 
                 num_classes, embed_dim=256,
                 s=64, margin=0.3, margin_inc_epochs=0,
                 num_embed_layers=1, 
                 hid_act={'name':'relu', 'inplace':True}, 
                 dropout_rate=0,
                 use_norm=True, 
                 norm_before=False,
                 in_norm=True, embed_layer=0, proj_feats=None):

        
        if tdnn_type == 'tdnn':
            encoder_net = TDNNV1(
                num_enc_blocks, in_feats, enc_hid_units, 
                kernel_size=kernel_size, 
                dilation=dilation, dilation_factor=dilation_factor,
                hid_act=hid_act, dropout_rate=dropout_rate,
                use_norm=use_norm, norm_before=norm_before, in_norm=in_norm)
        elif tdnn_type == 'etdnn':
            encoder_net = ETDNNV1(
                num_enc_blocks, in_feats, enc_hid_units, 
                kernel_size=kernel_size, 
                dilation=dilation, dilation_factor=dilation_factor,
                hid_act=hid_act, dropout_rate=dropout_rate,
                use_norm=use_norm, norm_before=norm_before, in_norm=in_norm)
        elif tdnn_type == 'resetdnn':
            if enc_expand_units is None:

                enc_expand_units = enc_hid_units
            encoder_net = ResETDNNV1(
                num_enc_blocks, in_feats, enc_hid_units, enc_expand_units,
                kernel_size=kernel_size, 
                dilation=dilation, dilation_factor=dilation_factor,
                hid_act=hid_act, dropout_rate=dropout_rate,
                use_norm=use_norm, norm_before=norm_before, in_norm=in_norm)
        else:
            raise Exception('%s is not valid TDNN network' % (tdnn_type))
        
        
        super(TDNNXVector, self).__init__(
            encoder_net, pool_net, pool_feats, 
            num_classes, embed_dim, num_embed_layers, hid_act,
            loss_type, s=64, margin=0.3, margin_inc_epochs=0,
            use_norm=use_norm, norm_before=norm_before, 
            dropout_rate=dropout_rate,
            embed_layer=embed_layer, 
            in_feats=None, enc_feats=None, proj_feats=proj_feats)


        self.tdnn_type = tdnn_type
        
    @property
    def num_enc_blocks(self):
        return self.encoder_net.num_blocks

    @property
    def in_feats(self):
        return self.encoder_net.in_units

    @property
    def enc_hid_units(self):
        return self.encoder_net.hid_units

    @property
    def enc_expand_units(self):
        try:
            return self.encoder_net.expand_units
        except:
            return None


    @property
    def kernel_size(self):
        return self.encoder_net.kernel_size

    @property
    def dilation(self):
        return self.encoder_net.dilation

    @property
    def dilation_factor(self):
        return self.encoder_net.dilation_factor



    def get_config(self):

        base_config = super(XVector, self).get_config()
        del base_config['encoder_cfg']

        pool_cfg = self.pool_net.get_config()

        config = {'tdnn_type': self.tdnn_type, 
                  'num_enc_blocks': self.num_enc_blocks, 
                  'in_feats': self.in_feats, 
                  'enc_hid_units': self.enc_hid_units, 
                  'enc_expand_units': self.enc_expand_units,
                  'kernel_size': self.kernel_size, 
                  'dilation': self.dilation, 
                  'dilation_factor': self.dilation_factor }

        config.update(base_config)
        return config


    @classmethod
    def load(cls, file_path=None, cfg=None, state_dict=None):
        cfg, state_dict = TorchModel._load_cfg_state_dict(
            file_path, cfg, state_dict)

        model = TDNNXVector(encoder_net, **cfg) 
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return model

