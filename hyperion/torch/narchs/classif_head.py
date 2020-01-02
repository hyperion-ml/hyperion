"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import numpy as np

import torch.nn as nn
from torch.nn import Linear

from ..layers import CosLossOutput, ArcLossOutput
from ..layer_blocks import FCBlock
from .net_arch import NetArch

class ClassifHead(NetArch):

    def __init__(self, in_feats, num_classes, embed_dim=256,
                 num_embed_layers=1, 
                 hid_act={'name':'relu', 'inplace': True}, 
                 loss_type='arc-softmax',
                 s=64, margin=0.3, margin_warmup_epochs=0,
                 use_norm=True, norm_before=True, 
                 dropout_rate=0):

        super(ClassifHead, self).__init__()
        assert num_embed_layers >= 1, 'num_embed_layers (%d < 1)' % num_embed_layers

        self.num_embed_layers = num_embed_layers
        self.in_feats = in_feats
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_norm = use_norm
        self.norm_before = norm_before
        
        self.dropout_rate = dropout_rate
        self.loss_type = loss_type
        self.s = s
        self.margin = margin
        self.margin_warmup_epochs = margin_warmup_epochs
        
        prev_feats = in_feats
        fc_blocks = []
        for i in range(num_embed_layers-1):
            fc_blocks.append(
                FCBlock(prev_feats, embed_dim, 
                        activation=hid_act,
                        dropout_rate=dropout_rate,
                        use_norm=use_norm, 
                        norm_before=norm_before))
            prev_feats = embed_dim
                
        if loss_type != 'softmax':
            act = None
        else:
            act = hid_act

        fc_blocks.append(
            FCBlock(prev_feats, embed_dim, 
                    activation=act,
                    use_norm=use_norm, 
                    norm_before=norm_before))

        self.fc_blocks = nn.ModuleList(fc_blocks)

        # output layer
        if loss_type == 'softmax':
            self.output = Linear(embed_dim, num_classes)
        elif loss_type == 'cos-softmax':
            self.output = CosLossOutput(
                embed_dim, num_classes, 
                s=s, margin=margin, margin_warmup_epochs=margin_warmup_epochs)
        elif loss_type == 'arc-softmax':
            self.output = ArcLossOutput(
                embed_dim, num_classes, 
                s=s, margin=margin, margin_warmup_epochs=margin_warmup_epochs)
                

    def update_margin(self, epoch):
        if hasattr(self.output, 'update_margin'):
            self.output.update_margin(epoch)

                
    def forward(self, x, y=None):

        for l in range(self.num_embed_layers):
            x = self.fc_blocks[l](x)
        
        if self.loss_type == 'softmax':
            y = self.output(x)
        else:
            y = self.output(x, y)

        return y


    def extract_embed(self, x, embed_layer=0):

        for l in range(embed_layer):
            x = self.fc_blocks[l](x)

        y = self.fc_blocks[embed_layer].forward_linear(x)
        return y
                    
        
    
    def get_config(self):
        
        hid_act = AF.get_config(self.fc_blocks[0].activation)

        config = {
            'in_feats': self.in_feats,
            'num_classes': self.num_classes,
            'embed_dim': self.embed_dim,
            'num_embed_layers': self.num_embed_layers,
            'hid_act': hid_act,
            'lost_type': self.lost_type,
            's': self.s,
            'margin': self.margin,
            'margin_warmpu_epocs': self.margin_warmup_epochs,
            'use_norm': self.use_norm,
            'norm_before': self.norm_before,
            'dropout_rate': self.dropout_rate
        }
        
        base_config = super(ClassifHead, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
