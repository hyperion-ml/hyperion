"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
#from __future__ import absolute_import

#import numpy as np

import torch.nn as nn
from torch.nn import Linear

from ..layers import CosLossOutput, ArcLossOutput
from ..layers import NormLayer1dFactory as NLF
from ..layer_blocks import FCBlock
from .net_arch import NetArch

class ClassifHead(NetArch):
    """Classification Head for x-vector style networks

    Attributes:
       in_feats: input features
       num_classes: number of output classes
       embed_dim: dimension of embedding layer
       num_embed_layers: number of hidden layers
       hid_act: str or dict hidden activation type in ['relu', 'relu6', 'swish', ... ]
       loss_type: type of loss function that will be used with the x-vector in ['softmax', 'cos-softmax', 'arc-softmax'],
                  corresponding to standard cross-entorpy, additive margin softmax or additive angular margin softmax.
       s: scale parameter for cos-softmax and arc-softmax
       margin: margin parameter for cos-softmax and arc-softmax
       margin_warmup_epochs: number of epochs to anneal the margin from 0 to margin
       norm_layer: norm_layer object or str indicating type norm layer, if None it uses BatchNorm1d
       use_norm: it True it uses layer/batch-normalization
       norm_before: if True, layer-norm is before the activation function

    """

    def __init__(self, in_feats, num_classes, embed_dim=256,
                 num_embed_layers=1, 
                 hid_act={'name':'relu', 'inplace': True}, 
                 loss_type='arc-softmax',
                 s=64, margin=0.3, margin_warmup_epochs=0,
                 norm_layer=None,
                 use_norm=True, norm_before=True, 
                 dropout_rate=0):

        super().__init__()
        assert num_embed_layers >= 1, 'num_embed_layers (%d < 1)' % num_embed_layers

        self.num_embed_layers = num_embed_layers
        self.in_feats = in_feats
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.norm_layer = norm_layer
        
        if use_norm:
            norm_groups = None
            if norm_layer == 'group-norm':
                norm_groups = min(embed_dim//8, 32)
            self._norm_layer = NLF.create(norm_layer, norm_groups)
        else:
            self._norm_layer = None
            
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
                        norm_layer=self._norm_layer,
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
                    norm_layer=self._norm_layer,
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


    def rebuild_output_layer(self, num_classes, loss_type, s, margin, margin_warmup_epochs):
        embed_dim = self.embed_dim
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.s = s
        self.margin = margin
        self.margin_warmup_epochs = margin_warmup_epochs

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


    def set_margin(self, margin):
        if self.loss_type == 'softmax':
            return

        self.margin = margin
        self.output.margin = margin


    def set_margin_warmup_epochs(self, margin_warmup_epochs):
        if self.loss_type == 'softmax':
            return

        self.margin_warmup_epochs = margin_warmup_epochs
        self.output.margin_warmup_epochs = margin_warmup_epochs


    def set_s(self, s):
        if self.loss_type == 'softmax':
            return

        self.s = s
        self.output.s = s

    
    def update_margin(self, epoch):
        if hasattr(self.output, 'update_margin'):
            self.output.update_margin(epoch)


    def freeze_layers(self, layer_list):
        for l in layer_list:
            for param in self.fc_blocks[l].parameters():
                param.requires_grad = False

    def put_layers_in_eval_mode(self, layer_list):
        for l in layer_list:
            self.fc_blocks[l].eval()
    
                
    def forward(self, x, y=None):

        for l in range(self.num_embed_layers):
            x = self.fc_blocks[l](x)
        
        if self.loss_type == 'softmax':
            y = self.output(x)
        else:
            y = self.output(x, y)

        return y


    def forward_hid_feats(self, x, y=None, layers=None, return_output=False):

        assert layers is not None or return_output
        if layers is None:
            layers = []

        h = []
        for l in range(self.num_embed_layers):
            x = self.fc_blocks[l](x)
            if l in layers:
                h.append(x)
        
        if self.loss_type == 'softmax':
            y = self.output(x)
        else:
            y = self.output(x, y)

        if return_output:
            return h, y
        return h


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
            'margin_warmup_epochs': self.margin_warmup_epochs,
            'norm_layer': self.norm_layer,
            'use_norm': self.use_norm,
            'norm_before': self.norm_before,
            'dropout_rate': self.dropout_rate
        }
        
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
