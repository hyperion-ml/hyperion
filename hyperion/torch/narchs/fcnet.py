"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import torch.nn as nn
from torch.nn import Linear, BatchNorm1d, Dropout

from ..layers import ActivationFactory as AF
from .net_arch import NetArch
from ..layer_blocks import FCBlock

class FCNetV1(NetArch):

    def __init__(self, num_blocks, 
                 in_units, hid_units, out_units=0,
                 hid_act={'name':'relu', 'inplace':True}, out_act=None, 
                 dropout_rate=0,
                 use_norm=True, 
                 norm_before=False,
                 in_norm=True):

        super(FCNetV1, self).__init__()

        self.num_blocks = num_blocks
        self.out_units = out_units
        self.in_units = in_units
        self.hid_units = hid_units
        self.dropout_rate = dropout_rate
        self.use_norm = use_norm
        self.norm_before = norm_before
        self.in_norm = in_norm


        if isinstance(hid_units, list):
            assert num_blocks == len(hid_units)
        else:
            hid_units = [hid_units for i in range(num_blocks)]
        
        units = [in_units] + hid_units
        hid_act='relu'
        blocks = []
        for i in range(1, num_blocks+1):
            print(hid_act)
            blocks.append(
                FCBlock(units[i-1], units[i],
                        activation=hid_act, dropout_rate=dropout_rate, 
                        use_norm=use_norm, norm_before=norm_before))

        self.blocks = nn.ModuleList(blocks)

        self.with_output = False
        if out_units == 0:
            return 

        self.with_output = True
        self.out_act = AF.create(out_act)

        self.output = Linear(units[-1], out_units)


    def forward(self, x):

        for i in range(self.num_blocks):
            x = self.blocks[i](x)

        if self.with_output:
            x = self.output(x)
            if self.out_act is not None:
                x = self.out_act(x)

        return x

    
    def get_config(self):
        
        out_act = AF.get_config(self.out_act)
        hid_act =  AF.get_config(self.blocks[0].activation)

        config = {'num_blocks': self.num_blocks,
                  'in_units': self.in_units,
                  'hid_units': self.hid_units,
                  'out_units': self.out_units,
                  'dropout_rate': self.dropout_rate,
                  'use_norm': self.use_norm,
                  'norm_before': self.norm_before,
                  'in_norm' : self.in_norm,
                  'out_act': out_act,
                  'hid_act': hid_act}
        
        base_config = super(FCNetV1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


