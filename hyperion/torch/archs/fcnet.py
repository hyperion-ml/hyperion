"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import torch.nn as nn
from torch.nn import Linear, BatchNorm1d, Dropout

from ..helpers import ActivationFactory as AF
from .net_arch import NetArch

class FCNetV1(NetArch):

    def __init__(self, num_hid_layers, 
                 input_units, hid_units, output_units,
                 hid_act='relu', output_act=None, 
                 use_batchnorm=True, dropout_rate=0,
                 without_output_layer=False,
                 use_output_batchnorm=False,
                 use_output_dropout=False):

        super(FCNetV1, self).__init__()
        assert num_hid_layers >= 1, 'num_hid_layers (%d < 1)' % num_hid_layers

        self.num_hid_layers = num_hid_layers
        self.output_units = output_units
        self.input_units = input_units
        self.hid_units = hid_units
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        self.without_output_layer = without_output_layer
        self.use_output_batchnorm = use_output_batchnorm
        self.use_output_dropout = use_output_dropout
        
        self.output_act = AF.create(output_act)
        
        if isinstance(hid_units, list):
            assert num_hid_layers == len(hid_units)
        else:
            hid_units = [hid_units for i in xrange(num_hid_layers)]

        units = [input_units] + hid_units

        #fully connected layers
        fc_layers = []
        for i in xrange(1, num_hid_layers+1):
            fc_layers.append(Linear(units[i-1], units[i]))

        self.fc_layers = nn.ModuleList(fc_layers)

        #hidden activations
        self.hid_acts = None
        if hid_act is not None:
            hid_acts = []
            for i in xrange(num_hid_layers):
                hid_act = AF.create(hid_act)
                hid_acts.append(hid_act)
            self.hid_acts = nn.ModuleList(hid_acts)
            
        #batch normalization
        self.batchnorm_layers = None
        if use_batchnorm:
            batchnorm_layers = []
            for i in xrange(num_hid_layers):
                batchnorm_layers.append(BatchNorm1d(units[i]))
            self.batchnorm_layers = nn.ModuleList(batchnorm_layers)

        # dropout
        self.dropout_layers = None
        if dropout_rate > 0:
            dropout_layers = []
            for i in xrange(num_hid_layers):
                dropout_layers.append(Dropout(dropout_rate))
            self.dropout_layers = nn.ModuleList(dropout_layers)

        # output layers
        self.output_dropout = None
        self.output_batchnorm = None

        if without_output_layer:
            if use_output_batchnorm:
                self.output_batchnorm = BatchNorm1d(units[-1])
        else:
            if use_batchnorm:
                self.batchnorm_layers.append(BatchNorm1d(units[-1]))

            self.fc_layers.append(Linear(units[-1], output_units))
            if use_output_dropout and dropout_rate > 0:
                self.output_dropout = Dropout(dropout_rate)
        
            if use_output_batchnorm:
                self.output_batchnorm = BatchNorm1d(output_units)


                
    def forward(self, x):

        for l in xrange(self.num_hid_layers):
            if self.use_batchnorm:
                x = self.batchnorm_layers[l](x)
                
            x = self.fc_layers[l](x)
            if self.hid_acts is not None:
                x = self.hid_acts[l](x)

            if self.dropout_rate > 0:
                x = self.dropout_layers[l](x)

        if not self.without_output_layer:
            if self.batchnorm_layers is not None:
                x = self.batchnorm_layers[self.num_hid_layers](x)
            
            x = self.fc_layers[self.num_hid_layers](x)
            if self.output_act is not None:
                x = self.output_act(x)

            if self.output_dropout is not None:
                x = self.droput_layers[self.num_hid_layers](x)

        if self.use_output_batchnorm:
            x = self.output_dropout(x)

        return x


    
    def get_config(self):
        
        output_act = AF.get_config(self.output_act)
        if self.hid_acts is None:
            hid_act = None
        else:
            hid_act = AF.get_config(self.hid_acts[0])

        config = {'num_hid_layers': self.num_hid_layers,
                  'output_units': self.output_units,
                  'hid_units': self.hidden_units,
                  'input_units': self.input_units,
                  'use_batchnorm': self.use_batchnorm,
                  'dropout_rate': self.dropout_rate,
                  'use_output_batchnorm': self.output_batchnorm,
                  'use_output_dropout': self.output_dropout,
                  'output_act': output_act,
                  'hid_act': hid_act }
        
        base_config = super(FCNetV1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
