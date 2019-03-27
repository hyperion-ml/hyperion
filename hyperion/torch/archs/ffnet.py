"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

import torch.nn as nn
from torch.nn import Linear, BatchNorm1d, Dropout

from ..helpers import ActivationFactory as AF

class FFNetV1(nn.Module):

    def __init__(self, num_layers, 
                 output_units, hidden_units, input_units,
                 output_activation=None, hidden_activation='relu',
                 use_batchnorm=True, dropout_rate=0,
                 output_batchnorm=False, output_dropout=False):

        super(FFNetV1, self).__init__()
        assert num_layers >= 1, 'num_layers (%d < 1)' % num_layers

        self.num_layers = num_layers
        self.output_units = output_units
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        self.output_batchnorm = output_batchnorm
        self.output_dropout = output_dropout
        
        self.output_activation = AF.create(output_activation)
        self.hidden_activation = AF.create(hidden_activation)
        
        if isinstance(hidden_units, list):
            assert num_layers-1 == len(hidden_units)
        else:
            hidden_units = [hidden_units for i in xrange(num_layers-1)]

        units = [input_units] + hidden_units
            
        layers = []
        for i in xrange(1, num_layers):
            if use_batchnorm:
                layers.append(BatchNorm1d(units[i-1]))
            
            layers.append(Linear(units[i-1], units[i]))
            if hidden_activation is not None:
                layers.append(self.hidden_activation)

            if dropout_rate > 0:
                layers.append(Dropout(dropout_rate))


        if use_batchnorm:
            layers.append(BatchNorm1d(units[-1]))

        layers.append(Linear(units[-1], output_units))

        if output_activation is not None:
            layers.append(self.output_activation)

        if output_dropout and dropout_rate > 0:
            layers.append(Dropout(dropout_rate))
        
        if output_batchnorm:
            layers.append(BatchNorm1d(output_units))

        self.layers = nn.ModuleList(layers)


    def forward(self, x):

        for l in xrange(len(self.layers)):
            x = self.layers[l](x)

        return x


    def get_config(self):
        
        output_activation = AF.get_config(self.output_activation)
        hidden_activation = AF.get_config(self.hidden_activation)

        config = {'num_layers': self.num_layers,
                  'output_units': self.output_units,
                  'hidden_units': self.hidden_units,
                  'input_units': self.input_units,
                  'use_batchnorm': self.use_batchnorm,
                  'dropout_rate': self.dropout_rate,
                  'output_batchnorm': self.output_batchnorm,
                  'output_dropout': self.output_dropout,
                  'output_activation': output_activation,
                  'hidden_activation': hidden_activation }
        base_config = super(FFNetV1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
