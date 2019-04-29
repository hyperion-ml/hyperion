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
from torch.nn import Conv1d, Linear, BatchNorm1d

from ..helpers import ActivationFactory as AF
from ..layers import Dropout1d
from .net_arch import NetArch

class TDNNV1(NetArch):

    def __init__(self, num_td_layers, num_fc_layers, 
                 input_units, td_units, fc_units, output_units,
                 kernel_size, dilation=1, dilation_factor=1,
                 hid_act='relu', output_act=None, 
                 use_batchnorm=True, dropout_rate=0,
                 without_output_layer=False,
                 use_output_batchnorm=False,
                 use_output_dropout=False):

        super(TDNNV1, self).__init__()
        assert num_td_layers >= 1, 'num_td_layers (%d < 1)' % num_td_layers

        self.num_td_layers = num_td_layers
        self.num_fc_layers = num_fc_layers
        self.output_units = output_units
        self.input_units = input_units
        self.td_units = td_units
        self.fc_units = fc_units
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dilation_factor = dilation_factor
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        self.without_output_layer = without_output_layer
        self.use_output_batchnorm = use_output_batchnorm
        self.use_output_dropout = use_output_dropout
        
        self.output_act = AF.create(output_act)
        
        if isinstance(td_units, list):
            assert num_td_layers == len(td_units)
        else:
            td_units = [td_units for i in xrange(num_td_layers)]

        td_units = [input_units] + td_units

        if isinstance(fc_units, list):
            assert num_fc_layers == len(fc_units)
        else:
            fc_units = [fc_units for i in xrange(num_fc_layers)]

        fc_units = [td_units[-1]] + fc_units

        if isinstance(kernel_size, list):
            assert num_td_layers == len(kernel_size)
        else:
            kernel_size = [kernel_size for i in xrange(num_td_layers)]

        if isinstance(dilation_rate, list):
            assert num_td_layers == len(dilation_rate)
        else:
            dilation = [dilation_factor*i+dilation for i in xrange(num_td_layers)]

        # past and future context
        self._context = int(np.sum(np.array(dilation)*(
            np.array(kernel_size)-1)/2))

        # time delay layers
        td_layers = []
        for i in xrange(1, num_hid_layers+1):
            td_layers.append(Conv1D(td_units[i-1], td_units[i],
                                    kernel_size=kernel_size[i],
                                    dilation=dilation[i]))

        self.td_layers = nn.ModuleList(td_layers)
        
        # fully connected layers
        fc_layers = []
        for i in xrange(1, num_fc_layers+1):
            fc_layers.append(Linear(fc_units[i-1], fc_units[i]))

        self.fc_layers = nn.ModuleList(fc_layers)

        # hidden activations
        self.hid_acts = None
        if hid_act is not None:
            hid_acts = []
            for i in xrange(num_td_layers+num_num_fc_layers):
                hid_act = AF.create(hid_act)
                hid_acts.append(hid_act)
            self.hid_acts = nn.ModuleList(hid_acts)
            
        # batchnorm layers
        self.batchnorm_layers = None
        if use_batchnorm:
            batchnorm_layers = []
            for i in xrange(num_td_layers):
                batchnorm_layers.append(BatchNorm1d(td_units[i]))
            for i in xrange(num_fc_layers):
                batchnorm_layers.append(BatchNorm1d(fc_units[i]))

            self.batchnorm_layers = nn.ModuleList(batchnorm_layers)

        # dropout layers
        self.dropout_layers = None
        if dropout_rate > 0:
            dropout_layers = []
            for i in xrange(num_td_layers+num_fc_layers):
                dropout_layers.append(Dropout1d(dropout_rate))
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
                self.output_dropout = Dropout1d(dropout_rate)
        
            if use_output_batchnorm:
                self.output_batchnorm = BatchNorm1d(output_units)



    @property
    def context(self):
        return (self._context, self._context)
    
                
    def forward(self, x):

        for l in xrange(self.num_td_layers+self.num_fc_layers):
            if self.use_batchnorm:
                x = self.batchnorm_layers[l](x)

            if i < self.num_td_layers:
                x = self.td_layers[l](x)
            else:
                x = self.fc_layers[l-self.num_td_layers](x)
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

        config = {'num_td_layers': self.num_td_layers,
                  'num_fc_layers': self.num_fc_layers,
                  'output_units': self.output_units,
                  'td_units': self.td_units,
                  'fc_units': self.fc_units,
                  'input_units': self.input_units,
                  'kernel_size': self.kernel_size,
                  'dilation': self.dilation,
                  'dilation_factor': self.dilation_factor,
                  'use_batchnorm': self.use_batchnorm,
                  'dropout_rate': self.dropout_rate,
                  'use_output_batchnorm': self.output_batchnorm,
                  'use_output_dropout': self.output_dropout,
                  'output_act': output_act,
                  'hid_act': hid_act }
        
        base_config = super(TDNNV1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
