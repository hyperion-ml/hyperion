"""
x-vector categorical embeddings
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from keras import backend as K
from keras import optimizers
from keras import objectives
from keras.layers import Input, Concatenate, Multiply
from keras.models import Model, load_model, model_from_json

from .. import objectives as hyp_obj
from ..keras_utils import *
from ..layers import *

from ...hyp_model import HypModel
from .seq_embed import SeqEmbed

class SeqEmbedAtt(SeqEmbed):

    def __init__(self, embed_net1, embed_net2,
                 att_net, **kwargs):

        super(SeqEmbedAtt, self).__init__(embed_net1, embed_net2, **kwargs)
        self.att_net=att_net
        
        
    def build(self, max_seq_length=None):
        self.x_dim = self.embed_net1.internal_input_shapes[0][-1]

        x = Input(shape=(max_seq_length, self.x_dim,))
        mask = CreateMask(0)(x)
        frame_embed = self.embed_net1(x)

        weights = self.att_net(x)
        weights = Multiply()([weights, mask])
        
        if self.pooling == 'mean+std':
            pool = Concatenate(axis=-1, name='pooling')(GlobalWeightedMeanStdPooling1D()([frame_embed, mask]))
        elif self.pooling == 'mean+logvar':
            pool = Concatenate(axis=-1, name='pooling')(GlobalWeightedMeanLogVarPooling1D()([frame_embed, weights]))
        elif self.pooling == 'mean':
            pool = GlobalWeightedAveragePooling1D(name='pooling')([frame_embed, weights])
        else:
            raise ValueError('Invalid pooling %s' % self.pooling)

        y = self.embed_net2(pool)
        self.model = Model(x, y)
        self.model.summary()
            
    
    
    def save(self, file_path):
        file_model = '%s.json' % (file_path)
        with open(file_model, 'w') as f:
            f.write(self.to_json())
        
        file_model = '%s.net1.h5' % (file_path)
        self.embed_net1.save(file_model)
        file_model = '%s.net2.h5' % (file_path)
        self.embed_net2.save(file_model)
        file_model = '%s.att.h5' % (file_path)
        self.att_net.save(file_model)

        

    @classmethod
    def load(cls, file_path):
        file_config = '%s.json' % (file_path)
        with open(file_config,'r') as f:
            config=SeqEmbedAtt.load_config_from_json(f.read())

        file_model = '%s.net1.h5' % (file_path)
        embed_net1 = load_model(file_model, custom_objects=get_keras_custom_obj())
        file_model = '%s.net2.h5' % (file_path)
        embed_net2 = load_model(file_model, custom_objects=get_keras_custom_obj())
        file_model = '%s.att.h5' % (file_path)
        att_net = load_model(file_model, custom_objects=get_keras_custom_obj())

        filter_args = ('loss', 'pooling', 'name')
        kwargs = {k: config[k] for k in filter_args if k in config }
        return cls(embed_net1, embed_net2, att_net, **kwargs)

    
    
    
