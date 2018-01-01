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
from keras.layers import Input, Concatenate
from keras.models import Model, load_model, model_from_json

from .. import objectives as hyp_obj
from ..keras_utils import *
from ..layers import *

from ...hyp_model import HypModel


class SeqEmbed(HypModel):

    def __init__(self, embed_net1, embed_net2,
                 loss='categorical_crossentropy',
                 pooling='mean+std',
                 **kwargs):

        super(SeqEmbed, self).__init__(**kwargs)

        self.embed_net1 = embed_net1
        self.embed_net2 = embed_net2
        self.pooling = pooling
        self.loss = loss

        self.x_dim = None
        self.model = None
        self.embed_net = None

        
        
    def build(self, max_seq_length=None):
        self.x_dim = self.embed_net1.get_input_shape_at(0)[-1]

        x = Input(shape=(max_seq_length, self.x_dim,))
        mask = CreateMask(0)(x)
        frame_embed = self.embed_net1(x)

        if self.pooling == 'mean+std':
            pool = Concatenate(axis=-1, name='pooling')(GlobalWeightedMeanStdPooling1D()([frame_embed, mask]))
        elif self.pooling == 'mean+logvar':
            pool = Concatenate(axis=-1, name='pooling')(GlobalWeightedMeanLogVarPooling1D()([frame_embed, mask]))
        elif self.pooling == 'mean':
            pool = GlobalWeightedAveragePooling1D(name='pooling')([frame_embed, mask])
        else:
            raise ValueError('Invalid pooling %s' % self.pooling)

        y = self.embed_net2(pool)
        self.model = Model(x, y)
        self.model.summary()
        
    def compile(self, **kwargs):
        print(self.loss)
        self.model.compile(loss=self.loss, **kwargs)

        
        
    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)


    
    def build_embed(self, layers):
        if isinstance(layers, str):
            layers = [layers]

        pooling_model = Model(self.model.input, self.model.get_layer('pooling').output)
        pooling = pooling_model(self.model.input)
        outputs = []
        x = self.model.input
        for layer_name in layers:
            embed_i = Model(self.embed_net2.get_input_at(0), self.embed_net2.get_layer(layer_name).get_output_at(0))(pooling)
            outputs.append(embed_i)

        self.embed_net = Model(x, outputs)


        
    @property
    def embed_dim(self):
        if self.embed_net is None:
            return None
        embed_dim=0
        for node in xrange(len(self.embed_net.inbound_nodes)):
            output_shape = self.embed_net.get_output_shape_at(node)
            if isinstance(output_shape, list):
                for shape in output_shape:
                    embed_dim += shape[-1]
            else:
                embed_dim += output_shape[-1]

        return embed_dim


    
    def predict_embed(self, x, **kwargs):
        embeds = self.embed_net.predict(x, **kwargs)
        return np.hstack(tuple(embeds))

    
    
    def fit(**kwargs):
        self.model.fit(**kwargs)


        
    def fit_generator(self, generator, steps_per_epoch, **kwargs):
        self.model.fit_generator(generator, steps_per_epoch, **kwargs)


        
    def get_config(self):
        config = { 'pooling': self.pooling,
                   'loss': self.loss }
        base_config = super(SeqEmbed, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    
    def save(self, file_path):
        file_model = '%s.json' % (file_path)
        with open(file_model, 'w') as f:
            f.write(self.to_json())
        
        file_model = '%s.net1.h5' % (file_path)
        self.embed_net1.save(file_model)
        file_model = '%s.net2.h5' % (file_path)
        self.embed_net2.save(file_model)

        

    @classmethod
    def load(cls, file_path):
        file_config = '%s.json' % (file_path)
        with open(file_config,'r') as f:
            config=SeqEmbed.load_config_from_json(f.read())

        file_model = '%s.net1.h5' % (file_path)
        embed_net1 = load_model(file_model, custom_objects=get_keras_custom_obj())
        file_model = '%s.net2.h5' % (file_path)
        embed_net2 = load_model(file_model, custom_objects=get_keras_custom_obj())

        filter_args = ('loss', 'pooling', 'name')
        kwargs = {k: config[k] for k in filter_args if k in config }
        return cls(embed_net1, embed_net2, **kwargs)

    
    
    
