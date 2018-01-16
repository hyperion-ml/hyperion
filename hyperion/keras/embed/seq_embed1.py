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

        self.model = None
        self.embed_net = None

        self.prepool_net = None
        self.pool_net = None
        self.left_context = 0
        self.right_context = 0
        self.begin_context = None
        self.end_context = None
        
        
    @property
    def x_dim(self):
        return self.embed_net1.get_input_shape_at(0)[-1]

    
    
    @property
    def pool_in_dim(self):
        return self.embed_net1.get_output_shape_at(0)[-1]


    
    @property
    def pool_out_dim(self):
        return self.embed_net2.get_input_shape_at(0)[-1]


    
    def build(self, max_seq_length=None):

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

        pooling_model = Model(self.model.input,
                              self.model.get_layer('pooling').output)
        pooling = pooling_model(self.model.input)
        outputs = []
        x = self.model.input
        for layer_name in layers:
            embed_i = Model(self.embed_net2.get_input_at(0),
                            self.embed_net2.get_layer(layer_name).get_output_at(0))(pooling)
            outputs.append(embed_i)

        self.embed_net = Model(x, outputs)



    def build_embed2(self, layers, seq_length=None,
                     left_context=0, right_context=0,
                     begin_context=None, end_context=None):

        if begin_context is None:
            begin_context = left_context
        if end_context is None:
            end_context = right_context

        self.left_context = left_context
        self.right_context = right_context
        self.begin_context = begin_context
        self.end_context = end_context
        
        if seq_length is None:
            seq_length = self.embed_net1.get_input_shape_at(0)[-2]
        print('seq length: %d' % seq_length)
        x = Input(shape=(seq_length, self.x_dim,))
        y1 = self.embed_net1(x)
        
        self.prepool_net = Model(x, y1)

        x2 = Input(shape=(None, self.pool_in_dim,))
        mask = Input(shape=(None,))
        #mask = CreateMask(0)(x2)

        if self.pooling == 'mean+std':
            pool = Concatenate(axis=-1, name='pooling')(
                GlobalWeightedMeanStdPooling1D()([x2, mask]))
        elif self.pooling == 'mean+logvar':
            pool = Concatenate(axis=-1, name='pooling')(
                GlobalWeightedMeanLogVarPooling1D()([x2, mask]))
        elif self.pooling == 'mean':
            pool = GlobalWeightedAveragePooling1D(name='pooling')([x2, mask])
        else:
            raise ValueError('Invalid pooling %s' % self.pooling)

        outputs = []
        for layer_name in layers:
            embed_i = Model(self.embed_net2.get_input_at(0),
                            self.embed_net2.get_layer(layer_name).get_output_at(0))(pool)
            outputs.append(embed_i)

        self.pool_net = Model([x2, mask], outputs)
        self.pool_net.summary()
        # xx = [np.zeros((1,10,400), dtype='float32'), np.zeros((1,10), dtype='float32')]
        # self.pool_net.predict(xx, batch_size=1)

        
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


    
    def predict_embed2(self, x, **kwargs):

        in_net_length = self.prepool_net.get_input_shape_at(0)[-2]
        out_net_length = self.prepool_net.get_output_shape_at(0)[-2]

        r = in_net_length/out_net_length
        assert np.ceil(r) == np.floor(r)
        r = int(r)
        
        assert np.ceil(self.left_context/r) == np.floor(self.left_context/r)
        assert np.ceil(self.right_context/r) == np.floor(self.right_context/r)
        assert np.ceil(self.begin_context/r) == np.floor(self.begin_context/r)
        assert np.ceil(self.end_context/r) == np.floor(self.end_context/r) 
        out_begin_context = self.begin_context/r
        out_end_context = self.end_context/r
        out_left_context = self.left_context/r
        out_right_context = self.right_context/r

        in_length = x.shape[-2]
        out_length = int(in_length/r)
        in_shift = in_net_length - self.left_context - self.right_context
        out_shift = int(in_shift/r)
        
        y = np.zeros((out_length, self.pool_in_dim), dtype=float_keras())
        mask = np.ones((1, out_length), dtype=float_keras())
        mask[0,:out_begin_context] = 0
        mask[0,out_length - out_end_context:] = 0

        num_batches = int(np.ceil((in_length-in_net_length)/in_shift+1))
        x_i = np.zeros((1,in_net_length, x.shape[-1]), dtype=float_keras())
        j_in = 0
        j_out = 0
        for i in xrange(num_batches):
            # if i == 0:
            #     left = self.begin_context/r
            # else:
            #     left = self.left_context/r
            # if i == num_batches-1:
            #     right = self.end_context/r
            # else:
            #     right = self.right_context/r
            k_in = min(j_in+in_net_length, in_length)
            k_out = min(j_out+out_net_length, out_length)
            l_in = k_in - j_in
            l_out = k_out - j_out

            x_i[0,:l_in] = x[j_in:k_in]
            y_i = self.prepool_net.predict(x_i, batch_size=1, **kwargs)[0]
            y[j_out:k_out] = y_i[:l_out]
            
            j_in += in_shift
            j_out += out_shift
            if i==0:
                j_out+=out_left_context

        print('debug embed2 %d %d %d' % (out_length, j_out-out_shift, j_out-out_shift+l_out))
        y = np.expand_dims(y, axis=0)
        embeds = self.pool_net.predict([y, mask], batch_size=1, **kwargs)
        return np.hstack(tuple(embeds))


    
    @property
    def embed_dim2(self):
        if self.pool_net is None:
            return None
        embed_dim=0
        for node in xrange(len(self.pool_net.inbound_nodes)):
            output_shape = self.pool_net.get_output_shape_at(node)
            if isinstance(output_shape, list):
                for shape in output_shape:
                    embed_dim += shape[-1]
            else:
                embed_dim += output_shape[-1]

        return embed_dim

    
    
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
        config = SeqEmbed.load_config(file_config)
        
        file_model = '%s.net1.h5' % (file_path)
        embed_net1 = load_model(file_model, custom_objects=get_keras_custom_obj())
        file_model = '%s.net2.h5' % (file_path)
        embed_net2 = load_model(file_model, custom_objects=get_keras_custom_obj())

        filter_args = ('loss', 'pooling', 'name')
        kwargs = {k: config[k] for k in filter_args if k in config }
        return cls(embed_net1, embed_net2, **kwargs)

    
    
    
