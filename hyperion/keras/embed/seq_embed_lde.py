"""
x-vector categorical embeddings
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import os
import logging

import numpy as np

from keras import backend as K
from keras import optimizers
from keras import objectives
from keras.layers import Input, Concatenate, MaxPooling1D
from keras.models import Model, load_model, model_from_json

from .. import objectives as hyp_obj
from ..keras_utils import *
from ..layers import *
from ..losses import categorical_mbr
from ...hyp_model import HypModel


class SeqEmbedLDE(HypModel):

    def __init__(self, enc_net, pt_net,
                 loss='categorical_crossentropy',
                 pooling='mean+std',
                 lde_net=None,
                 num_comp=64,
                 lde_order=2,
                 left_context=0,
                 right_context=0,
                 begin_context=None,
                 end_context=None,
                 enc_downsampling=None,
                 **kwargs):

        super(SeqEmbedLDE, self).__init__(**kwargs)

        self.enc_net = enc_net
        self.pt_net = pt_net
        self.pooling = pooling
        self.loss = loss
        self.num_comp = num_comp
        self.lde_order = lde_order
        
        self.model = None
        self.pool_net = None
        self.lde_net = lde_net

        
        self.left_context = left_context
        self.right_context = right_context
        self.begin_context = left_context if begin_context is None else begin_context
        self.end_context = right_context if end_context is None else end_context
        self._enc_downsampling = enc_downsampling
        self.max_seq_length = None

        
    @property
    def x_dim(self):
        return self.enc_net.get_input_shape_at(0)[-1]


    @property
    def num_classes(self):
        return self.pt_net.get_output_shape_at(0)[-1]


    
    @property
    def pool_in_dim(self):
        return self.enc_net.get_output_shape_at(0)[-1]


    
    @property
    def pool_out_dim(self):
        return self.pt_net.get_input_shape_at(0)[-1]


    
    @property
    def in_length(self):
        if self.max_seq_length is None:
            return self.enc_net.get_input_shape_at(0)[-2]
        return self.max_seq_length


    
    @property
    def pool_in_length(self):
        pool_length = self.enc_net.get_output_shape_at(0)[-2]
        if pool_length is None:
            in_length = self.in_length
            if in_length is None:
                return None
            x = Input(shape=(in_length, self.x_dim))
            net = Model(x, self.enc_net(x))
            pool_length = net.get_output_shape_at(0)[-2]
        return pool_length


    
    @property
    def enc_downsampling(self):
        if self._enc_downsampling is None:
            assert self.in_length is not None
            assert self.pool_in_length is not None
            r = self.in_length/self.pool_in_length
            assert np.ceil(r) == np.floor(r)
            self._enc_downsampling = int(r)
        return self._enc_downsampling


    
    def _apply_pooling(self, x, mask):
        
        if self.pooling == 'mean+std':
            pool = Concatenate(axis=-1, name='pooling')(
                GlobalWeightedMeanStdPooling1D(name='mean--std')([x, mask]))
        elif self.pooling == 'mean+logvar':
            pool = Concatenate(axis=-1, name='pooling')(
                GlobalWeightedMeanLogVarPooling1D(name='mean--logvar')([x, mask]))
        elif self.pooling == 'mean':
            pool = GlobalWeightedAveragePooling1D(name='pooling')([x, mask])
        elif self.pooling == 'lde':
            if self.lde_net is None:
                xx = Input(shape=(None, self.pool_in_dim))
                zz = Input(shape=(None,1))
                yy = LDE1D(num_comp=self.num_comp, order=self.lde_order, name='lde')([xx, zz])
                self.lde_net = Model(input=[xx, zz], output=yy, name='pooling')
            pool = self.lde_net([x, mask])
        else:
            raise ValueError('Invalid pooling %s' % self.pooling)

        return pool


    
    def compile(self, metrics=None, **kwargs):

        if self.loss == 'categorical_mbr':
            loss = categorical_mbr
        else:
            loss = self.loss
        
        if metrics is None:
            self.model.compile(loss=loss, **kwargs)
        else:
            self.model.compile(loss=loss,
                               metrics=metrics,
                               weighted_metrics=metrics, **kwargs)


        
    def freeze_enc_net(self):
        self.enc_net.trainable = False

        
    def freeze_enc_net_layers(self, layers):
        for layer_name in layers:
            self.enc_net.get_layer(layer_name).trainable = False


            
    def freeze_pt_net_layers(self, layers):
        for layer_name in layers:
            self.pt_net.get_layer(layer_name).trainable = False

        
    def build(self, max_seq_length=None):

        if max_seq_length is None:
            max_seq_length = self.enc_net.get_input_shape_at(0)[-2]
        self.max_seq_length = max_seq_length

        x = Input(shape=(max_seq_length, self.x_dim,))
        mask = CreateMask(0)(x)
        frame_embed = self.enc_net(x)

        dec_ratio = int(max_seq_length/frame_embed._keras_shape[1])
        if dec_ratio > 1:
            mask = MaxPooling1D(dec_ratio, padding='same')(mask)
        
        pool = self._apply_pooling(frame_embed, mask)
        y = self.pt_net(pool)
        self.model = Model(x, y)
        self.model.summary()
        
        

    def build_embed(self, layers):

        frame_embed = Input(shape=(None, self.pool_in_dim,))
        mask = Input(shape=(None,))
        pool = self._apply_pooling(frame_embed, mask)
        
        outputs = []
        for layer_name in layers:
            embed_i = Model(self.pt_net.get_input_at(0),
                            self.pt_net.get_layer(layer_name).get_output_at(0))(pool)
            outputs.append(embed_i)

        self.pool_net = Model([frame_embed, mask], outputs)
        self.pool_net.summary()


        
    def predict_embed(self, x, **kwargs):

        in_seq_length = self.in_length
        pool_seq_length = self.pool_in_length
        r = self.enc_downsampling
        
        assert np.ceil(self.left_context/r) == np.floor(self.left_context/r)
        assert np.ceil(self.right_context/r) == np.floor(self.right_context/r)
        assert np.ceil(self.begin_context/r) == np.floor(self.begin_context/r)
        assert np.ceil(self.end_context/r) == np.floor(self.end_context/r) 
        pool_begin_context = int(self.begin_context/r)
        pool_end_context = int(self.end_context/r)
        pool_left_context = int(self.left_context/r)
        pool_right_context = int(self.right_context/r)

        in_length = x.shape[-2]
        pool_length = int(in_length/r)
        in_shift = in_seq_length - self.left_context - self.right_context
        pool_shift = int(in_shift/r)
        
        y = np.zeros((pool_length, self.pool_in_dim), dtype=float_keras())
        mask = np.ones((1, pool_length), dtype=float_keras())
        mask[0,:pool_begin_context] = 0
        mask[0,pool_length - pool_end_context:] = 0

        num_batches = max(int(np.ceil((in_length-in_seq_length)/in_shift+1)), 1)
        x_i = np.zeros((1,in_seq_length, x.shape[-1]), dtype=float_keras())
        j_in = 0
        j_out = 0
        for i in xrange(num_batches):
            k_in = min(j_in+in_seq_length, in_length)
            k_out = min(j_out+pool_seq_length, pool_length)
            l_in = k_in - j_in
            l_out = k_out - j_out

            x_i[0,:l_in] = x[j_in:k_in]
            y_i = self.enc_net.predict(x_i, batch_size=1, **kwargs)[0]
            y[j_out:k_out] = y_i[:l_out]

            j_in += in_shift
            j_out += pool_shift
            if i==0:
                j_out += pool_left_context
        logging.debug(pool_seq_length)
        logging.debug(pool_left_context)
        logging.debug(pool_right_context)
        logging.debug(pool_begin_context)
        logging.debug(pool_end_context)
        logging.debug('embed2 %d %d %d' % (pool_length, j_out-pool_shift, j_out-pool_shift+l_out))
        y = np.expand_dims(y, axis=0)
        embeds = self.pool_net.predict([y, mask], batch_size=1, **kwargs)
        return np.hstack(tuple(embeds))


    
    @property
    def embed_dim(self):
        if self.pool_net is None:
            return None
        embed_dim=0
        for node in xrange(len(self.pool_net._inbound_nodes)):
            output_shape = self.pool_net.get_output_shape_at(node)
            if isinstance(output_shape, list):
                for shape in output_shape:
                    embed_dim += shape[-1]
            else:
                embed_dim += output_shape[-1]

        return embed_dim


    
    def build_eval(self):

        frame_embed = Input(shape=(None, self.pool_in_dim,))
        mask = Input(shape=(None,))
        pool = self._apply_pooling(frame_embed, mask)
        
        score = self.pt_net(pool)
        self.pool_net = Model([frame_embed, mask], score)
        self.pool_net.summary()


        
    def predict_eval(self, x, **kwargs):
        return np.log(self.predict_embed(x, **kwargs)+1e-10)

    
    
    def fit(self, x, y, **kwargs):
        self.model.fit(x, y, **kwargs)


        
    def fit_generator(self, generator, steps_per_epoch, **kwargs):
        self.model.fit_generator(generator, steps_per_epoch, **kwargs)


        
    def get_config(self):
        config = { 'pooling': self.pooling,
                   'loss': self.loss,
                   'num_comp': self.num_comp,
                   'left_context': self.left_context,
                   'right_context': self.right_context,
                   'begin_context': self.begin_context,
                   'end_context': self.end_context}
        base_config = super(SeqEmbedLDE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    
    def save(self, file_path):
        file_model = '%s.json' % (file_path)
        with open(file_model, 'w') as f:
            f.write(self.to_json())
        
        file_model = '%s.enc.h5' % (file_path)
        self.enc_net.save(file_model)
        file_model = '%s.pt.h5' % (file_path)
        self.pt_net.save(file_model)

        if self.pooling == 'lde':
            file_model = '%s.lde.h5' % (file_path)
            self.lde_net.save(file_model)

            
    @classmethod
    def load(cls, file_path):
        file_config = '%s.json' % (file_path)        
        config = SeqEmbedLDE.load_config(file_config)
        
        file_model = '%s.enc.h5' % (file_path)
        enc_net = load_model(file_model, custom_objects=get_keras_custom_obj())
        file_model = '%s.pt.h5' % (file_path)
        pt_net = load_model(file_model, custom_objects=get_keras_custom_obj())

        file_model = '%s.lde.h5' % (file_path)
        lde_net = None
        if os.path.exists(file_model):
            lde_net = load_model(file_model, custom_objects=get_keras_custom_obj())
        
        
        filter_args = ('loss', 'pooling', 'num_comp', 'lde_order',
                       'left_context', 'right_context',
                       'begin_context', 'end_context', 'name')
        kwargs = {k: config[k] for k in filter_args if k in config }
        return cls(enc_net, pt_net, lde_net=lde_net, **kwargs)
    
    
    
    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('pooling', 'num_comp', 'lde_order', 'left_context', 'right_context',
                      'begin_context', 'end_context')
        return dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)

        
        
    @staticmethod
    def add_argparse_args(parser, prefix=None):
        if prefix is None:
            p1 = '--'
            p2 = ''
        else:
            p1 = '--' + prefix + '-'
            p2 = prefix + '_'

        parser.add_argument(p1+'pooling', dest=p2+'pooling', default='lde',
                            choices=['mean+std', 'mean+logvar', 'mean', 'lde'])
        parser.add_argument(p1+'lde-num-comp', dest=(p2+'num_comp'),
                            default=64, type=int)
        parser.add_argument(p1+'lde-order', dest=(p2+'lde_order'),
                            default=2, type=int)
        parser.add_argument(p1+'left-context', dest=(p2+'left_context'),
                            default=0, type=int)
        parser.add_argument(p1+'right-context', dest=(p2+'right_context'),
                            default=0, type=int)
        parser.add_argument(p1+'begin-context', dest=(p2+'begin_context'),
                            default=None, type=int)
        parser.add_argument(p1+'end-context', dest=(p2+'end_context'),
                            default=None, type=int)

