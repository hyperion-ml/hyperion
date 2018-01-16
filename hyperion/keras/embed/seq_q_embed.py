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
from keras.layers import Input, Concatenate, Activation
from keras.models import Model, load_model, model_from_json

from .. import objectives as hyp_obj
from ..keras_utils import *
from ..layers import *

from ...hyp_model import HypModel


class SeqQEmbed(HypModel):

    def __init__(self, prepool_net,
                 num_classes,
                 score_net=None,
                 post_pdf='diag_normal',
                 pooling_input='mean+logvar',
                 pooling_output='nat+prec',
                 left_context=0,
                 right_context=0,
                 begin_context=None,
                 end_context=None,
                 prepool_downsampling=None,
                 min_var=0.1, kl_weight=0, **kwargs):

        super(SeqQEmbed, self).__init__(**kwargs)

        self.prepool_net = prepool_net
        self.score_net = score_net
        self.post_pdf = post_pdf
        self.num_classes = num_classes
        self.pooling_input = pooling_input
        self.pooling_output = pooling_output
        self.min_var = min_var
        self.kl_weight = kl_weight
        
        self.model = None
        self.pool_net = None

        self.left_context = left_context
        self.right_context = right_context
        self.begin_context = left_context if begin_context is None else begin_context
        self.end_context = right_context if end_context is None else end_context
        self._prepool_downsampling = prepool_downsampling
        self.max_seq_length = None


        
    @property
    def x_dim(self):
        return self.prepool_net.get_input_shape_at(0)[-1]

    
    
    @property
    def embed_dim(self):
        return self.prepool_net.get_output_shape_at(0)[0][-1]


    
    @property
    def in_length(self):
        if self.max_seq_length is None:
            return self.prepool_net.get_input_shape_at(0)[-2]
        return self.max_seq_length


    
    @property
    def pool_in_length(self):
        pool_length = self.prepool_net.get_output_shape_at(0)[0][-2]
        if pool_length is None:
            in_length = self.in_length
            if in_length is None:
                return None
            x = Input(shape=(in_length, self.x_dim))
            net = Model(x, self.prepool_net(x))
            pool_length = net.get_output_shape_at(0)[0][-2]
        return pool_length


    
    @property
    def prepool_downsampling(self):
        if self._prepool_downsampling is None:
            assert self.in_length is not None
            assert self.pool_in_length is not None
            r = self.in_length/self.pool_in_length
            assert np.ceil(r) == np.floor(r)
            self._prepool_downsampling = int(r)
        return self._prepool_downsampling


        
    def build(self, max_seq_length=None):

        if max_seq_length is None:
            max_seq_length = self.prepool_net.get_input_shape_at(0)[-2]
        self.max_seq_length = max_seq_length

        x = Input(shape=(max_seq_length, self.x_dim,))
        mask = CreateMask(0)(x)
        frame_embed = self.prepool_net(x)

        q_embed = GlobalDiagNormalPostStdPriorPooling1D(
            input_format=self.pooling_input,
            output_format=self.pooling_output,
            min_var=self.min_var, name='pooling')(frame_embed+[mask])

        if self.score_net is None:
            p1_input = Input(shape=(self.embed_dim,))
            p2_input = Input(shape=(self.embed_dim,))
            score_input = [p1_input, p2_input]
            score_layer = CatQScoringDiagNormalHomoPostStdPrior(
                self.num_classes,
                input_format=self.pooling_output,
                q_class_format='mean+prec', name='scoring')
            self.score_net = Model(score_input, score_layer(score_input))
            
            
        score = self.score_net(q_embed)
        y = Activation('softmax', name='pclass_x')(score)

        kl_loss = KLDivNormalVsStdNormal(input_format=self.pooling_output, name='kl_loss')(q_embed)
        
        self.model = Model(x, [y, kl_loss])
        self.model.summary()


        
    def compile(self, **kwargs):

        kl_loss=(lambda y_true, y_pred: y_pred)
        self.model.compile(loss=['categorical_crossentropy', kl_loss],
                           loss_weights = [1, self.kl_weight],
                           metrics={'pclass_x': 'accuracy'},
                           weighted_metrics={'pclass_x': 'accuracy'}, **kwargs)

        
        
    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)


    
    def build_embed(self, pooling_output=None):
        if pooling_output is None:
            pooling_output = self.pooling_output
        p1_frame_embed = Input(shape=(None, self.embed_dim,))
        p2_frame_embed = Input(shape=(None, self.embed_dim,))
        mask = Input(shape=(None,))
        q_embed = GlobalDiagNormalPostStdPriorPooling1D(
            input_format=self.pooling_input,
            output_format=pooling_output,
            min_var=self.min_var, name='pooling')(
                [p1_frame_embed, p2_frame_embed, mask])
        
        self.pool_net = Model([p1_frame_embed, p2_frame_embed, mask], q_embed)
        self.pool_net.summary()
        

    
    def predict_embed(self, x, **kwargs):
        in_seq_length = self.in_length
        pool_seq_length = self.pool_in_length
        r = self.prepool_downsampling
        
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
        
        p1 = np.zeros((pool_length, self.embed_dim), dtype=float_keras())
        p2 = np.zeros((pool_length, self.embed_dim), dtype=float_keras())
        mask = np.ones((1, pool_length), dtype=float_keras())
        mask[0,:pool_begin_context] = 0
        mask[0,pool_length - pool_end_context:] = 0

        num_batches = int(np.ceil((in_length-in_seq_length)/in_shift+1))
        x_i = np.zeros((1,in_seq_length, x.shape[-1]), dtype=float_keras())
        j_in = 0
        j_out = 0
        for i in xrange(num_batches):
            k_in = min(j_in+in_seq_length, in_length)
            k_out = min(j_out+pool_seq_length, pool_length)
            l_in = k_in - j_in
            l_out = k_out - j_out

            x_i[0,:l_in] = x[j_in:k_in]
            p1_i, p2_i = self.prepool_net.predict(x_i, batch_size=1, **kwargs)
            p1[j_out:k_out] = p1_i[0,:l_out]
            p2[j_out:k_out] = p2_i[0,:l_out]
            
            j_in += in_shift
            j_out += pool_shift
            if i==0:
                j_out += pool_left_context

        p1 = np.expand_dims(p1, axis=0)
        p2 = np.expand_dims(p2, axis=0)
        q_embed = self.pool_net.predict([p1, p2, mask], batch_size=1, **kwargs)
        return q_embed

    
    
    def fit(**kwargs):
        self.model.fit(**kwargs)


        
    def fit_generator(self, generator, steps_per_epoch, **kwargs):
        self.model.fit_generator(generator, steps_per_epoch, **kwargs)


        
    def get_config(self):
        config = { 'num_classes': self.num_classes,
                   'post_pdf': self.post_pdf,
                   'pooling_input' : self.pooling_input,
                   'pooling_output': self.pooling_output,
                   'min_var': self.min_var,
                   'kl_weight': self.kl_weight,
                   'left_context': self.left_context,
                   'right_context': self.right_context,
                   'begin_context': self.begin_context,
                   'end_context': self.end_context}

        base_config = super(SeqQEmbed, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    
    def save(self, file_path):
        file_model = '%s.json' % (file_path)
        with open(file_model, 'w') as f:
            f.write(self.to_json())
        
        file_model = '%s.net1.h5' % (file_path)
        self.prepool_net.save(file_model)
        file_model = '%s.score.h5' % (file_path)
        self.score_net.save(file_model)

        

    @classmethod
    def load(cls, file_path):
        file_config = '%s.json' % (file_path)
        with open(file_config,'r') as f:
            config=SeqQEmbed.load_config_from_json(f.read())

        file_model = '%s.net1.h5' % (file_path)
        prepool_net = load_model(file_model, custom_objects=get_keras_custom_obj())
        file_model = '%s.score.h5' % (file_path)
        score_net = load_model(file_model, custom_objects=get_keras_custom_obj())

        filter_args = ('post_pdf', 'pooling_input', 'pooling_output',
                       'left_context', 'right_context',
                       'begin_context', 'end_context',
                       'min_var', 'kl_weight', 'name')
        kwargs = {k: config[k] for k in filter_args if k in config }
        return cls(prepool_net, config['num_classes'], score_net, **kwargs)

    
    
    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('post_pdf', 'pooling_input', 'pooling_output',
                      'min_var', 'kl_weight',
                      'left_context', 'right_context',
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


        parser.add_argument(p1+'post-pdf', dest=p2+'post_pdf', default='diag_normal',
                            choices=['diag_normal'])
        parser.add_argument(p1+'pooling-input', dest=p2+'pooling_input',
                            default='nat+logitvar',
                            choices=['nat+logitvar', 'nat+logprec-1',
                                     'nat+logvar', 'nat+logprec',
                                     'nat+var', 'nat+prec', 'nat+prec-1',
                                     'mean+logitvar', 'mean+logprec-1',
                                     'mean+logvar', 'mean+logprec',
                                     'mean+var', 'mean+prec', 'mean+prec-1'])
        parser.add_argument(p1+'pooling-output', dest=p2+'pooling_output',
                            default='nat+prec',
                            choices=['nat+logar', 'nat+logprec',
                                     'nat+var', 'nat+prec',
                                     'mean+logar', 'mean+logprec',
                                     'mean+var', 'mean+prec'])
    
        parser.add_argument(p1+'min-var', dest=p2+'min_var', default=0.9, type=float,
                            help=('Minimum frame variance (default: %(default)s)'))
        parser.add_argument(p1+'kl-weight', dest=p2+'kl_weight', default=0, type=float,
                            help=('Weight of the KL divergence (default: %(default)s)'))


        parser.add_argument(p1+'left-context', dest=(p2+'left_context'),
                            default=0, type=int)
        parser.add_argument(p1+'right-context', dest=(p2+'right_context'),
                            default=0, type=int)
        parser.add_argument(p1+'begin-context', dest=(p2+'begin_context'),
                            default=None, type=int)
        parser.add_argument(p1+'end-context', dest=(p2+'end_context'),
                            default=None, type=int)


