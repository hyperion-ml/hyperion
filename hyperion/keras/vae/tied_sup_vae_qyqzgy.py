"""
Discriminative trained variational autoencoder 
Optimize E[logP(T|Y)|Q(Y|X)]
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
from keras.layers import Input, Concatenate, Activation, Reshape
from keras.models import Model, load_model, model_from_json

from ..losses import *
from ..keras_utils import *
from ..layers import *

from ...hyp_model import HypModel


class TiedSupVAE_QYQZgY(HypModel):

    def __init__(self, px_net, qy_net, qz_net, pt_net=None,
                 px_form='normal_diag_cov',
                 pt_form='categorical',
                 qy_form='normal_diag_cov',
                 qz_form='normal_diag_cov',
                 px_fmt='mean+logvar',
                 pt_fmt='categorical',
                 qy_pool_in_fmt='mean+logvar',
                 qy_pool_out_fmt='mean+var',
                 qz_fmt = 'mean+logvar',
                 left_context=0,
                 right_context=0,
                 begin_context=None,
                 end_context=None,
                 frame_corr_penalty=1,
                 prepool_downsampling=None,
                 qy_min_var=0.1, qz_min_var=0.01,
                 px_weight=1, pt_weight=1,
                 kl_qy_weight=1, kl_qz_weight=1, **kwargs):

        super(TiedSupVAE_QYQZgY, self).__init__(**kwargs)

        self.px_net = px_net
        self.pt_net = pt_net
        self.qy_net = qy_net
        self.qz_net = qz_net

        self.px_form = px_form
        self.pt_form = pt_form
        self.qy_form = qy_form
        self.qz_form = qz_form

        self.px_fmt = px_fmt
        self.pt_fmt = pt_fmt
        self.qy_pool_in_fmt = qy_pool_in_fmt
        self.qy_pool_out_fmt = qy_pool_out_fmt
        self.qz_fmt = qz_fmt
        
        self.qy_min_var = qy_min_var
        self.qz_min_var = qz_min_var

        self.px_weight = px_weight
        self.pt_weight = pt_weight
        self.kl_qy_weight = kl_qy_weight
        self.kl_qz_weight = kl_qz_weight

        self.frame_corr_penalty = frame_corr_penalty
        
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
        return self.qy_net.get_input_shape_at(0)[-1]

    
    
    @property
    def embed_dim(self):
        return self.qy_net.get_output_shape_at(0)[0][-1]


    @property
    def y_dim(self):
        return self.qy_net.get_output_shape_at(0)[0][-1]


    @property
    def z_dim(self):
        return self.qz_net.get_output_shape_at(0)[0][-1]


    
    @property
    def in_length(self):
        if self.max_seq_length is None:
            return self.qy_net.get_input_shape_at(0)[-2]
        return self.max_seq_length


    
    @property
    def pool_in_length(self):
        pool_length = self.qy_net.get_output_shape_at(0)[0][-2]
        if pool_length is None:
            in_length = self.in_length
            if in_length is None:
                return None
            x = Input(shape=(in_length, self.x_dim))
            net = Model(x, self.qy_net(x))
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



    def freeze_embed(self):
        self.qy_net.trainable = False


        
    def freeze_embed_layers(self, layers):
        for layer_name in layers:
            self.qy_net.get_layer(layer_name).trainable = False

            
        
    def build(self, max_seq_length=None):

        if max_seq_length is None:
            max_seq_length = self.qy_net.get_input_shape_at(0)[-2]
        self.max_seq_length = max_seq_length

        x = Input(shape=(max_seq_length, self.x_dim,))
        mask = CreateMask(0)(x)
        qy_x = self.qy_net(x)

        dec_ratio = int(max_seq_length/qy_x[0]._keras_shape[1])
        if dec_ratio > 1:
            mask = MaxPooling1D(dec_ratio, padding='same')(mask)
        
        qy = GlobalDiagNormalPostStdPriorPooling1D(
            in_fmt=self.qy_pool_in_fmt,
            out_fmt=self.qy_pool_out_fmt,
            min_var=self.qy_min_var, frame_corr_penalty=self.frame_corr_penalty,
            name='qy_pooling')(qy_x + [mask])


        y_seq = TiedNormalDiagCovSampler(
            max_seq_length, in_fmt=self.qy_pool_out_fmt,
            name='qy_sampling_1')(qy)

        qz = self.qz_net([x, y_seq])

        qz = ConvertNormalDiagCovPostStdPriorFmt(in_fmt=self.qz_fmt,
                                                 out_fmt='mean+logvar',
                                                 min_var=self.qz_min_var)(qz)
        z = NormalDiagCovSampler(in_fmt='mean+logvar', name='qz_sampling')(qz)
        
        px = self.px_net([y_seq, z])
        # if self.px_form == 'normal_diag_cov':
        #     px_1 = Reshape((max_seq_length, 1, self.x_dim))(px[0])
        #     px_2 = Reshape((max_seq_length, 1, self.x_dim))(px[1])
        #     px = Concatenate(axis=-2, 'px')([px_1, px_2])

        if self.px_form == 'normal_diag_cov':
            px = Concatenate(axis=-1, name='px_cat')([px[0], px[1]])
            
        kl_qy = KLDivNormalVsStdNormal(in_fmt=self.qy_pool_out_fmt, name='kl_qy')(qy)
        kl_qz = KLDivNormalVsStdNormal(in_fmt='mean+logvar', name='kl_qz',
                                       time_norm=False)(qz)
        logging.debug(K.ndim(kl_qz))
        logging.debug(K.int_shape(px))

        logging.debug(K.int_shape(kl_qy))
        logging.debug(K.int_shape(kl_qz))
        if self.pt_net is None:
            self.model = Model(x, [px, kl_qy, kl_qz])
        else:
            y = NormalDiagCovSampler(in_fmt=self.qy_pool_out_fmt, name='qy_sampling_2')(qy)
            pt = self.pt_net(y)
            self.model = Model(x, [px, pt, kl_qy, kl_qz])
        logging.debug(K.int_shape(self.model.outputs[-1]))
        logging.debug(K.int_shape(pt))
        self.model.summary()


        
    def compile(self, **kwargs):

        kl_loss=(lambda y_true, y_pred: y_pred)
        loss_dict = {'categorical': nllk_categorical,
                     'bernoulli': nllk_bernoulli,
                     'normal_diag_cov': nllk_normal_diag_cov}

        # if self.px_form == 'normal_diag_cov':
        #     metrics = {'px_cat': 'mean_squared_error'}
        # else:
        #     metrics = {}
        metrics = None
        sample_weight_mode={'px_cat': 'temporal', 'kl_qy': None, 'kl_qz': 'temporal'}
        loss = [lambda x,y: loss_dict[self.px_form](x,y,False)]
        loss_w = [self.px_weight]
        if self.pt_net is not None:
            loss.append(loss_dict[self.pt_form])
            loss_w.append(self.pt_weight)
            sample_weight_mode['pt'] = None
            metrics = {'pt':'accuracy'}
            #metrics['pt'] = 'accuracy'
            
        loss += 2*[kl_loss]
        loss_w += [self.kl_qy_weight, self.kl_qz_weight]


        self.model.compile(loss=loss, loss_weights=loss_w,
                           metrics=metrics, weighted_metrics=metrics,
                           sample_weight_mode=sample_weight_mode, **kwargs)

        
        
    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)


    
    def build_qy(self, qy_pool_out_fmt=None):
        if  qy_pool_out_fmt is None:
            qy_pool_out_fmt = self.qy_pool_out_fmt
        p1_frame_qy = Input(shape=(None, self.y_dim,))
        p2_frame_qy = Input(shape=(None, self.y_dim,))
        mask = Input(shape=(None,))
        qy = GlobalDiagNormalPostStdPriorPooling1D(
            in_fmt=self.qy_pool_in_fmt,
            out_fmt=qy_pool_out_fmt,
            min_var=self.qy_min_var, name='pooling')(
                [p1_frame_qy, p2_frame_qy, mask])
        
        self.pool_net = Model([p1_frame_qy, p2_frame_qy, mask], qy)
        self.pool_net.summary()
        

    
    def compute_qy(self, x, **kwargs):
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
        
        p1 = np.zeros((pool_length, self.y_dim), dtype=float_keras())
        p2 = np.zeros((pool_length, self.y_dim), dtype=float_keras())
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
            p1_i, p2_i = self.qy_net.predict(x_i, batch_size=1, **kwargs)
            p1[j_out:k_out] = p1_i[0,:l_out]
            p2[j_out:k_out] = p2_i[0,:l_out]
            
            j_in += in_shift
            j_out += pool_shift
            if i==0:
                j_out += pool_left_context

        p1 = np.expand_dims(p1, axis=0)
        p2 = np.expand_dims(p2, axis=0)
        qy = self.pool_net.predict([p1, p2, mask], batch_size=1, **kwargs)
        return qy



    def build_pt(self):
        p1_frame_qy = Input(shape=(None, self.y_dim,))
        p2_frame_qy = Input(shape=(None, self.y_dim,))
        mask = Input(shape=(None,))
        qy = GlobalDiagNormalPostStdPriorPooling1D(
            in_fmt=self.qy_pool_in_fmt,
            out_fmt=self.qy_pool_out_fmt,
            min_var=self.qy_min_var, name='pooling')(
                [p1_frame_qy, p2_frame_qy, mask])
        y = NormalDiagCovSampler(in_fmt=self.qy_pool_out_fmt, name='qy_sampling')(qy)
        pt = self.pt_net(y)
        self.pool_net = Model([p1_frame_qy, p2_frame_qy, mask], pt)
        self.pool_net.summary()



    def predict_pt(self, x, **kwargs):
        return self.predict_qy(x, **kwargs)


    
    def fit(self, x, t=None, x_val=None, t_val=None,  **kwargs):
        y_kl = np.zeros((x.shape[0], 1), dtype=float_keras())
        z_kl = np.zeros((x.shape[0], x.shape[1], 1), dtype=float_keras())
        if t is None:
            y = [x, y_kl, z_kl]
        else:
            y = [x, t, y_kl, z_kl]

        validation_data = None
        if x_val is not None:
            y_kl = np.zeros((x_val.shape[0], 1), dtype=float_keras())
            z_kl = np.zeros((x_val.shape[0], x_val.shape[1], 1), dtype=float_keras())
            if t_val is None:
                validation_data = (x_val, [x_val, y_kl, z_kl])
            else:
                validation_data = (x_val, [x_val, t_val, y_kl, z_kl])
        
        self.model.fit(x, y, validation_data=validation_data, **kwargs)


        
    def fit_generator(self, generator, steps_per_epoch, **kwargs):
        self.model.fit_generator(generator, steps_per_epoch, **kwargs)



    def log_prob(self, x):
        raise NotImplementedError



    def compute_px_given_yz(self, y, z, batch_size):
        return self.px_net.predict([y, z], batch_size=batch_size)


    def sample_given_yz(self, y, z, batch_size):
        if y.ndim == 2:
            y=np.expand_dims(y, axis=1)
        if y.shape[1]==1:
            y=np.tile(y, (1, z.shape[1],1))
                
        y_input = Input(shape=(self.max_seq_length, self.y_dim,))
        z_input = Input(shape=(self.max_seq_length, self.z_dim,))

        px = self.px_net([y_input, z_input, r_input])
        if self.px_form == 'bernoulli' :
            x_sample = BernoulliSampler(name='x_sampler')(px)
        elif self.px_form == 'normal_diag_cov' :
            x_sample = NormalDiagCovSampler(in_fmt='mean+logvar', name='x_sampler')(px)
        else:    
            raise ValueError()

        generator = Model([y_input, z_input], x_sample)
        return generator.predict([y, z], batch_size=batch_size)


    def sample(self, num_seqs, seq_length, batch_size):
        y = np.random.normal(loc=0.,scale=1.,size=(num_seqs, 1, self.y_dim))
        z = np.random.normal(loc=0.,scale=1.,size=(num_seqs, num_samples, self.z_dim))
        return self.sample_given_yz(y, z, batch_size)


    def sample_given_y(self, y, seq_length, batch_size):
        num_seqs=y.shape[0]
        z=np.random.normal(loc=0.,scale=1., size=(num_seqs, seq_length, self.z_dim))
        return self.sample_given_yz(y, z, batch_size)                                                            

    
    def get_config(self):
        config = { 'px_form': self.px_form,
                   'pt_form': self.pt_form,
                   'qy_form': self.qy_form,
                   'qz_form': self.qz_form,
                   'px_fmt': self.px_fmt,
                   'pt_fmt': self.pt_fmt,
                   'qy_pool_in_fmt': self.qy_pool_in_fmt,
                   'qy_pool_out_fmt': self.qy_pool_out_fmt,
                   'qz_fmt': self.qz_fmt,
                   'qy_min_var': self.qy_min_var,
                   'qz_min_var': self.qz_min_var,
                   'px_weight': self.px_weight,
                   'pt_weight': self.pt_weight,
                   'kl_qy_weight': self.kl_qy_weight,
                   'kl_qz_weight': self.kl_qz_weight,
                   'frame_corr_penalty': self.frame_corr_penalty,
                   'left_context': self.left_context,
                   'right_context': self.right_context,
                   'begin_context': self.begin_context,
                   'end_context': self.end_context}

        base_config = super(TiedSupVAE_QYQZgY, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    
    def save(self, file_path):
        file_model = '%s.json' % (file_path)
        with open(file_model, 'w') as f:
            f.write(self.to_json())
        
        file_model = '%s.qy.h5' % (file_path)
        self.qy_net.save(file_model)
        file_model = '%s.qz.h5' % (file_path)
        self.qz_net.save(file_model)
        file_model = '%s.px.h5' % (file_path)
        self.px_net.save(file_model)
        if self.pt_net is not None:
            file_model = '%s.pt.h5' % (file_path)
            self.pt_net.save(file_model)

        

    @classmethod
    def load(cls, file_path):
        file_config = '%s.json' % (file_path)
        with open(file_config,'r') as f:
            config=TiedSupVAE_QYQZgY.load_config_from_json(f.read())

        file_model = '%s.qy.h5' % (file_path)
        qy_net = load_model(file_model, custom_objects=get_keras_custom_obj())
        file_model = '%s.qz.h5' % (file_path)
        qz_net = load_model(file_model, custom_objects=get_keras_custom_obj())
        file_model = '%s.px.h5' % (file_path)
        px_net = load_model(file_model, custom_objects=get_keras_custom_obj())
        file_model = '%s.pt.h5' % (file_path)
        if os.path.isfile(file_model):
            pt_net = load_model(file_model, custom_objects=get_keras_custom_obj())
        else:
            pt_net = None

        
        filter_args = ('px_form', 'pt_form', 'qy_form', 'qz_form',
                       'px_fmt', 'qy_pool_in_fmt', 'qy_pool_out_fmt',
                       'qz_fmt',
                       'left_context', 'right_context',
                       'begin_context', 'end_context',
                       'qy_min_var', 'qz_min_var',
                       'px_weight', 'pt_weight',
                       'kl_qy_weight', 'kl_qz_weight',
                       'frame_corr_penalty', 'name')
        
        kwargs = {k: config[k] for k in filter_args if k in config }
        return cls(px_net, qy_net, qz_net, pt_net, **kwargs)

    
    
    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('px_form', 'pt_form', 'qy_form', 'qz_form',
                      'px_fmt', 'qy_pool_in_fmt', 'qy_pool_out_fmt',
                      'qz_fmt',
                      'left_context', 'right_context',
                      'begin_context', 'end_context',
                      'qy_min_var', 'qz_min_var',
                      'px_weight', 'pt_weight',
                      'kl_qy_weight', 'kl_qz_weight',
                      'frame_corr_penalty', 'name')
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


        parser.add_argument(p1+'px-form', dest=p2+'px_form', default='normal_diag_cov',
                            choices=['normal_diag_cov', 'categorical', 'bernoulli'])
        parser.add_argument(p1+'pt-form', dest=p2+'pt_form', default='categorical',
                            choices=['categorical', 'bernoulli'])
        parser.add_argument(p1+'qy-form', dest=p2+'qy_form', default='normal_diag_cov',
                            choices=['normal_diag_cov'])
        parser.add_argument(p1+'qz-form', dest=p2+'qz_form', default='normal_diag_cov',
                            choices=['normal_diag_cov'])


        choices_fmt_in = ['nat+logitvar', 'nat+logprec-1',
                          'nat+logvar', 'nat+logprec',
                          'nat+var', 'nat+prec', 'nat+prec-1',
                          'mean+logitvar', 'mean+logprec-1',
                          'mean+logvar', 'mean+logprec',
                          'mean+var', 'mean+prec', 'mean+prec-1']

        choices_fmt_out = ['nat+logvar', 'nat+logprec',
                           'nat+var', 'nat+prec',
                           'mean+logvar', 'mean+logprec',
                           'mean+var', 'mean+prec']

        parser.add_argument(p1+'px-fmt', dest=p2+'px_fmt',
                            default='mean+logvar',
                            choices=['mean+logvar'])
        
        parser.add_argument(p1+'qy-pool-in-fmt', dest=p2+'qy_pool_in_fmt',
                            default='nat+logitvar',
                            choices=choices_fmt_in)
        parser.add_argument(p1+'qy-pool-out-fmt', dest=p2+'qy_pool_out_fmt',
                            default='mean+var',
                            choices=choices_fmt_out)
        parser.add_argument(p1+'qz-fmt', dest=p2+'qz_fmt',
                            default='nat+logitvar',
                            choices=choices_fmt_in)
    
        parser.add_argument(p1+'qy-min-var', dest=p2+'qy_min_var', default=0.9, type=float,
                            help=('Minimum frame variance (default: %(default)s)'))
        parser.add_argument(p1+'qz-min-var', dest=p2+'qz_min_var', default=0.1, type=float,
                            help=('Minimum frame variance (default: %(default)s)'))

        parser.add_argument(p1+'pt-weight', dest=p2+'pt_weight', default=1, type=float,
                            help=('Weight of the E[logP(T|Y)] (default: %(default)s)'))
        parser.add_argument(p1+'px-weight', dest=p2+'px_weight', default=1, type=float,
                            help=('Weight of the E[logP(X|Y,Z)] (default: %(default)s)'))

        parser.add_argument(p1+'kl-qy-weight', dest=p2+'kl_qy_weight', default=1, type=float,
                            help=('Weight of KL(Q(Y)||P(Y)) (default: %(default)s)'))
        parser.add_argument(p1+'kl-qz-weight', dest=p2+'kl_qz_weight', default=1, type=float,
                            help=('Weight of KL(Q(Z)||P(Z)) (default: %(default)s)'))

        parser.add_argument(p1+'frame-corr-penalty', dest=p2+'frame_corr_penalty', default=1, type=float,
                            help=('Scale to account for inter-frame dependency (default: %(default)s)'))

        parser.add_argument(p1+'left-context', dest=(p2+'left_context'),
                            default=0, type=int)
        parser.add_argument(p1+'right-context', dest=(p2+'right_context'),
                            default=0, type=int)
        parser.add_argument(p1+'begin-context', dest=(p2+'begin_context'),
                            default=None, type=int)
        parser.add_argument(p1+'end-context', dest=(p2+'end_context'),
                            default=None, type=int)


