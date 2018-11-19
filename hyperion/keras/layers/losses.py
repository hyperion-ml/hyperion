
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import keras.backend as K
from keras.engine import InputSpec, Layer
from keras import initializers, regularizers, constraints

from ...hyp_defs import float_keras
from ..losses import get_seq_length

log2pi=np.log(2*np.pi).astype(float_keras())

def normal_param2_to_var():
    return {'var': lambda x: x,
            'prec': lambda x: 1/x,
            'logvar': lambda x: K.exp(x),
            'logprec': lambda x: K.exp(-x)}


def normal_param2_to_logvar():
    return {'var': lambda x: K.log(x),
            'prec': lambda x: -K.log(x),
            'logvar': lambda x: x,
            'logprec': lambda x: -x}



class KLDivNormalVsStdNormal(Layer):

    def __init__(self, in_fmt='mean+prec', min_kl=0, beta=1, time_norm=True, **kwargs):
        super(KLDivNormalVsStdNormal, self).__init__(**kwargs)
        self.in_fmt=in_fmt.split(sep='+', maxsplit=1)
        self.min_kl = min_kl # free bits
        self.beta = beta # annealing
        self.keepdims = True
        self.time_norm = time_norm
        
        self.input_spec = [InputSpec(min_ndim=2), InputSpec(min_ndim=2)]
        self.supports_masking = True

    # def build(self, input_shape):
    #     assert len(input_shape[0]) >= 2
    #     if len(input_shape[0]) == 2:
    #         self.keepdims = True
    #     else:
    #         self.keepdims = False


    def call(self, inputs):
        p1, p2 = inputs

        var = normal_param2_to_var()[self.in_fmt[1]](p2)
        logvar = normal_param2_to_logvar()[self.in_fmt[1]](p2)
        if self.in_fmt[0] == 'nat':
            mu = var*p1
        else:
            mu = p1

        beta = K.cast(self.beta, float_keras())
        T = 1
        keepdims = False
        # if K.ndim(mu) == 2:
        keepdims = True
        if K.ndim(mu) == 3:
            if not self.time_norm:
                T = K.expand_dims(get_seq_length(mu), axis=-1)
                T = K.expand_dims(T, axis=-1)
        kl_div = 1/(2*beta)*K.sum((beta-1)*log2pi - logvar - 1 + beta*(K.square(mu) + var), axis=-1, keepdims=keepdims)
        return T*K.clip(kl_div, self.min_kl, None) 


    def compute_output_shape(self, input_shape):
        assert input_shape
        # if len(input_shape) == 2:
        #     return (input_shape[0][0], 1)
        
        output_shape = list(input_shape[0])[:-1] + [1]
        return tuple(output_shape)

        assert input_shape and len(input_shape) == 2
        output_shape = (input_shape[0][0], 1)
        return output_shape


    def get_config(self):
        config = {'in_fmt': '+'.join(self.in_fmt),
                  'min_kl': self.min_kl,
                  'beta': self.beta}
        base_config =super(KLDivNormalVsStdNormal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



    
class KLDivNormalVsDiagNormal(Layer):

    def __init__(self, in_fmt='mean+prec', min_kl=0, beta=1, **kwargs):
        super(KLDivNormalVsDiagNormal, self).__init__(**kwargs)
        self.in_fmt=in_fmt.split(sep='+', maxsplit=1)
        self.min_kl = min_kl # free bits
        self.beta = beta # annealing
        
        self.input_spec = [InputSpec(min_ndim=2), InputSpec(min_ndim=2)]
        self.supports_masking = True
    

    def call(self, inputs):
        p11, p12, p21, p22 = inputs

        var1 = normal_param2_to_var()[self.in_fmt[1]](p12)
        logvar1 = normal_param2_to_logvar()[self.in_fmt[1]](p12)
        var2 = normal_param2_to_var()[self.in_fmt[1]](p22)
        logvar2 = normal_param2_to_logvar()[self.in_fmt[1]](p22)

        if self.in_fmt[0] == 'nat':
            mu1 = p11*var1
            mu2 = p21*var2
        else:
            mu1 = p11
            mu2 = p21

        beta = K.cast(self.beta, float_keras())
        kl_div = 1/(2*beta)*K.sum((beta-1)*log2pi + beta*logvar2 - logvar1 - 1 + beta*(K.square(mu1-mu2) + var1)/var2, axis=-1)
        return K.clip(kl_div, self.min_kl, None) 


    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        output_shape = list(input_shape[0])
        output_shape[-1] = self.units
        return tuple(output_shape)


    def get_config(self):
        config = {'in_fmt': '+'.join(self.in_fmt),
                  'min_kl': self.min_kl,
                  'beta': self.beta}
        base_config =super(KLDivNormalVsDiagNormal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



            

        
