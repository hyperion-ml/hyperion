
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from scipy.fftpack import dct, idct
from scipy import linalg as la

import keras.backend as K
from keras.engine import InputSpec, Layer
from keras.layers import Dense
from keras import initializers

from ...hyp_defs import float_keras


class MelFB(Dense):
    def __init__(self, units, fs, f_low, f_high,
                 inv=False, normalize=False, type='kaldi',
                 trainable=False, **kwargs):
        kwargs['trainable'] = trainable
        kwargs['kernel_initializer'] = 'zeros'
        kwargs['activation'] = None
        kwargs['use_bias'] = False
        super(MelFB, self).__init__(units, **kwargs)
        self.fs = fs
        self.f_low = f_low
        self.f_high = f_high
        self.inv = inv
        self.normalize = normalize
        self.type = type

        
    @staticmethod
    def _lin2mel(x):
        return 1127.0 * np.log(1+x/700)



    @staticmethod
    def _mel2lin(x):
        return 700 * (np.exp(x/1127.0) - 1)



    @staticmethod
    def _make_melfb_kaldi(NFFT, output_dim, fs, f_low, f_high):
        fs_2 = fs/2
        mel_f_low = lin2mel(f_low)
        mel_f_high = lin2mel(f_high)
        melfc = np.linspace(mel_f_low, mel_f_high, output_dim+2)
        mels = lin2mel(np.linspace(0,fs,NFFT))

        M = np.zeros((output_dim, int(NFFT/2)))
        for k in xrange(output_dim):
            left_mel = melfc[k]
            center_mel = melfc[k+1]
            right_mel = melfc[k+2]
            for j in xrange(int(NFFT/2)):
                mel_j = mels[j]
                if mel_j > left_mel and mel_j < right_mel:
                    if mel_j <= center_mel:
                        M[k,j] = (mel_j - left_mel)/(center_mel - left_mel)
                    else:
                        M[k,j] = (right_mel - mel_j)/(right_mel - center_mel)

        return M.T



    @staticmethod
    def _make_melfb_etsi(NFFT, output_dim, fs, f_low, f_high):
        fs_2 = fs/2
        mel_f_low = lin2mel(f_low)
        mel_f_high = lin2mel(f_high)
        fc = mel2lin(np.linspace(mel_f_low, mel_f_high, output_dim+2))
        cbin = np.round(fc/fs*NFFT).astype(int)
        M = np.zeros((output_dim, int(NFFT/2)))
        for k in xrange(output_dim):
            for j in xrange(cbin[k], cbin[k+1]+1):
                M[k,j] = (j - cbin[k] + 1)/(cbin[k+1]-cbin[k]+1)
            for j in xrange(cbin[k+1]+1, cbin[k+2]+1):
                M[k,j] = (cbin[k+2] - j + 1)/(cbin[k+2]-cbin[k+1]+1)

        return M.T

        
    
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if self.type == 'etsi':
            B = self._make_melfb_etsi(input_dim, self.units,
                                      self.fs, self.f_low, self.f_high)
        else:
            B = self._make_melfb_kaldi(input_dim, self.units,
                                       self.fs, self.f_low, self.f_high)

        if self.normalize:
            B /= np.sum(B, axis=0, keepdims=True)

        if self.inv:
            if input_dim == units:
                B = la.inv(B, overwrite_a=True)
            else:
                B = la.pinv(B)
        
        self.kernel = self.add_weight((input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='dct')
        self._initial_weights = [B]
        self.input_spec = InputSpec(dtype=K.floatx(),
                                    min_ndim=2, axes={-1: input_dim})
        self.built = True


        
    def get_config(self):
        config = {'fs': self.fs,
                  'f_low': self.f_low,
                  'f_hig': self.f_high,
                  'inv': self.inv,
                  'normalize': self.normalize}
        base_config = super(MelFB, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    

class DCT(Dense):

    def __init__(self, units, inv=False, normalize=True, trainable=False, **kwargs):
        kwargs['trainable'] = trainable
        kwargs['kernel_initializer'] = 'zeros'
        kwargs['activation'] = None
        kwargs['use_bias'] = False
        super(DCT, self).__init__(units, **kwargs)
        self.inv = inv
        self.normalize = normalize

        
    @staticmethod
    def _make_dct(input_dim, output_dim, inv, normalize):
        
        if normalize:
            norm = 'ortho'
        else:
            norm = None

        if inv:
            C = idct(np.eye(input_dim), type=2, norm=norm, overwrite_x=True)
        else:
            C = dct(np.eye(input_dim), type=2, norm=norm, overwrite_x=True)
        
        return C[:,:output_dim]

    
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        C = self._make_dct(input_dim, self.units, self.inv, self.normalize)
        
        self.kernel = self.add_weight((input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='dct')
        self._initial_weights = [C]
        self.input_spec = InputSpec(dtype=K.floatx(),
                                    min_ndim=2, axes={-1: input_dim})
        self.built = True


        
    def get_config(self):
        config = {'inv': self.inv,
                  'normalize': self.normalize}
        base_config = super(DCT, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    

    
class Liftering(Layer):
    
    def __init__(self, Q=22, inv=False, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        kwargs['trainable'] = False
        super(Liftering, self).__init__(**kwargs)
        self.Q = Q
        self.inv = inv
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True


    @staticmethod
    def _make_liftering(N, Q):
        return 1 + 0.5*Q*np.sin(np.pi*np.arange(N)/Q)

        
        
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        f = self._make_liftering(input_dim, self.Q)
        if self.inv:
            f = 1/f

        self.filter = self.add_weight((input_dim,),
                                      initializer=initializers.Zeros(),
                                      name='filter')

        self._initial_weights = [f]
        self.input_spec = InputSpec(dtype=K.floatx(),
                                    min_ndim=2, axes={-1: input_dim})
        self.built = True

        
    def call(self, inputs, mask=None):
        output = inputs * self.filter
        return output

    
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        return tuple(input_shape)

    
    def get_config(self):
        config = {'Q': self.Q,
                  'inv': self.inv}
        base_config = super(Liftering, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
