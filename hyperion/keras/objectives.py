
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from keras import backend as K
from . import backend_addons as K2

from ..hyp_defs import float_keras

log2pi=np.log(2*np.pi).astype(float_keras())
one=np.exp(0).astype(float_keras())


def bernoulli(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)


def diag_normal(y_true, y_pred):
    y_mean, y_logvar = y_pred
    y_var = K.exp(y_logvar)
    dist2=K.square(y_true-y_mean)/y_var
    return 0.5*K.sum(log2pi+y_logvar+dist2, axis=-1)


def normal_1chol(y_true, y_pred):
    y_mean, y_logvar, y_chol = y_pred

    isigma = K.exp(-y_logvar/2)
    iy_chol = K2.matrix_inverse(y_chol)
    
    distn  = K.dot(y_true-y_mean, iy_chol) * isigma
    dist2 = K.square(distn)
    return 0.5*K.sum(log2pi+y_logvar+dist2, axis=-1)


def normal(y_true, y_pred):
    y_mean, y_logvar, y_chol = y_pred

    isigma = K.exp(-y_logvar/2)
    f = lambda x: K2.matrix_inverse(x)
    iy_chol = K.map_fn(f, y_chol)
    
    dist  = y_true-y_mean
    distn = K.batch_dot(dist, iy_chol, axes=(1, 1)) * isigma
    dist2 = K.square(distn)
    return 0.5*K.sum(log2pi+y_logvar+dist2, axis=-1)


def normal_3d(y_true, y_pred):
    y_mean, y_logvar, y_chol = y_pred

    x_dim = K.cast(K.shape(y_mean)[-1], 'int32')
    seq_length = K.cast(K.shape(y_mean)[-2], 'int32')

    y_chol = K.reshape(y_chol, (-1, x_dim, x_dim))
    
    isigma = K.exp(-y_logvar/2)
    f = lambda x: K2.matrix_inverse(x)
    iy_chol = K.map_fn(f, y_chol)
    
    dist = y_true-y_mean
    dist = K.reshape(dist, (-1, x_dim))
    distn = K.batch_dot(dist, iy_chol, axes=(1, 1))
    distn = K.reshape(distn, (-1, seq_length, x_dim)) * isigma
    dist2 = K.square(distn)
    return 0.5*K.sum(log2pi+y_logvar+dist2, axis=-1)


def normal_1chol_3d(y_true, y_pred):
    y_mean, y_logvar, y_chol = y_pred

    x_dim = K.cast(K.shape(y_mean)[-1], 'int32')
    seq_length = K.cast(K.shape(y_mean)[-2], 'int32')
    
    isigma = K.exp(-y_logvar/2)
    iy_chol = K2.matrix_inverse(y_chol) 

    dist = y_true-y_mean
    #dist = K.reshape(dist, (-1, x_dim))
    distn  = K.dot(dist, iy_chol) * isigma
    #distn = K.reshape(distn, (-1, seq_length, x_dim)) 
    dist2 = K.square(distn)
    return 0.5*K.sum(log2pi+y_logvar+dist2, axis=-1)


def kl_normal_vs_std_normal(qy, beta=one):
    y_mean, y_logvar = qy[:2]
    T = one/beta
    return -0.5*K.sum((T-one) * log2pi + T + T * y_logvar
                      - K.square(y_mean) - K.exp(y_logvar), axis=-1)


def kl_normal_vs_diag_normal(qy1, qy2, beta=one):
    y1_mean, y1_logvar = qy1[:2]
    y2_mean, y2_logvar = qy2[:2]
    T = one/beta
    v1 = K.exp(y1_logvar)
    v2 = K.exp(y2_logvar)
    return -0.5*K.sum((T-one) * log2pi + T + T * y1_logvar - y2_logvar
                      - (K.square(y1_mean-y2_mean)+v1)/v2, axis=-1)


def kl_normal_commonvar2(qy1, qy2, beta=one):
    y1_mean, y1_logvar, y1_chol = qy1
    y2_mean, y2_logvar, y2_chol = qy2

    dim=K.shape(y1_mean)[-1]

    iy2_chol = K2.matrix_inverse(y2_chol)
    iy2_chol*= iy_chol * K.exp(-y2_logvar/2)
    
    dist  = y1_mean-y2_mean
    distn = K.batch_dot(dist, iy2_chol, axes=(2, 0))
    dist2 = K.square(distn)

    iy2_var = K.reshape(np.dot(iy2_chol, iy2_chol.T), shape=(-1))
    y1_chol *= K.exp(y1_logvar/2)
    f = lambda x: K.reshape(K.dot(x.T, x), shape=(-1, dim**2))
    y1_var = K.map_fh(f, y1_chol)

    tr = K.dot(y1_var, iy2_var)
    
    T = one/beta
    v1 = K.exp(y1_logvar)
    v2 = K.exp(y2_logvar)
    return -0.5*K.sum((T-1)*log2pi + T + T*y1_logvar - y2_logvar
                      - dist2 - tr, axis=-1)

