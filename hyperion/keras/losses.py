
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from keras import backend as K
from . import backend_addons as K2

from ..hyp_defs import float_keras

log2pi = np.log(2*np.pi).astype(float_keras())
one = np.exp(0).astype(float_keras())

from ..hyp_defs import float_keras

def get_seq_length(x):
    return K.sum(K.cast(K.any(K.not_equal(x, 0), axis=-1), K.floatx()), axis=-1)


def nllk_categorical(y_true, y_pred, time_norm=True):
    T = 1
    if not time_norm:
        T = get_seq_length(y_true)
    return T*K.categorical_crossentropy(y_true, y_pred)


def nllk_bernoulli(y_true, y_pred, time_norm=True):
    T = 1
    if not time_norm:
        T = get_seq_length(y_true)
    return T*K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)


def nllk_normal_diag_cov(y_true, y_pred, time_norm=True):
    T = 1
    if not time_norm:
        T = K.expand_dims(get_seq_length(y_true), axis=-1)
    y_dim = K.shape(y_true)[-1]
    idx_mean = [slice(None)]*(K.ndim(y_pred)-1) + [slice(0, y_dim)]
    idx_logvar = [slice(None)]*(K.ndim(y_pred)-1) + [slice(y_dim, None)]
    y_mean = y_pred[idx_mean]
    y_logvar = y_pred[idx_logvar]
    y_var = K.exp(y_logvar)
    dist2 = K.square(y_true-y_mean)/y_var
    return 0.5*T*K.sum(log2pi+y_logvar+dist2, axis=-1)

# def nllk_normal_diag_cov(y_true, y_pred, time_norm=True):
#     T = 1
#     if not time_norm:
#         T = get_seq_length(y_true)
#     idx_mean = [slice(None)]*(K.ndim(y_pred)-2) + [0]
#     idx_logvar = [slice(None)]*(K.ndim(y_pred)-2) + [1]
#     y_mean = y_pred[idx_mean]
#     y_logvar = y_pred[idx_logvar]
#     y_var = K.exp(y_logvar)
#     dist2 = K.square(y_true-y_mean)/y_var
#     return 0.5*T*K.sum(log2pi+y_logvar+dist2, axis=-1)

def categorical_mbr(y_true, y_pred):
    pos = K.sum(y_true * y_pred, axis=-1)
    return 1 - pos
