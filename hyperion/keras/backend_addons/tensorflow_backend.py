import os
import logging
import subprocess
import tensorflow as tf

import keras.backend as K

def diag(x):
    return tf.diag(x)

def matrix_inverse(x):
    return tf.matrix_inverse(x)

def cholesky(x, lower=False):
    if lower:
        return tf.cholesky(x)
    return tf.cholesky(x).T

def tile(x, n):
    return tf.tile(x, n)


def max_with_mask(x, mask, axis=None, keepdims=False):
    y = x - 1e10 * (1-mask)
    return tf.reduce_max(y*mask, axis=axis, keepdims=keepdims)



def reserve_gpu():
    result = subprocess.run('free-gpu', stdout=subprocess.PIPE)
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = result.stdout.decode('utf-8')
    logging.debug(os.environ['CUDA_VISIBLE_DEVICES'])
    return K.get_session()
