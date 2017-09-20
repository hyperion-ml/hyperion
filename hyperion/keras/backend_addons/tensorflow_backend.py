import tensorflow as tf

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
