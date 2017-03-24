import tensorflow as tf

def matrix_inverse(x):
    return tf.matrix_inverse(x)

def cholesky(x, lower=False):
    if lower:
        return tf.cholesky(x)
    return tf.cholesky(x).T
