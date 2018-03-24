import theano
from theano import tensor as T
from theano.tensor import nlinalg as nla
from theano.tensor import slinalg as sla

def diag(x):
    return nla.diag(x)

def matrix_inverse(x):
    return nla.matrix_inverse(x)

def cholesky(x, lower=False):
    return sla.Cholesky(lower)(x)

def tile(x, n):
    return T.tile(x, n)


def max_with_mask(x, mask, axis=None, keepdims=False):
    y = x - 1e10 * (1-mask)
    return T.max(y, axis=axis, keepdims=keepdims)

def reserve_gpu():
    return 0
