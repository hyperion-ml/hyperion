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

