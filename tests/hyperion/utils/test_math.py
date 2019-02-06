"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import pytest
import numpy as np
import scipy.linalg as la

from hyperion.utils.math import *

def create_matrices(dim):
    x1 = np.random.randn(dim*10, dim)
    x2 = np.random.randn(dim*10, dim)
    A = np.dot(x1.T, x1)
    return A, x1, x2


def test_logdet_pdmat(dim=10):

    A, x1, x2 = create_matrices(dim)
    logA_t = np.log(la.det(A))
    # test logdet_pdmat
    logA = logdet_pdmat(A)
    assert(np.allclose(logA_t, logA))

def test_invert_pdmat_leftinv(dim=10):

    A, x1, x2 = create_matrices(dim)
    invA_t = la.inv(A)
    logA_t = np.log(la.det(A))
    invAx2_t = np.dot(invA_t, x2.T)


    # test invert_pdmat
    invA_f, RA, logA, invA = invert_pdmat(
        A, right_inv=False, return_logdet=True, return_inv=True)
    
    invAx2 = invA_f(x2.T)
    
    assert(np.allclose(logA_t, logA))
    assert(np.allclose(invA_t, invA))
    assert(np.allclose(invAx2_t, invAx2))

    
def test_invert_pdmat_rightinv(dim=10):

    A, x1, x2 = create_matrices(dim)
    invA_t = la.inv(A)
    logA_t = np.log(la.det(A))
    x2invA_t = np.dot(x2, invA_t)

    invA_f, RA, logA, invA = invert_pdmat(
        A, right_inv=True, return_logdet=True, return_inv=True)

    x2invA = invA_f(x2)
    
    assert(np.allclose(logA_t, logA))
    assert(np.allclose(invA_t, invA))
    assert(np.allclose(x2invA_t, x2invA))

    
# test invert_trimat upper triangular
def test_invert_uppertrimat_leftinv(dim=10):

    A, x1, x2 = create_matrices(dim)

    RA = invert_pdmat(A)[1]

    B=RA
    invB_t = la.inv(B)
    logB_t = np.log(la.det(B))
    invBx2_t = np.dot(invB_t, x2.T)
    
    invB_f, logB, invB = invert_trimat(
        B, lower=False, right_inv=False, return_logdet=True, return_inv=True)

    invBx2 = invB_f(x2.T)
    
    assert(np.allclose(logB_t, logB))
    assert(np.allclose(invB_t, invB))
    assert(np.allclose(invBx2_t, invBx2))

    
def test_invert_uppertrimat_rightinv(dim=10):
    
    A, x1, x2 = create_matrices(dim)

    RA = invert_pdmat(A)[1]

    B=RA
    invB_t = la.inv(B)
    logB_t = np.log(la.det(B))
    
    x2invB_t = np.dot(x2, invB_t)
    invB_f, logB, invB = invert_trimat(
        B, lower=False, right_inv=True, return_logdet=True, return_inv=True) 

    x2invB = invB_f(x2)
    
    assert(np.allclose(logB_t, logB))
    assert(np.allclose(invB_t, invB))
    assert(np.allclose(x2invB_t, x2invB))


# test invert_trimat lower triangular    
def test_invert_lowertrimat_leftinv(dim=10):
    
    A, x1, x2 = create_matrices(dim)

    RA = invert_pdmat(A)[1]

    C=RA.T
    invC_t = la.inv(C)
    logC_t = np.log(la.det(C))
    invCx2_t = np.dot(invC_t, x2.T)

    
    invC_f, logC, invC = invert_trimat(
        C, lower=True, right_inv=False, return_logdet=True, return_inv=True)

    invCx2 = invC_f(x2.T)
    
    assert(np.allclose(logC_t, logC))
    assert(np.allclose(invC_t, invC))
    assert(np.allclose(invCx2_t, invCx2))

    

def test_invert_lowertrimat_rightinv(dim=10):
    
    A, x1, x2 = create_matrices(dim)

    RA = invert_pdmat(A)[1]

    C=RA.T
    invC_t = la.inv(C)
    logC_t = np.log(la.det(C))

    x2invC_t = np.dot(x2, invC_t)
    invC_f, logC, invC = invert_trimat(
        C, lower=True, right_inv=True, return_logdet=True, return_inv=True) 

    x2invC = invC_f(x2)
    
    assert(np.allclose(logC_t, logC))
    assert(np.allclose(invC_t, invC))
    assert(np.allclose(x2invC_t, x2invC))


def test_softmax(dim=10):
    # test softmax
    rng = np.random.RandomState(seed=0)
    y_t = rng.uniform(low=0., high=1.0, size=(dim*10, dim))
    y_t /= np.sum(y_t, axis=-1, keepdims=True)

    z = np.log(y_t)+10
    y = softmax(z)
    assert(np.allclose(y_t, y))



def test_logsumexp(dim=10):
    # test softmax
    rng = np.random.RandomState(seed=0)
    y_t = rng.uniform(low=0., high=1.0, size=(dim*10, dim))
    z = np.log(y_t)
    y_t = np.log(np.sum(y_t, axis=-1)+1e-20)

    y = logsumexp(z)
    assert(np.allclose(y_t, y, rtol=1e-5))

    
# test fisher ratio
def test_fisher_ratio(dim=10):

    A = create_matrices(dim)[0]
    invA = invert_pdmat(
        A, right_inv=False, return_logdet=False, return_inv=True)[-1]
    
    mu1 = np.random.randn(dim)
    mu2 = np.random.randn(dim)
    r1 = fisher_ratio(mu1, A, mu2, A)
    r2 = fisher_ratio_with_precs(mu1, invA, mu2, invA)
    assert(np.allclose(r1, r2))


# test mat2vec conversions
def test_symmat2vec(dim=10):

    A = create_matrices(dim)[0]
    
    v = symmat2vec(A, lower=False)
    Ac = vec2symmat(v, lower=False)
    assert(np.allclose(A, Ac))

    v = symmat2vec(A, lower=True)
    Ac = vec2symmat(v, lower=True)
    assert(np.allclose(A, Ac))

    
def test_trimat2vec(dim=10):

    A = create_matrices(dim)[0]
    B = la.cholesky(A, lower=False)
    C = B.T
    
    v = trimat2vec(B, lower=False)
    Bc = vec2trimat(v, lower=False)
    assert(np.allclose(B, Bc))

    v = trimat2vec(C, lower=True)
    Cc = vec2trimat(v, lower=True)
    assert(np.allclose(C, Cc))

    
# test fullcov flooring
def test_fullcov_varfloor(dim=10):

    A = create_matrices(dim)[0]
    u, d, _= la.svd(A, full_matrices=False)
    assert(np.allclose(A, np.dot(u*d, u.T)))
    d1=d
    d1[int(dim/2):]=0.0001
    D1=np.dot(u*d1, u.T)

    F=A
    RF=la.cholesky(F)
    DF_1=fullcov_varfloor(D1, RF, F_is_chol=True, lower=False)

    RF=la.cholesky(F).T
    DF_2=fullcov_varfloor(D1, RF, F_is_chol=True, lower=True)
    assert(np.allclose(DF_1, F))
    assert(np.allclose(DF_1, F))




def test_fullcov_varfloor_from_cholS(dim=10):

    A = create_matrices(dim)[0]
    u, d, _= la.svd(A, full_matrices=False)
    assert(np.allclose(A, np.dot(u*d, u.T)))
    d1=d
    d1[int(dim/2):]=0.0001
    D1=np.dot(u*d1, u.T)

    F=A
    RD1=la.cholesky(D1)
    RF=la.cholesky(F)
    RF_1=fullcov_varfloor_from_cholS(RD1, RF, lower=False)

    RD1=la.cholesky(D1).T
    RF=la.cholesky(F).T
    RF_2=fullcov_varfloor_from_cholS(RD1, RF, lower=True)
    assert(np.allclose(RF, RF_2))
    assert(np.allclose(RF_1, RF_2.T))
    

if __name__ == '__main__':
    pytest.main([__file__])
