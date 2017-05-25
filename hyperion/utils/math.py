"""
Some math functions.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np
import scipy.linalg as la

def logdet_pdmat(A):
    assert(A.shape[0] == A.shape[1])
    R=la.cholesky(A)
    return 2*np.sum(np.log(np.diag(R)))


def invert_pdmat(A, right_inv=False, return_logdet=False, return_inv=False):
    assert(A.shape[0] == A.shape[1])
    R=la.cholesky(A, lower=False)

    if right_inv:
        fh=lambda x: la.cho_solve((R, False), x.T).T
    else:
        fh=lambda x: la.cho_solve((R, False), x)
        #fh=lambda x: la.solve_triangular(R, la.solve_triangular(R.T, x, lower=True), lower=False)

    r = [fh, R]
        
    logdet = None
    invA = None

    if return_logdet:
        logdet=2*np.sum(np.log(np.diag(R)))
        r.append(logdet)

    if return_inv:
        invA=fh(np.eye(A.shape[0]))
        r.append(invA)

    return r
    #return fh, R, logdet, invA


def invert_trimat(A, lower=False, right_inv=False, return_logdet=False, return_inv=False):
    if right_inv:
        fh=lambda x: la.solve_triangular(A.T, x.T, lower=not(lower)).T
    else:
        fh=lambda x: la.solve_triangular(A, x, lower=lower)
    logdet = None
    invA = None
    if return_logdet:
        logdet=np.sum(np.log(np.diag(A)))

    if return_inv:
        invA=fh(np.eye(A.shape[0]))

    return fh, logdet, invA


def softmax(r):
    max_r=np.max(r, axis=1, keepdims=True)
    r=np.exp(r-max_r)
    r/=np.sum(r, axis=1, keepdims=True)
    return r


def fisher_ratio(mu1, Sigma1, mu2, Sigma2):
    S=Sigma1+Sigma2
    L=invert_pdmat(S)[0]
    delta=mu1-mu2
    return np.inner(delta, L(delta))


def fisher_ratio_with_precs(mu1, Lambda1, mu2, Lambda2):
    Sigma1 = invert_pdmat(Lambda1, return_inv=True)[-1]
    Sigma2 = invert_pdmat(Lambda2, return_inv=True)[-1]
    return fisher_ratio(mu1, Sigma1, mu2, Sigma2)


def symmat2vec(A, lower=False, diag_factor=None):
    if diag_factor is not None:
        A = np.copy(A)
        A[np.diag_indices(A.shape[0])] *= diag_factor
    if lower:
         return A[np.tril_indices(A.shape[0])]
    return A[np.triu_indices(A.shape[0])]


def vec2symmat(v, lower=False, diag_factor=None):
    dim=int((-1+np.sqrt(1+8*v.shape[0]))/2)
    idx_u=np.triu_indices(dim)
    idx_l=np.tril_indices(dim)
    A=np.zeros((dim,dim))
    if lower:
        A[idx_l]=v
        A[idx_u]=A.T[idx_u]
        return A
    A[idx_u]=v
    A[idx_l]=A.T[idx_l]
    if diag_factor is not None:
        A[np.diag_indices(A.shape[0])] *= diag_factor
    return A


def trimat2vec(A, lower=False):
    return symmat2vec(A, lower)


def vec2trimat(v, lower=False):
    dim=int((-1+np.sqrt(1+8*v.shape[0]))/2)
    A=np.zeros((dim,dim))
    if lower:
        A[np.tril_indices(dim)]=v
        return A
    A[np.triu_indices(dim)]=v
    return A


def fullcov_varfloor(cholS, cholF, lower=False):
    if isinstance(cholF, np.ndarray):
        if lower:
            cholS = cholS.T
            cholF = cholF.T
        T = np.dot(cholS, invert_trimat(cholF, return_inv=True)[2])
    else:
        if lower:
            cholS=cholS.T
        T = cholS/cholF
    T = np.dot(T.T,T)
    u, d, _ = la.svd(T, full_matrices=False, overwrite_a=True)
    d[d<1.]=1
    T = np.dot(u*d, u.T)
    if isinstance(cholF, np.ndarray):
        S = np.dot(cholF.T, np.dot(T, cholF))
    else:
        S = (cholF**2)*T
    return la.cholesky(S, lower)


# def test_math(dim):
    
#     x1 = np.random.randn(dim*10, dim)
#     x2 = np.random.randn(dim*10, dim)
#     A = np.dot(x1.T, x1)
#     invA_t = la.inv(A)
#     logA_t = np.log(la.det(A))
#     invAx2_t = np.dot(invA_t, x2.T)
#     x2invA_t = np.dot(x2, invA_t)

#     # test logdet_pdmat
#     logA = logdet_pdmat(A)
#     assert(np.allclose(logA_t, logA))

#     # test invert_pdmat
#     invA_f, RA, logA, invA = invert_pdmat(
#         A, right_inv=False, return_logdet=True, return_inv=True)

#     invAx2 = invA_f(x2.T)
    
#     assert(np.allclose(logA_t, logA))
#     assert(np.allclose(invA_t, invA))
#     assert(np.allclose(invAx2_t, invAx2))


#     invA_f, RA, logA, invA = invert_pdmat(
#         A, right_inv=True, return_logdet=True, return_inv=True)

#     x2invA = invA_f(x2)
    
#     assert(np.allclose(logA_t, logA))
#     assert(np.allclose(invA_t, invA))
#     assert(np.allclose(x2invA_t, x2invA))


#     # test invert_trimat upper triangular
#     B=RA
#     invB_t = la.inv(B)
#     logB_t = np.log(la.det(B))
#     invBx2_t = np.dot(invB_t, x2.T)
#     x2invB_t = np.dot(x2, invB_t)
    
#     invB_f, logB, invB = invert_trimat(
#         B, lower=False, right_inv=False, return_logdet=True, return_inv=True)

#     invBx2 = invB_f(x2.T)
    
#     assert(np.allclose(logB_t, logB))
#     assert(np.allclose(invB_t, invB))
#     assert(np.allclose(invBx2_t, invBx2))


#     invB_f, logB, invB = invert_trimat(
#         B, lower=False, right_inv=True, return_logdet=True, return_inv=True) 

#     x2invB = invB_f(x2)
    
#     assert(np.allclose(logB_t, logB))
#     assert(np.allclose(invB_t, invB))
#     assert(np.allclose(x2invB_t, x2invB))


#     # test invert_trimat lower triangular
#     C=RA.T
#     invC_t = la.inv(C)
#     logC_t = np.log(la.det(C))
#     invCx2_t = np.dot(invC_t, x2.T)
#     x2invC_t = np.dot(x2, invC_t)
    
#     invC_f, logC, invC = invert_trimat(
#         C, lower=True, right_inv=False, return_logdet=True, return_inv=True)

#     invCx2 = invC_f(x2.T)
    
#     assert(np.allclose(logC_t, logC))
#     assert(np.allclose(invC_t, invC))
#     assert(np.allclose(invCx2_t, invCx2))


#     invC_f, logC, invC = invert_trimat(
#         C, lower=True, right_inv=True, return_logdet=True, return_inv=True) 

#     x2invC = invC_f(x2)
    
#     assert(np.allclose(logC_t, logC))
#     assert(np.allclose(invC_t, invC))
#     assert(np.allclose(x2invC_t, x2invC))

#     # test softmax
#     y_t = np.random.uniform(low=0., high=1.0, size=(dim*10, dim))
#     y_t/=np.sum(y_t, axis=-1, keepdims=True)

#     z = np.log(y_t)+10
#     y = softmax(z)
#     assert(np.allclose(y_t, y))

    
#     # test fisher ratio
#     mu1 = np.random.randn(dim)
#     mu2 = np.random.randn(dim)
#     r1 = fisher_ratio(mu1, A, mu2, A)
#     r2 = fisher_ratio_with_precs(mu1, invA, mu2, invA)
#     assert(np.allclose(r1, r2))

#     # test mat2vec conversions
#     v = symmat2vec(A, lower=False)
#     Ac = vec2symmat(v, lower=False)
#     assert(np.allclose(A, Ac))

#     v = symmat2vec(A, lower=True)
#     Ac = vec2symmat(v, lower=True)
#     assert(np.allclose(A, Ac))

#     v = trimat2vec(B, lower=False)
#     Bc = vec2trimat(v, lower=False)
#     assert(np.allclose(B, Bc))

#     v = trimat2vec(C, lower=True)
#     Cc = vec2trimat(v, lower=True)
#     assert(np.allclose(C, Cc))

#     # test fullcov flooring
#     u, d, _= la.svd(A, full_matrices=False)
#     assert(np.allclose(A, np.dot(u*d, u.T)))
#     d1=d
#     d1[int(dim/2):]=0.0001
#     D1=np.dot(u*d1, u.T)

#     F=A
#     RD1=la.cholesky(D1)
#     RF=la.cholesky(F)
#     RF_1=fullcov_varfloor(RD1, RF, lower=False)

#     RD1=la.cholesky(D1).T
#     RF=la.cholesky(F).T
#     RF_2=fullcov_varfloor(RD1, RF, lower=True)
#     assert(np.allclose(RF, RF_2))
#     assert(np.allclose(RF_1, RF_2.T))
    
