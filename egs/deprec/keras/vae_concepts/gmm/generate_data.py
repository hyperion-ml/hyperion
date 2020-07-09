#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from scipy.stats import wishart
import h5py

from utils import *


def generate_gmm(D, K, beta, nu, rng):

    W=np.eye(D)

    w=rng.uniform(size=(K,))
    w=w/np.sum(w)

    mu0=np.zeros((D,),dtype='float32')
    mu=np.zeros((K,D),dtype='float32')
    Lambda=[]
    C=[]
    for k in xrange(K):
        Lambda_k=wishart.rvs(nu,W,random_state=rng)
        C_k=np.linalg.inv(Lambda_k)
        mu_k=rng.multivariate_normal(mu0,C_k/beta)

        mu[k,:]=mu_k
        Lambda.append(Lambda_k)
        C.append(C_k)

    return w, mu, C


def generate_samples(w, mu, C, N, rng):

    D = mu.shape[1]
    K = mu.shape[0]
    r=rng.multinomial(1,w,size=(N,))
    x=np.zeros((N,D),dtype='float32')
    y=np.zeros((N,),dtype='int32')
    for k in xrange(K):
        C_k=C[k]
        mu_k=mu[k,:]
        
        idx=(r[:,k]==1)
        n_k=np.sum(idx)
        x_k=rng.multivariate_normal(mu_k,C_k,size=(n_k,)).astype('float32')
        y[idx]=k
        x[idx,:]=x_k
    return x, y
        
if __name__ == "__main__":

    rng=np.random.RandomState(256)

    w, mu, C = generate_gmm(D=2, K=10, beta=0.01, nu=10, rng=rng)

    N = 10000
    xx = []
    rr = []
    for s in xrange(3):
        x, r = generate_samples(w, mu, C, N, rng)
        xx.append(x)
        rr.append(r)

    xx = normalize_x(xx)

        
    f=h5py.File('data.h5','w')
    f.create_dataset('w', data=w)
    f.create_dataset('mu', data=mu)
    f.create_dataset('C', data=C)
    f.create_dataset('K', data=len(C))
    save_xr(f, xx, rr)

    plot_xr(xx[0], rr[0], 'x_train.pdf')
