#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from scipy.stats import wishart
import h5py

from utils import *


def generate_gmm_ivec(D, K, beta, nu, rng):

    W=np.eye(D)

    w=rng.uniform(size=(K,))
    w=w/np.sum(w)

    T=np.zeros((K, D), dtype='float32')
    indx=np.arange(K)
    indx=np.zeros((K,1),'int32')
    indx[0]=8
    indx[1]=7
    indx[2]=5
    indx[3]=0
    indx[4]=9
    indx[5]=4
    indx[6]=1
    indx[7]=3
    indx[8]=2
    indx[9]=6
    for i in xrange(K):
        theta=2*np.pi*indx[i]/K
        T[i,:]=[np.cos(theta), np.sin(theta)]

    T=T.reshape(1,K*D)
    
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

    return w, mu, C, T


def generate_samples(w, mu, C, T, N_i, M, rng):

    D = mu.shape[1]
    K = mu.shape[0]

    x = np.zeros((0, D), dtype='float32')
    r = np.zeros((0,), dtype='int32')
    t = np.zeros((0,), dtype='int32')
    phi = rng.normal(size=(M, 1)).astype('float32')
    for i in xrange(M):
        rr_i = rng.multinomial(1, w, size=(N_i,))
        x_i = np.zeros((N_i, D), dtype='float32')
        r_i = np.zeros((N_i,), dtype='int32')
        t_i = i*np.ones((N_i,), dtype='int32')
        mu_i = mu+np.dot(phi[i,:], T).reshape(K, D)
        for k in xrange(K):
            C_k = C[k]
            mu_k = mu_i[k,:]
            
            idx = (rr_i[:,k]==1)
            n_k = np.sum(idx)
            x_k = rng.multivariate_normal(
                mu_k, C_k, size=(n_k,)).astype('float32')
            r_i[idx] = k
            x_i[idx,:] = x_k
        x = np.vstack((x, x_i))
        r = np.hstack((r, r_i))
        t = np.hstack((t, t_i))

    return x, r, t


if __name__ == "__main__":

    rng=np.random.RandomState(256)

    w, mu, C, T = generate_gmm_ivec(D=2, K=10, beta=0.01, nu=10, rng=rng)

    N_i = 1000
    M = 50
    xx = []
    rr = []
    tt = []
    for s in xrange(3):
        x, r, t = generate_samples(w, mu, C, T, N_i, M, rng)
        xx.append(x)
        rr.append(r)
        tt.append(t)

    xx = normalize_x(xx)

        
    f=h5py.File('data.h5','w')
    f.create_dataset('w', data=w)
    f.create_dataset('mu', data=mu)
    f.create_dataset('C', data=C)
    f.create_dataset('T', data=T)
    f.create_dataset('K', data=len(C))
    f.create_dataset('M', data=M)
    save_xrt(f, xx, rr, tt)

    plot_xr(xx[0], rr[0], 'x_train_r.pdf')
    plot_xr(xx[0], tt[0], 'x_train_t.pdf')

    idx=np.logical_or(tt[0]==0, np.logical_or(tt[0]==1, tt[0]==4))
    plot_xr(xx[0][idx,:], tt[0][idx], 'x_train_t3.pdf')
