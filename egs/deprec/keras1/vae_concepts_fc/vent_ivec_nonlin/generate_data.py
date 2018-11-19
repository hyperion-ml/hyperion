#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from scipy.stats import wishart
import h5py

from utils import *


def f(x,a,b):
    y = x
    y[:,1] += a*np.sin(b*x[:,0])
    return y

def polar_sum(x,r,theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]]).T
    x0 = np.array([r*np.cos(theta), r*np.sin(theta)])
    y = np.dot(x,R)+x0
    return y.astype('float32')


def generate_samples(w, d, dd, theta, C, N_i, M, rng):

    K = len(w)

    x = np.zeros((0,D), dtype='float32')
    r = np.zeros((0,), dtype='int32')
    t = np.zeros((0,), dtype='int32')
    phi = rng.normal(size=(M,1)).astype('float32')
    for i in xrange(M):
        rr_i = rng.multinomial(1, w,size=(N_i,))
        x_i = np.zeros((N_i, D), dtype='float32')
        r_i = np.zeros((N_i,), dtype='int32')
        t_i = i*np.ones((N_i,), dtype='int32')
        d_i = d+dd*phi[i,0]
        theta0=theta/4*phi[i,0]
        for k in xrange(K):
            theta_k = theta0+theta*k
            idx = (rr_i[:,k]==1)
            n_k = np.sum(idx)
            x_k = rng.multivariate_normal([0,0], C, size=(n_k,))
            x_k = f(x_k, 3, 2*np.pi/50)
            x_k = polar_sum(x_k, d_i, theta_k)
            r_i[idx] = k
            x_i[idx,:] = x_k
        x = np.vstack((x, x_i))
        r = np.hstack((r, r_i))
        t = np.hstack((t, t_i))

    return x, r, t


if __name__ == "__main__":

    rng = np.random.RandomState(256)

    D = 2
    K = 8
    theta = 2*np.pi/K

    C = np.diag([100,1])
    d=120
    dd=40

    N_i = 1000
    M = 50
    
    w = rng.uniform(size=(K,))
    w = w/np.sum(w)

    xx = []
    rr = []
    tt = []
    for s in xrange(3):
        x, r, t = generate_samples(w, d, dd, theta, C, N_i, M, rng)
        xx.append(x)
        rr.append(r)
        tt.append(t)

    xx = normalize_x(xx)

        
    f=h5py.File('data.h5','w')
    f.create_dataset('K', data=K)
    f.create_dataset('M', data=M)
    save_xrt(f, xx, rr, tt)

    plot_xr(xx[0], rr[0], 'x_train_r.pdf')
    plot_xr(xx[0], tt[0], 'x_train_t.pdf')

    idx=np.logical_or(tt[0]==0, np.logical_or(tt[0]==1, tt[0]==4))
    plot_xr(xx[0][idx,:], tt[0][idx], 'x_train_t3.pdf')
