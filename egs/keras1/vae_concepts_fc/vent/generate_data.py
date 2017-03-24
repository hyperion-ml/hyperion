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


def generate_samples(w, d, theta, C, N, rng):

    K = len(w)
    r = rng.multinomial(1, w, size=(N,))
    x = np.zeros((N,D), dtype='float32')
    y = np.zeros((N,), dtype='int32')

    for k in xrange(K):
        theta_k = theta*k
        idx = (r[:,k]==1)
        n_k = np.sum(idx)
        x_k = rng.multivariate_normal([0,0], C, size=(n_k,))
        x_k = f(x_k, 3, 2*np.pi/50)
        x_k = polar_sum(x_k, d, theta_k)
        y[idx] = k
        x[idx,:] = x_k

    return x, y
        
if __name__ == "__main__":

    rng = np.random.RandomState(256)

    D = 2
    K = 8
    theta = 2*np.pi/K

    C = np.diag([100,1])
    d = 40

    N = 10000

    w = rng.uniform(size=(K,))
    w = w/np.sum(w)

    xx = []
    rr = []
    for s in xrange(3):
        x, r = generate_samples(w, d, theta, C, N, rng)
        xx.append(x)
        rr.append(r)

    xx = normalize_x(xx)

        
    f=h5py.File('data.h5','w')
    f.create_dataset('K', data=K)
    save_xr(f, xx, rr)

    plot_xr(xx[0], rr[0], 'x_train.pdf')
