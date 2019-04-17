"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import os

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from numpy.testing import assert_allclose
from scipy import linalg as la

from hyperion.utils.plotting import plot_gaussian_ellipsoid_2D as pge2d
from hyperion.pdfs import FRPLDA


x_dim = 2
num_classes = 50
num_spc = 10

mu = np.array([0.5, 0])
Sb = np.array([[1, 0.05],
               [0.05, 0.1]])
Sw = np.array([[0.1, 0.05],
               [0.05, 1]])

output_dir = './tests/data_out/pdfs/plda/frplda'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def create_plda():
    B = la.inv(Sb)
    W = la.inv(Sw)
    plda = FRPLDA(mu=mu, B=B, W=W)
    return plda


def plot_plda(plda, colors=['b','g','r'], linestyle='--', label=None):

    hw = 0.05
    hl = 0.1
    fc=colors[0]
    ec=colors[0]
    ax = plt.gca()
    ax.arrow(0,0,plda.mu[0],plda.mu[1],
             head_width=hw, head_length=hl, fc=fc, ec=ec,
             linestyle=linestyle, label=label)
    Sb = la.inv(plda.B)
    pge2d(plda.mu, Sb, color=colors[1], linestyle=linestyle)
    Sw = la.inv(plda.W)
    pge2d(plda.mu, Sw, color=colors[2], linestyle=linestyle)
    

    

def test_sample():

    plda = create_plda()
    x = plda.sample(num_classes, num_spc)

    spk_idx = [0, 1, 2, 3]
    colors = ['b','r','g','m','c']
    markers = ['o','v','*','p','s']
    plt.figure()
    for i in xrange(len(spk_idx)):
        idx = spk_idx[i]
        x_i = x[idx*num_spc:(idx+1)*num_spc]
        plt.scatter(x_i[:,0], x_i[:,1], 2, colors[i], markers[i])

    plt.savefig(output_dir + '/sample.pdf')
    plt.close()


    
def test_initialize():

    plda1 = create_plda()
    x_train = plda1.sample(num_classes, num_spc, seed=1024)
    class_ids = np.repeat(np.arange(num_classes), num_spc)

    plda2 = FRPLDA()
    D = plda2.compute_stats_hard(x_train, class_ids)
    plda2.initialize(D)

    plt.figure()
    plot_plda(plda1, label='ground truth')
    plot_plda(plda2, linestyle='--', label='trained')
    plt.legend()
    plt.grid()
    plt.savefig(output_dir + '/init.pdf')
    plt.close()

    

def test_py():
    
    plda = create_plda()
    x,y = plda.sample(3, 3, seed=1024, return_y=True)
    class_ids = np.repeat(np.arange(3), 3)
    y_t = y[::3]

    D = plda.compute_stats_hard(x, class_ids)
    y, Cy = plda.compute_py_g_x(D, return_cov=True)

    colors = ['b','r','g']

    plt.figure()
    for i in xrange(len(y_t)):
        plt.plot(y_t[i,0], y_t[i,1], color=colors[i], marker='*')
        plt.plot(y[i,0], y[i,1], color=colors[i], marker='s')
        pge2d(y[i], Cy[i], color=colors[i])
    plt.grid()
    plt.savefig(output_dir+'/py.pdf')
    plt.close()



def test_fit():

    plda1 = create_plda()
    x_train = plda1.sample(num_classes, num_spc, seed=1024)
    class_ids = np.repeat(np.arange(num_classes), num_spc)

    plda2 = FRPLDA()
    elbo = plda2.fit(x_train, class_ids)

    plt.figure()
    plt.plot(elbo)
    plt.grid()
    plt.savefig(output_dir + '/fit_elbo.pdf')
    plt.close()

    plt.figure()
    plot_plda(plda1, label='ground truth')
    plot_plda(plda2, linestyle='--', label='trained')
    plt.legend()
    plt.grid()
    plt.savefig(output_dir + '/fit.pdf')
    plt.close()
    


def test_llr_1vs1():

    plda = create_plda()
    x = plda.sample(num_classes, 2, seed=1024)
    x_e = x[::2]
    x_t = x[1::2]
    tar = np.eye(num_classes, dtype=bool)
    non = np.logical_not(tar)
    
    scores = plda.llr_1vs1(x_e, x_t)
    scores_tar = scores[tar]
    scores_non = scores[non]

    assert np.mean(scores_tar) > np.mean(scores_non)

    plt.figure()
    plt.hist(scores_tar, int(num_classes/10), density=True, label='tar', color='b')
    plt.hist(scores_non, int(num_classes**2/20), density=True, label='non', color='r')
    plt.grid()
    plt.savefig(output_dir + '/llr_1vs1.pdf')
    plt.close()
    

def test_llrNvsM():

    plt.figure()
    
    plda = create_plda()
    x = plda.sample(num_classes, 6, seed=1024)
    x_e = x[::2]
    x_t = x[1::2]
    class_ids = np.repeat(np.arange(num_classes), 3)
    tar = np.eye(num_classes, dtype=bool)
    non = np.logical_not(tar)

    ## by the book
    scores = plda.llr_NvsM(x_e, x_t, ids1=class_ids, ids2=class_ids, method='book')
    scores_tar = scores[tar]
    scores_non = scores[non]

    assert np.mean(scores_tar) > np.mean(scores_non)

    plt.hist(scores_tar, int(num_classes/10), density=True, label='book', color='b')
    plt.hist(scores_non, int(num_classes**2/20), density=True, color='b')

    ## score averaging
    scores = plda.llr_NvsM(x_e, x_t, ids1=class_ids, ids2=class_ids, method='savg')
    scores_tar = scores[tar]
    scores_non = scores[non]

    assert np.mean(scores_tar) > np.mean(scores_non)

    plt.hist(scores_tar, int(num_classes/10), density=True, label='s-avg', color='r')
    plt.hist(scores_non, int(num_classes**2/20), density=True, color='r')


    ## i-vector averaging
    scores = plda.llr_NvsM(x_e, x_t, ids1=class_ids, ids2=class_ids, method='vavg')
    scores_tar = scores[tar]
    scores_non = scores[non]

    assert np.mean(scores_tar) > np.mean(scores_non)

    plt.hist(scores_tar, int(num_classes/10), density=True, label='iv-avg', color='g')
    plt.hist(scores_non, int(num_classes**2/20), density=True, color='g')


    ## i-vector averaging
    scores = plda.llr_NvsM(x_e, x_t, ids1=class_ids, ids2=class_ids, method='vavg-lnorm')
    scores_tar = scores[tar]
    scores_non = scores[non]

    assert np.mean(scores_tar) > np.mean(scores_non)

    plt.hist(scores_tar, int(num_classes/10), density=True, label='iv-avg+lnorm', color='m')
    plt.hist(scores_non, int(num_classes**2/20), density=True, color='m')

    
    plt.grid()
    plt.savefig(output_dir + '/llr_NvsM.pdf')
    plt.close()


    
def test_llrNvs1():

    plt.figure()
    
    plda = create_plda()
    x = plda.sample(num_classes, 4, seed=1024)
    mask = np.zeros((len(x),), dtype=bool)
    mask[::4] = True
    x_e = x[mask==False]
    x_t = x[mask]
    class_ids = np.repeat(np.arange(num_classes), 3)
    tar = np.eye(num_classes, dtype=bool)
    non = np.logical_not(tar)

    ## by the book
    scores = plda.llr_Nvs1(x_e, x_t, ids1=class_ids, method='book')
    scores_tar = scores[tar]
    scores_non = scores[non]

    assert np.mean(scores_tar) > np.mean(scores_non)

    plt.hist(scores_tar, int(num_classes/10), density=True, label='book', color='b')
    plt.hist(scores_non, int(num_classes**2/20), density=True, color='b')

    ## score averaging
    scores = plda.llr_Nvs1(x_e, x_t, ids1=class_ids, method='savg')
    scores_tar = scores[tar]
    scores_non = scores[non]

    assert np.mean(scores_tar) > np.mean(scores_non)

    plt.hist(scores_tar, int(num_classes/10), density=True, label='s-avg', color='r')
    plt.hist(scores_non, int(num_classes**2/20), density=True, color='r')


    ## i-vector averaging
    scores = plda.llr_Nvs1(x_e, x_t, ids1=class_ids, method='vavg')
    scores_tar = scores[tar]
    scores_non = scores[non]

    assert np.mean(scores_tar) > np.mean(scores_non)

    plt.hist(scores_tar, int(num_classes/10), density=True, label='iv-avg', color='g')
    plt.hist(scores_non, int(num_classes**2/20), density=True, color='g')


    ## i-vector averaging
    scores = plda.llr_Nvs1(x_e, x_t, ids1=class_ids, method='vavg-lnorm')
    scores_tar = scores[tar]
    scores_non = scores[non]

    assert np.mean(scores_tar) > np.mean(scores_non)

    plt.hist(scores_tar, int(num_classes/10), density=True, label='iv-avg+lnorm', color='m')
    plt.hist(scores_non, int(num_classes**2/20), density=True, color='m')

    
    plt.grid()
    plt.savefig(output_dir + '/llr_Nvs1.pdf')
    plt.close()




    
    
