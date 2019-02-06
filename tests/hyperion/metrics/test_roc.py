"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import pytest
import os
import numpy as np
from numpy.testing import assert_allclose

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperion.metrics.roc import *

output_dir = './tests/data_out/metrics'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def test_roc():

    plt.figure()

    plt.subplot(2,3,1)
    tar = np.array([1])
    non = np.array([0])
    pmiss, pfa = compute_rocch(tar,non)
    pm, pf = compute_roc(tar,non)
    h1, = plt.plot(pfa,pmiss,'r-^', label='ROCCH',linewidth=2)
    h2, = plt.plot(pf,pm,'g--v', label='ROC',linewidth=2)
    plt.axis('square')
    plt.grid(True)
    plt.legend(handles=[h1, h2])
    plt.title('2 scores: non < tar')
    print(pmiss, pfa)
    print(pm, pf)
    assert_allclose(pmiss,[ 0.,0.,1.])
    assert_allclose(pfa,[1.,0.,0.])
    assert_allclose(pm,[0.,0.,1.])
    assert_allclose(pf,[1.,0.,0.])

    
    plt.subplot(2,3,2)
    tar = np.array([0])
    non = np.array([1])
    pmiss, pfa = compute_rocch(tar,non)
    pm, pf = compute_roc(tar,non)
    plt.plot(pfa,pmiss,'r-^',pf,pm,'g--v',linewidth=2)
    plt.axis('square')
    plt.grid(True)
    plt.title('2 scores: tar < non')
    print(pmiss, pfa)
    print(pm, pf)
    assert_allclose(pmiss,[0.,1.])
    assert_allclose(pfa,[1.,0.])
    assert_allclose(pm,[0.,1.,1.])
    assert_allclose(pf,[1.,1.,0.])

    
    plt.subplot(2,3,3)
    tar = np.array([0])
    non = np.array([-1,1])
    pmiss, pfa = compute_rocch(tar,non)
    pm, pf = compute_roc(tar,non)
    plt.plot(pfa,pmiss,'r-^',pf,pm,'g--v',linewidth=2)
    plt.axis('square')
    plt.grid(True)
    plt.title('3 scores: non < tar < non')
    print(pmiss, pfa)
    print(pm, pf)
    assert_allclose(pmiss,[0.,0.,1.])
    assert_allclose(pfa,[1.,0.5,0.])
    assert_allclose(pm,[0.,0.,1.,1.])
    assert_allclose(pf,[1.,0.5,0.5,0.])

    
    plt.subplot(2,3,4)
    tar = np.array([-1,1])
    non = np.array([0])
    pmiss, pfa = compute_rocch(tar,non)
    pm, pf = compute_roc(tar,non)
    plt.plot(pfa,pmiss,'r-^',pf,pm,'g--v',linewidth=2)
    plt.axis('square')
    plt.grid(True)
    plt.title('3 scores: tar < non < tar')
    plt.xlabel(r'$P_{fa}$')
    plt.ylabel(r'$P_{miss}$')
    print(pmiss, pfa)
    print(pm, pf)
    assert_allclose(pmiss, [0.,0.5,1.])
    assert_allclose(pfa, [1.,0.,0.])
    assert_allclose(pm, [0.,0.5,0.5,1.])
    assert_allclose(pf, [1.,1.,0.,0.])

    
    plt.subplot(2,3,5)
    rng = np.random.RandomState(100)
    tar = rng.randn(100)+1
    non = rng.randn(100)
    pmiss, pfa = compute_rocch(tar,non)
    pm, pf = compute_roc(tar,non)
    plt.plot(pfa,pmiss,'r-^',pf,pm,'g',linewidth=2)
    plt.axis('square')
    plt.grid(True)
    plt.title('DET')
    print(pmiss, pfa)
    print(pm[:10], pf[:10])
    assert_allclose(pmiss, [0.,0.,0.01,0.16,0.22,0.29,0.45,0.5,0.89,0.92,1.])
    assert_allclose(pfa, [1.,0.91,0.77,0.48,0.4,0.33,0.19,0.15,0.01,0.,0.])
    assert_allclose(pm[:10], [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
    assert_allclose(pf[:10], [1.,0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91])

    
    plt.subplot(2,3,6)
    tar = rng.randn(100)*2+1
    non = rng.randn(100)
    pmiss, pfa = compute_rocch(tar,non)
    pm, pf = compute_roc(tar,non)
    plt.plot(pfa,pmiss,'r-^',pf,pm,'g',linewidth=2)
    plt.axis('square')
    plt.grid(True)
    plt.title('flatter DET')
    print(pmiss, pfa)
    print(pm[:10], pf[:10])
    assert_allclose(pmiss,[0.,0.31,0.48,0.5,0.58,0.62,0.85,1.])
    assert_allclose(pfa,[1.,0.41,0.15,0.12,0.06,0.05,0.,0.])
    assert_allclose(pm[:10], [0.,0.01,0.02,0.03,0.04,0.05,0.05,0.05,0.05,0.05])
    assert_allclose(pf[:10], [1.,1.,1.,1.,1.,1.,0.99,0.98,0.97, 0.96])

    # plt.show()
    plt.savefig(output_dir + '/roc.pdf')
    plt.close()



def test_rocch2eer():

    rng = np.random.RandomState(100)
    tar = rng.randn(100)+1
    non = rng.randn(100)
    pmiss, pfa = compute_rocch(tar,non)
    eer = rocch2eer(pmiss, pfa)
