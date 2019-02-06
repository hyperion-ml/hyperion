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

import matplotlib
matplotlib.use('Agg')
from hyperion.utils.plotting import *

output_dir = './tests/data_out/utils/plot'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def test_plot_gaussian():
    
    mu=np.array([1, 2, 3])
    C=np.array([[2, 0.5, 0.2], [0.5, 1., 0.1], [0.2, 0.1, 0.8]])
    la.cholesky(C)

    mu1 = mu[0]
    C1 = C[0,0]
    #plt.figure(figsize=(6, 6))
    plot_gaussian_1D(mu1, C1)
    # plt.show()
    plt.savefig(output_dir + '/plot_gaussian_1D.pdf')
    plt.close()
    
    mu2 = mu[:2]
    C2 = C[:2,:2]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    plot_gaussian_3D(mu2, C2, ax=ax)
    # plt.show()
    plt.savefig(output_dir + '/plot_gaussian_3D.pdf')
    plt.close()
    
    #plt.figure(figsize=(6, 6))
    plot_gaussian_ellipsoid_2D(mu2, C2)
    #plt.show()
    plt.savefig(output_dir + '/plot_gaussian_ellipsoid_2D.pdf')
    plt.close()
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    plot_gaussian_ellipsoid_3D(mu, C, ax=ax)
    # plt.show()
    plt.savefig(output_dir + '/plot_gaussian_ellipsoid_3D.pdf')
    plt.close()
    


if __name__ == '__main__':
    pytest.main([__file__])
