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
import matplotlib.pyplot as plt

from hyperion.pdfs import GMMTiedDiagCov
from numpy.testing import assert_allclose

output_dir = './tests/data_out/pdfs/core/mixtures/gmm_tied_diag_cov'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

x_dim = 3
pi1 = np.array([0.5, 0.25, 0.125, 0.125])
mu1 = np.array([[-2, 1.5, -1],
                [1, 1, 0],
                [0, -1, 1],
                [1.5, -1.5, 0.5]])
S1 = np.square(np.array([1, 0.75, 0.5]))

num_samples = 1000
batch_size = 250
num_samples_init = 100
num_samples_train = 10000
model_file = output_dir + '/model.h5'


def create_pdf():

    model = GMMTiedDiagCov(num_comp=len(pi1), pi=pi1, mu=mu1, Lambda=1/S1, x_dim=x_dim)
    return model

    
def test_properties():

    model = create_pdf()
    assert_allclose(model.log_pi, np.log(model.pi))
    assert_allclose(model.Sigma, 1/model.Lambda)
    assert_allclose(model.cholLambda, np.sqrt(model.Lambda))
    assert_allclose(model.logLambda, np.sum(np.log(model.Lambda), axis=-1))


    
def test_initialize():

     model1 = create_pdf()
     model1.initialize()

     model2 = GMMTiedDiagCov(num_comp=model1.num_comp,
                      pi=model1.pi,
                      eta=model1.eta, x_dim=model1.x_dim)
     model2.initialize()

     model3 = GMMTiedDiagCov(num_comp=model2.num_comp,
                      pi=model2.pi,
                      mu=model2.mu,
                      Lambda=model2.Lambda,
                      x_dim=model1.x_dim)
     model3.initialize()

     assert_allclose(model1.eta, model2.eta)
     assert_allclose(model1.eta, model3.eta)

     assert_allclose(model1.A, model2.A)
     assert_allclose(model1.A, model3.A)

     assert_allclose(model1.mu, model2.mu)
     assert_allclose(model1.mu, model3.mu)

     assert_allclose(model1.Lambda, model2.Lambda)
     assert_allclose(model1.Lambda, model3.Lambda)


     
def test_initialize_stdnormal():

    model = GMMTiedDiagCov(num_comp=1, x_dim=x_dim)
    model.initialize()

    assert(model.pi==1)
    assert_allclose(model.mu, np.zeros((1, x_dim)))
    assert_allclose(model.Lambda, np.ones((x_dim,)))



def test_initialize_kmeans():

    model1 = create_pdf()
    x = model1.sample(num_samples=num_samples_init)
    
    model2 = GMMTiedDiagCov(num_comp=4, x_dim=x_dim)
    model2.initialize(x)


    
def test_log_h():
     model1 = create_pdf()

     sample_weight = np.arange(1,num_samples+1, dtype=float)/num_samples
    
     assert(model1.log_h(None) == 0)
     assert(model1.accum_log_h(None, sample_weight=sample_weight) == 0)


     
def test_suff_stats():

    model1 = create_pdf()

    x = model1.sample(num_samples)
    sample_weight = 0.5*np.ones((num_samples,))
    
    u_x = np.hstack((x, x*x))
    assert_allclose(model1.compute_suff_stats(x), u_x)

    N, u_x = model1.accum_suff_stats(x)

    N1, u_x1 = model1.accum_suff_stats(
        x, batch_size=batch_size)
    assert_allclose(N1, N)
    assert_allclose(u_x1, u_x)

    N1, u_x1 = model1.accum_suff_stats(
        x, sample_weight=sample_weight)
    assert_allclose(N1, 0.5*N)
    assert_allclose(u_x1, 0.5*u_x)

    N1, u_x1 = model1.accum_suff_stats(
        x, sample_weight=sample_weight, batch_size=batch_size)
    assert_allclose(N1, 0.5*N)
    assert_allclose(u_x1, 0.5*u_x)
                    

    
def test_suff_stats_segments():

    model1 = create_pdf()

    x = model1.sample(num_samples)
    sample_weight = 0.5*np.ones((num_samples,))
    
    N, u_x = model1.accum_suff_stats(x)

    segments = np.array([[0, num_samples/2-1],
                         [num_samples/2, num_samples-1],
                         [0, num_samples/4-1],
                         [num_samples/4, num_samples/2-1],
                         [num_samples/2, 3*num_samples/4-1],
                         [3*num_samples/4, num_samples-1]], dtype=int)

    print(N.shape)
    print(u_x.shape)
    N1, u_x1 = model1.accum_suff_stats_segments(
        x, segments, batch_size=batch_size)
    assert_allclose(np.sum(N1, axis=0), 2*N)
    assert_allclose(np.sum(u_x1, axis=0), 2*u_x)

    N2, u_x2 = model1.accum_suff_stats_segments(
        x, segments, sample_weight=sample_weight)
    assert_allclose(N2, 0.5*N1)
    assert_allclose(u_x2, 0.5*u_x1)

    N2, u_x2 = model1.accum_suff_stats_segments(
        x, segments, sample_weight=sample_weight, batch_size=batch_size)
    assert_allclose(N2, 0.5*N1)
    assert_allclose(u_x2, 0.5*u_x1)



def test_suff_stats_segments_prob():

    model1 = create_pdf()

    x = model1.sample(num_samples)
    sample_weight = 0.5*np.ones((num_samples,))
    
    N, u_x = model1.accum_suff_stats(x)

    prob = np.zeros((num_samples, 4))
    prob[:int(num_samples/2), 0]=1
    prob[int(num_samples/2):int(3*num_samples/4), 1]=1
    prob[int(3*num_samples/4):int(4*num_samples/5), 2]=1
    prob[int(4*num_samples/5):, 3]=1
    
    N1, u_x1 = model1.accum_suff_stats_segments_prob(
        x, prob, batch_size=batch_size)
    assert_allclose(np.sum(N1, axis=0), N)
    assert_allclose(np.sum(u_x1, axis=0), u_x)

    N2, u_x2 = model1.accum_suff_stats_segments_prob(
        x, prob, sample_weight=sample_weight)
    assert_allclose(N2, 0.5*N1)
    assert_allclose(u_x2, 0.5*u_x1)

    N2, u_x2 = model1.accum_suff_stats_segments_prob(
        x, prob, sample_weight=sample_weight, batch_size=batch_size)
    assert_allclose(N2, 0.5*N1)
    assert_allclose(u_x2, 0.5*u_x1)



def test_suff_stats_sorttime():

    model1 = create_pdf()

    x = model1.sample(num_samples)
    sample_weight = 0.5*np.ones((num_samples,))
    
    N, u_x = model1.accum_suff_stats(x)

    frame_length=int(num_samples/100)
    frame_shift=frame_length
    
    N1, u_x1 = model1.accum_suff_stats_sorttime(
        x, frame_length, frame_shift, batch_size=batch_size)
    assert_allclose(np.sum(N1, axis=0), N)
    assert_allclose(np.sum(u_x1, axis=0), u_x)

    N2, u_x2 = model1.accum_suff_stats_sorttime(
        x, frame_length, frame_shift, sample_weight=sample_weight)
    assert_allclose(N2, 0.5*N1)
    assert_allclose(u_x2, 0.5*u_x1)

    N2, u_x2 = model1.accum_suff_stats_sorttime(
        x, frame_length, frame_shift, sample_weight=sample_weight, batch_size=batch_size)
    assert_allclose(N2, 0.5*N1)
    assert_allclose(u_x2, 0.5*u_x1)




def test_log_prob():

    model1 = create_pdf()

    x = model1.sample(num_samples)
    
    assert_allclose(model1.log_prob(x, mode='nat'),
                    model1.log_prob(x, mode='std'))

    u_x = model1.compute_suff_stats(x)
    assert_allclose(model1.log_prob(x, u_x, mode='nat'),
                    model1.log_prob(x, mode='std'))
    

def test_elbo():

    model1 = create_pdf()

    x = model1.sample(num_samples)
    sample_weight = 0.5*np.ones((num_samples,))
    
    assert(model1.elbo(x)/num_samples + 0.4> np.mean(model1.log_prob(x, mode='std')))
    assert(model1.elbo(x, sample_weight=sample_weight)/num_samples + 0.2> 
           0.5*np.sum(model1.log_prob(x, mode='std')))
    

    
def test_log_cdf():

     model1 = create_pdf()

     assert(model1.log_cdf(1e20*np.ones((1,x_dim,))) > np.log(0.99))
     assert(model1.log_cdf(-1e20*np.ones((1,x_dim,))) < np.log(0.01))

    

def test_fit_kmeans():

    model1 = create_pdf()

    x = model1.sample(num_samples_train)
    x_val = model1.sample(num_samples)

    model2 = GMMTiedDiagCov(num_comp=4, x_dim=x_dim)
    model2.initialize(x)
    elbo = model2.fit(x, x_val=x_val)


    model2.plot2D(feat_idx=[0,1], num_sigmas=1)
    plt.savefig(output_dir + '/plot_fit_kmeans_init_D01.pdf')
    plt.close()
    model2.plot2D(feat_idx=[0,2], num_sigmas=1)
    plt.savefig(output_dir + '/plot_fit_kmeans_init_D02.pdf')
    plt.close()

    plt.figure()
    plt.plot(np.repeat(model1.elbo(x)/x.shape[0], len(elbo[1])), 'b')
    plt.plot(np.repeat(model1.elbo(x_val)/x.shape[0], len(elbo[1])), 'b--')
    plt.plot(elbo[1], 'r')
    plt.plot(elbo[3], 'r--')
    plt.savefig(output_dir + '/fit_kmeans_init_elbo.pdf')
    plt.close()
    

    
def test_fit_kmeans_split2():

    model1 = create_pdf()

    x = model1.sample(num_samples_train)
    x_val = model1.sample(num_samples)

    model2 = GMMTiedDiagCov(num_comp=1, x_dim=x_dim)
    model2.initialize()
    elbo = model2.fit(x, x_val=x_val, epochs=1)
    model2 = model2.split_comp(2)
    elbo = model2.fit(x, x_val=x_val)
    model2 = model2.split_comp(2)
    elbo = model2.fit(x, x_val=x_val)


    model2.plot2D(feat_idx=[0,1], num_sigmas=1)
    plt.savefig(output_dir + '/plot_fit_split2_init_D01.pdf')
    plt.close()
    model2.plot2D(feat_idx=[0,2], num_sigmas=1)
    plt.savefig(output_dir + '/plot_fit_split2_init_D02.pdf')
    plt.close()

    plt.figure()
    plt.plot(np.repeat(model1.elbo(x)/x.shape[0], len(elbo[1])), 'b')
    plt.plot(np.repeat(model1.elbo(x_val)/x.shape[0], len(elbo[1])), 'b--')
    plt.plot(elbo[1], 'r')
    plt.plot(elbo[3], 'r--')
    plt.savefig(output_dir + '/fit_split2_init_elbo.pdf')
    plt.close()

    

def test_fit_kmeans_split4():

    model1 = create_pdf()
    model1.initialize()

    x = model1.sample(num_samples_train)
    x_val = model1.sample(num_samples)

    model2 = GMMTiedDiagCov(num_comp=1, x_dim=x_dim)
    model2.initialize()
    elbo = model2.fit(x, x_val=x_val, epochs=1)
    model2 = model2.split_comp(4)
    elbo = model2.fit(x, x_val=x_val)

    model2.plot2D(feat_idx=[0,1], num_sigmas=1)
    plt.savefig(output_dir + '/plot_fit_split4_init_D01.pdf')
    plt.close()
    model2.plot2D(feat_idx=[0,2], num_sigmas=1)
    plt.savefig(output_dir + '/plot_fit_split4_init_D02.pdf')
    plt.close()

    plt.figure()
    plt.plot(np.repeat(model1.elbo(x)/x.shape[0], len(elbo[1])), 'b')
    plt.plot(np.repeat(model1.elbo(x_val)/x.shape[0], len(elbo[1])), 'b--')
    plt.plot(elbo[1], 'r')
    plt.plot(elbo[3], 'r--')
    plt.savefig(output_dir + '/fit_split4_init_elbo.pdf')
    plt.close()

    

def test_plot():
    
     model1 = create_pdf()

     model1.plot1D()
     plt.savefig(output_dir + '/plot_1D.pdf')
     plt.close()

     model1.plot2D()
     plt.savefig(output_dir + '/plot_2D.pdf')
     plt.close()

     model1.plot3D()
     plt.savefig(output_dir + '/plot_3D.pdf')
     plt.close()

     model1.plot3D_ellipsoid()
     plt.savefig(output_dir + '/plot_3De.pdf')
     plt.close()


if __name__ == '__main__':
    pytest.main([__file__])


