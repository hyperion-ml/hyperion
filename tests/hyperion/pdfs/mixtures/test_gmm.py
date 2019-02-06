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

from numpy.testing import assert_allclose
from scipy import linalg as la

from hyperion.utils.math import symmat2vec
from hyperion.pdfs import GMMDiagCov, GMM

output_dir = './tests/data_out/pdfs/core/mixtures/gmm'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


x_dim = 3
pi1 = np.array([0.5, 0.25, 0.125, 0.125])
mu1 = np.array([[-2, 1.5, -1],
                [1, 1, 0],
                [0, -1, 1],
                [1.5, -1.5, 0.5]])
S1 = np.square(np.array([[1, 0.75, 0.5],
                         [0.5, 0.3, 0.1],
                         [0.5, 0.6, 0.7],
                         [0.5, 0.4, 0.3]]))

S0fc = np.array([[1, 0.3, 0.1],
                 [0.3, 1, -0.25],
                 [0.1, -0.25, 1]])

S1fc = np.zeros((len(pi1), x_dim, x_dim))
L1fc = np.zeros((len(pi1), x_dim, x_dim))
L1dc = np.zeros((len(pi1), x_dim, x_dim))
for k in xrange(len(pi1)):
    SS = S1[k]*S0fc
    S1fc[k] = (SS+SS.T)/2
    L1fc[k] = la.inv(S1fc[k])
    L1dc[k] = np.diag(1/S1[k])
    
num_samples = 1000
batch_size = 250
num_samples_init = 100
num_samples_train = 10000
model_file = output_dir + '/model.h5'


def create_diag_pdf():

    model_diag = GMMDiagCov(num_comp=len(pi1), pi=pi1, mu=mu1, Lambda=1/S1, x_dim=x_dim)
    model = GMM(num_comp=len(pi1), pi=pi1, mu=mu1, Lambda=L1dc, x_dim=x_dim)
    return model, model_diag



def create_pdf():
    
    model = GMM(num_comp=len(pi1), pi=pi1, mu=mu1, Lambda=L1fc, x_dim=x_dim)
    return model



def test_diag_properties():

    model, model_diag = create_diag_pdf()
    assert_allclose(model.log_pi, model_diag.log_pi)
    assert_allclose(model.logLambda, model_diag.logLambda)
    for k in xrange(model.num_comp):
        assert_allclose(model.Sigma[k], np.diag(model_diag.Sigma[k]))
        assert_allclose(model.cholLambda[k], np.diag(np.sqrt(model_diag.Lambda[k])))
        
        

def test_properties():

    model = create_pdf()
    assert_allclose(model.log_pi, np.log(model.pi))

    for k in xrange(model.num_comp):
        assert_allclose(model.Sigma[k], la.inv(model.Lambda[k]))
        assert_allclose(model.cholLambda[k], la.cholesky(model.Lambda[k], lower=True))
        assert_allclose(model.logLambda[k],
                        2*np.sum(np.log(np.diag(la.cholesky(model.Lambda[k])))))



def test_diag_initialize():

     model1, model1_diag = create_diag_pdf()
     model1.initialize()
     model1_diag.initialize()

     model2 = GMM(num_comp=model1.num_comp,
                  pi=model1.pi,
                  eta=model1.eta, x_dim=model1.x_dim)
     model2.initialize()

     assert_allclose(model1.compute_A_std(model1.mu, model1.Lambda), model1_diag.A)
     assert_allclose(model1.compute_A_nat(model1.eta), model1_diag.A)
     assert_allclose(model1.A, model1_diag.A)
     assert_allclose(model2.A, model1_diag.A)
     
     assert_allclose(model1.mu, model1_diag.mu)
     assert_allclose(model2.mu, model1_diag.mu)

     for k in xrange(model1.num_comp):
         assert_allclose(model1.Lambda[k], np.diag(model1_diag.Lambda[k]))
         assert_allclose(model2.Lambda[k], np.diag(model1_diag.Lambda[k]))

     
        
def test_initialize():

     model1 = create_pdf()
     model1.initialize()

     model2 = GMM(num_comp=model1.num_comp,
                  pi=model1.pi,
                  eta=model1.eta, x_dim=model1.x_dim)
     assert_allclose(model1.eta, model2.eta)
     model2.initialize()

     model3 = GMM(num_comp=model2.num_comp,
                  pi=model2.pi,
                  mu=model2.mu,
                  Lambda=model2.Lambda,
                  x_dim=model1.x_dim)
     assert_allclose(model1.eta, model3.eta, atol=1e-5)
     model3.initialize()

     assert_allclose(model1.eta, model2.eta, atol=1e-5)
     assert_allclose(model1.eta, model3.eta, atol=1e-5)

     assert_allclose(model1.A, model2.A)
     assert_allclose(model1.A, model3.A)

     assert_allclose(model1.mu, model2.mu, atol=1e-10)
     assert_allclose(model1.mu, model3.mu, atol=1e-10)

     assert_allclose(model1.Lambda, model2.Lambda)
     assert_allclose(model1.Lambda, model3.Lambda)


     
def test_initialize_stdnormal():

    model = GMM(num_comp=1, x_dim=x_dim)
    model.initialize()
    
    assert(model.pi==1)
    assert_allclose(model.mu, np.zeros((1, x_dim)))
    assert_allclose(model.Lambda[0], np.eye(x_dim))



def test_initialize_kmeans():

    model1 = create_pdf()
    x = model1.sample(num_samples=num_samples_init)
    
    model2 = GMM(num_comp=4, x_dim=x_dim)
    model2.initialize(x)

    print(model1.mu)
    print(model2.mu[[2,3,1,0]])


    
def test_log_h():
     model1 = create_pdf()

     sample_weight = np.arange(1,num_samples+1, dtype=float)/num_samples
    
     assert(model1.log_h(None) == 0)
     assert(model1.accum_log_h(None, sample_weight=sample_weight) == 0)


     
def test_suff_stats():

    model1 = create_pdf()

    x = model1.sample(num_samples)
    sample_weight = 0.5*np.ones((num_samples,))

    xx = []
    for i in xrange(x.shape[0]):
        xx.append(symmat2vec(np.outer(x[i], x[i])))
    xx = np.vstack(xx)
    u_x = np.hstack((x, xx))
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
                         [3*num_samples/4, num_samples-1]])

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



def test_diag_log_prob():

    model1, model1_diag = create_diag_pdf()

    x = model1.sample(num_samples)
    
    assert_allclose(model1.log_prob(x, mode='nat'),
                    model1_diag.log_prob(x, mode='std'))
    assert_allclose(model1.log_prob(x, mode='std'),
                    model1_diag.log_prob(x, mode='std'))


    
def test_log_prob():

    model1 = create_pdf()

    x = model1.sample(num_samples)
    
    assert_allclose(model1.log_prob(x, mode='nat'),
                    model1.log_prob(x, mode='std'))

    u_x = model1.compute_suff_stats(x)
    assert_allclose(model1.log_prob(x, u_x, mode='nat'),
                    model1.log_prob(x, mode='std'))
    



def test_diag_elbo():

    model1, model1_diag = create_diag_pdf()

    x = model1.sample(num_samples)
    sample_weight = 0.5*np.ones((num_samples,))
    
    assert_allclose(model1.elbo(x), model1_diag.elbo(x))
    assert_allclose(model1.elbo(x, sample_weight=sample_weight),
                    0.5*model1_diag.elbo(x))


    
def test_elbo():

    model1 = create_pdf()

    x = model1.sample(num_samples)
    sample_weight = 0.5*np.ones((num_samples,))
    
    assert(model1.elbo(x)/num_samples + 0.4> np.mean(model1.log_prob(x, mode='std')))
    assert(model1.elbo(x, sample_weight=sample_weight)/num_samples + 0.2> 
           0.5*np.sum(model1.log_prob(x, mode='std')))
    

    
def test_diag_fit_kmeans():

    model1, _ = create_diag_pdf()
    model1.initialize()

    x = model1.sample(num_samples_train)
    x_val = model1.sample(num_samples)

    model2 = GMM(num_comp=4, x_dim=x_dim)
    model2.initialize(x)
    elbo = model2.fit(x, x_val=x_val)

    model2_diag = GMMDiagCov(num_comp=4, x_dim=x_dim)
    model2_diag.initialize(x)
    elbo_diag = model2_diag.fit(x, x_val=x_val)


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
    plt.plot(elbo_diag[1], 'g')
    plt.plot(elbo_diag[3], 'g--')
    plt.savefig(output_dir + '/fit_kmeans_init_elbo.pdf')
    plt.close()


    
def test_fit_kmeans():

    model1 = create_pdf()
    model1.initialize()

    x = model1.sample(num_samples_train)
    x_val = model1.sample(num_samples)

    model2 = GMM(num_comp=4, x_dim=x_dim)
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
    plt.savefig('./tests/data_out/gmm_fit_kmeans_init_elbo.pdf')
    plt.close()

    

    
def test_fit_kmeans_split2():

    model1 = create_pdf()
    model1.initialize()

    x = model1.sample(num_samples_train)
    x_val = model1.sample(num_samples)

    model2 = GMMDiagCov(num_comp=1, x_dim=x_dim)
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
    plt.savefig('./tests/data_out/diag_gmm_fit_split2_init_elbo.pdf')
    plt.close()

    


def test_fit_kmeans_split4():

    model1 = create_pdf()
    model1.initialize()

    x = model1.sample(num_samples_train)
    x_val = model1.sample(num_samples)

    model2 = GMMDiagCov(num_comp=1, x_dim=x_dim)
    model2.initialize()
    elbo = model2.fit(x, x_val=x_val, epochs=1)
    model2 = model2.split_comp(4)
    elbo = model2.fit(x, x_val=x_val)

    model1.plot2D(feat_idx=[0,1], num_sigmas=1)
    plt.savefig(output_dir + '/plot_fit_split4_init_D01.pdf')
    plt.close()
    model1.plot2D(feat_idx=[0,2], num_sigmas=1)
    plt.savefig(output_dir + '/plot_fit_split4_init_D02.pdf')
    plt.close()

    plt.figure()
    plt.plot(np.repeat(model1.elbo(x)/x.shape[0], len(elbo[1])), 'b')
    plt.plot(np.repeat(model1.elbo(x_val)/x.shape[0], len(elbo[1])), 'b--')
    plt.plot(elbo[1], 'r')
    plt.plot(elbo[3], 'r--')
    plt.savefig(output_dir + '/fit_split4_init_elbo.pdf')
    plt.close()



# def test_eval_logcdf():

#      model1 = create_pdf()
#      model1.initialize()

#      assert(model1.eval_logcdf(1e20*np.ones((1,x_dim,))) > np.log(0.99))
#      assert(model1.eval_logcdf(-1e20*np.ones((1,x_dim,))) < np.log(0.01))


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


