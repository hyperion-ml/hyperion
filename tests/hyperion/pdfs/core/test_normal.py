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
from scipy import linalg as la
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from numpy.testing import assert_allclose

from hyperion.utils.math import symmat2vec
from hyperion.pdfs import NormalDiagCov, Normal

output_dir = './tests/data_out/pdfs/core/normal'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

x_dim = 10
num_samples = 1000
batch_size = 250
num_samples_train = 1000000
model_file = output_dir + '/model.h5'


def create_diag_pdf():
    rng = np.random.RandomState(seed=0)

    mu = rng.randn(x_dim)
    Lambda = np.abs(rng.randn(x_dim))

    model_diag = NormalDiagCov(mu=mu, Lambda=Lambda, x_dim=x_dim)
    model = Normal(mu=mu, Lambda=np.diag(Lambda), x_dim=x_dim)
    return model, model_diag



def create_pdf():

    rng = np.random.RandomState(seed=0)

    mu = rng.randn(x_dim)
    U = rng.randn(x_dim, x_dim)
    Lambda = np.dot(U, U.T)
    model = Normal(mu=mu, Lambda=Lambda, x_dim=x_dim)
    return model
    
    
    
def test_diag_properties():

    model, model_diag = create_diag_pdf()
    assert_allclose(model.Sigma, np.diag(model_diag.Sigma))
    assert_allclose(model.cholLambda, np.diag(np.sqrt(model_diag.Lambda)))
    assert_allclose(model.logLambda, np.sum(np.log(model_diag.Lambda)))


    
def test_properties():

    model = create_pdf()
    assert_allclose(model.Sigma, la.inv(model.Lambda))
    assert_allclose(model.cholLambda, la.cholesky(model.Lambda, lower=True))
    assert_allclose(model.logLambda, 2*np.sum(np.log(np.diag(la.cholesky(model.Lambda)))))
                    

    
def test_diag_initialize():
    
    model1, model1_diag = create_diag_pdf()
    model1.initialize()
    model1_diag.initialize()

    model2 = Normal(eta=model1.eta, x_dim=model1.x_dim)
    model2.initialize()

    assert_allclose(model1.compute_A_std(model1.mu, model1.Lambda), model1_diag.A)
    assert_allclose(model1.compute_A_nat(model1.eta), model1_diag.A)
    assert_allclose(model1.A, model1_diag.A)
    assert_allclose(model2.A, model1_diag.A)

    assert_allclose(model1.mu, model1_diag.mu)
    assert_allclose(model2.mu, model1_diag.mu)

    assert_allclose(model1.Lambda, np.diag(model1_diag.Lambda))
    assert_allclose(model2.Lambda, np.diag(model1_diag.Lambda))


    
def test_initialize():

    model1 = create_pdf()
    model1.initialize()

    model2 = Normal(eta=model1.eta, x_dim=model1.x_dim)
    model2.initialize()

    model3 = Normal(mu=model2.mu, Lambda=model2.Lambda,
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
    N2, u_x2 = model1.accum_suff_stats(x, batch_size=batch_size)

    assert_allclose(model1.accum_suff_stats(x, batch_size=batch_size)[1], u_x)
    assert_allclose(model1.accum_suff_stats(x, sample_weight=sample_weight)[1], 0.5*u_x)
    assert_allclose(model1.accum_suff_stats(x, sample_weight=sample_weight,
                                            batch_size=batch_size)[1], 0.5*u_x)
                    


def test_diag_log_prob():

    model1, model1_diag = create_diag_pdf()

    x = model1.sample(num_samples)
    
    assert_allclose(model1.log_prob(x, method='nat'),
                    model1_diag.log_prob(x, method='std'))
    assert_allclose(model1.log_prob(x, method='std'),
                    model1_diag.log_prob(x, method='std'))



def test_log_prob():

    model1 = create_pdf()

    x = model1.sample(num_samples)
    
    assert_allclose(model1.log_prob(x, method='nat'),
                    model1.log_prob(x, method='std'))

    u_x = model1.compute_suff_stats(x)
    assert_allclose(model1.log_prob(x, u_x, method='nat'),
                    model1.log_prob(x, method='std'))
    


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
    
    assert_allclose(model1.elbo(x),
                    np.sum(model1.log_prob(x, method='std')))
    assert_allclose(model1.elbo(x, sample_weight=sample_weight),
                    0.5*np.sum(model1.log_prob(x, method='std')))
    

    
# def test_eval_logcdf():

#     model1 = create_pdf()
#     model1.initialize()

#     assert(model1.eval_logcdf(model1.mu) == x_dim*np.log(0.5))
#     assert(model1.eval_logcdf(1e10*np.ones((x_dim,))) > np.log(0.99))
#     assert(model1.eval_logcdf(-1e10*np.ones((x_dim,))) < np.log(0.01))
    
    

def test_diag_fit():

    model1, model1_diag = create_diag_pdf()

    x = model1.sample(num_samples_train)
    x_val = model1.sample(num_samples)

    model2 = Normal(x_dim=x_dim)
    elbo = model2.fit(x, x_val=x_val)

    model2_diag = NormalDiagCov(x_dim=x_dim)
    elbo_diag = model2_diag.fit(x, x_val=x_val)
    
    assert_allclose(model2.mu, model2_diag.mu, atol=0.01)
    assert_allclose(np.diag(model2.Lambda), model2_diag.Lambda, atol=0.01)
    assert_allclose(model2.A, model2_diag.A, atol=0.02)
    assert_allclose(elbo[1], elbo_diag[1], rtol=1e-4)
    assert_allclose(elbo[3], elbo_diag[3], rtol=1e-4)



def test_fit():

    model1 = create_pdf()

    x = model1.sample(num_samples_train)
    x_val = model1.sample(num_samples)

    model2 = Normal(x_dim=x_dim)
    elbo = model2.fit(x, x_val=x_val)

    assert_allclose(model2.mu, np.mean(x, axis=0))
    assert_allclose(model2.Lambda, la.inv(np.dot(x.T, x)/num_samples_train
                                          -np.outer(model2.mu, model2.mu)))
    assert_allclose(model1.mu, model2.mu, atol=0.02)
    assert_allclose(model1.Lambda, model2.Lambda, atol=0.2)
    assert_allclose(model1.eta, model2.eta, atol=0.05)
    assert_allclose(model1.A, model2.A, atol=0.05)
    assert_allclose(elbo[1], np.mean(model1.log_prob(x)), rtol=1e-5)
    assert_allclose(elbo[3], np.mean(model1.log_prob(x_val)), rtol=1e-4)
    assert_allclose(elbo[1], np.mean(model2.log_prob(x)), rtol=1e-5)
    assert_allclose(elbo[3], np.mean(model2.log_prob(x_val)), rtol=1e-4)


    
def test_plot():
    
    model1 = create_pdf()

    model1.plot1D()
    plt.savefig(output_dir + '/normal_1D.pdf')
    plt.close()

    model1.plot2D()
    plt.savefig(output_dir + '/normal_2D.pdf')
    plt.close()

    model1.plot3D()
    plt.savefig(output_dir + '/normal_3D.pdf')
    plt.close()

    model1.plot3D_ellipsoid()
    plt.savefig(output_dir + '/normal_3De.pdf')
    plt.close()


    
if __name__ == '__main__':
    pytest.main([__file__])


