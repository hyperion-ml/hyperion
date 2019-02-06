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

from hyperion.pdfs import NormalDiagCov
from numpy.testing import assert_allclose

output_dir = './tests/data_out/pdfs/core/normal_diag_cov'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

x_dim = 10
num_samples = 1000
batch_size = 250
num_samples_train = 100000
model_file = output_dir + '/model.h5'


def create_pdf():
    rng = np.random.RandomState(seed=0)

    mu = rng.randn(x_dim)
    Lambda = np.abs(rng.randn(x_dim))

    model = NormalDiagCov(mu=mu, Lambda=Lambda, x_dim=x_dim)
    return model

    
def test_properties():

    model = create_pdf()
    assert(np.all(model.Sigma == 1/model.Lambda))
    assert(np.all(model.cholLambda == np.sqrt(model.Lambda)))
    assert(np.all(model.logLambda == np.sum(np.log(model.Lambda))))


    
def test_initialize():

    model1 = create_pdf()
    model1.initialize()

    model2 = NormalDiagCov(eta=model1.eta, x_dim=model1.x_dim)
    model2.initialize()

    model3 = NormalDiagCov(mu=model2.mu,
                        Lambda=model2.Lambda,
                        x_dim=model1.x_dim)
    model3.initialize()

    print(model3.eta)
    print(model3.mu)
    print(model3.Lambda)
    print(model1.A)
    print(model2.A)
    print(model3.A)
    
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
    
    u_x = np.hstack((x, x*x))
    assert_allclose(model1.compute_suff_stats(x), u_x)

    N, u_x = model1.accum_suff_stats(x)
    N2, u_x2 = model1.accum_suff_stats(x, batch_size=batch_size)

    assert_allclose(model1.accum_suff_stats(x, batch_size=batch_size)[1], u_x)
    assert_allclose(model1.accum_suff_stats(x, sample_weight=sample_weight)[1], 0.5*u_x)
    assert_allclose(model1.accum_suff_stats(x, sample_weight=sample_weight, batch_size=batch_size)[1], 0.5*u_x)
                    


def test_log_prob():

    model1 = create_pdf()

    x = model1.sample(num_samples)
    
    assert_allclose(model1.log_prob(x, method='nat'),
                    model1.log_prob(x, method='std'))

    u_x = model1.compute_suff_stats(x)
    assert_allclose(model1.log_prob(x, u_x, method='nat'),
                    model1.log_prob(x, method='std'))
    

def test_elbo():

    model1 = create_pdf()

    x = model1.sample(num_samples)
    sample_weight = 0.5*np.ones((num_samples,))
    
    assert_allclose(model1.elbo(x),
                    np.sum(model1.log_prob(x, method='std')))
    assert_allclose(model1.elbo(x, sample_weight=sample_weight),
                    0.5*np.sum(model1.log_prob(x, method='std')))
    

def test_log_cdf():

    model1 = create_pdf()

    assert_allclose(model1.log_cdf(model1.mu), x_dim*np.log(0.5))
    assert model1.log_cdf(1e10*np.ones((x_dim,))) > np.log(0.99)
    assert model1.log_cdf(-1e10*np.ones((x_dim,))) < np.log(0.01)
    
    

def test_fit():

    model1 = create_pdf()

    x = model1.sample(num_samples_train)
    x_val = model1.sample(num_samples)

    model2 = NormalDiagCov(x_dim=x_dim)
    elbo = model2.fit(x, x_val=x_val)

    assert_allclose(model2.mu, np.mean(x, axis=0))
    assert_allclose(model2.Lambda, 1/np.std(x, axis=0)**2)
    assert_allclose(model1.mu, model2.mu, atol=0.01)
    assert_allclose(model1.Lambda, model2.Lambda, atol=0.01)
    assert_allclose(model1.eta, model2.eta, atol=0.01)
    assert_allclose(model1.A, model2.A, atol=0.02)
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


