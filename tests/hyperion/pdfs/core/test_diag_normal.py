
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperion.pdfs import DiagNormal
from numpy.testing import assert_allclose


x_dim = 10
num_samples = 1000
batch_size = 250
num_samples_train = 100000
model_file = './tests/data_out/model.h5'


def create_pdf():
    rng = np.random.RandomState(seed=0)

    mu = rng.randn(x_dim)
    Lambda = np.abs(rng.randn(x_dim))

    model = DiagNormal(mu=mu, Lambda=Lambda, x_dim=x_dim)
    return model

    
def test_properties():

    model = create_pdf()
    assert(np.all(model.Sigma == 1/model.Lambda))
    assert(np.all(model.cholLambda == np.sqrt(model.Lambda)))
    assert(np.all(model.lnLambda == np.sum(np.log(model.Lambda))))


    
def test_initialize():

    model1 = create_pdf()
    model1.initialize()

    model2 = DiagNormal(eta=model1.eta, x_dim=model1.x_dim)
    model2.initialize()

    model3 = DiagNormal(mu=model2.mu,
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


def test_logh():
    model1 = create_pdf()
    model1.initialize()

    sample_weights = np.arange(1,num_samples+1, dtype=float)/num_samples
    
    assert(model1.logh(None) == 0)
    assert(model1.accum_logh(None, sample_weights=sample_weights) == 0)


def test_suff_stats():

    model1 = create_pdf()
    model1.initialize()

    x = model1.generate(num_samples)
    sample_weights = 0.5*np.ones((num_samples,))
    
    u_x = np.hstack((x, x*x))
    assert_allclose(model1.compute_suff_stats(x), u_x)

    N, u_x = model1.accum_suff_stats(x)
    N2, u_x2 = model1.accum_suff_stats(x, batch_size=batch_size)

    assert_allclose(model1.accum_suff_stats(x, batch_size=batch_size)[1], u_x)
    assert_allclose(model1.accum_suff_stats(x, sample_weights=sample_weights)[1], 0.5*u_x)
    assert_allclose(model1.accum_suff_stats(x, sample_weights=sample_weights, batch_size=batch_size)[1], 0.5*u_x)
                    


def test_eval_llk():

    model1 = create_pdf()
    model1.initialize()

    x = model1.generate(num_samples)
    
    assert_allclose(model1.eval_llk(x, mode='nat'),
                    model1.eval_llk(x, mode='std'))

    u_x = model1.compute_suff_stats(x)
    assert_allclose(model1.eval_llk(x, u_x, mode='nat'),
                    model1.eval_llk(x, mode='std'))
    

def test_elbo():

    model1 = create_pdf()
    model1.initialize()

    x = model1.generate(num_samples)
    sample_weights = 0.5*np.ones((num_samples,))
    
    assert_allclose(model1.elbo(x),
                    np.sum(model1.eval_llk(x, mode='std')))
    assert_allclose(model1.elbo(x, sample_weights=sample_weights),
                    0.5*np.sum(model1.eval_llk(x, mode='std')))
    

def test_eval_logcdf():

    model1 = create_pdf()
    model1.initialize()

    assert(model1.eval_logcdf(model1.mu) == x_dim*np.log(0.5))
    assert(model1.eval_logcdf(1e10*np.ones((x_dim,))) > np.log(0.99))
    assert(model1.eval_logcdf(-1e10*np.ones((x_dim,))) < np.log(0.01))
    
    

def test_fit():

    model1 = create_pdf()
    model1.initialize()

    x = model1.generate(num_samples_train)
    x_val = model1.generate(num_samples)

    model2 = DiagNormal(x_dim=x_dim)
    elbo = model2.fit(x, x_val=x_val)

    assert_allclose(model2.mu, np.mean(x, axis=0))
    assert_allclose(model2.Lambda, 1/np.std(x, axis=0)**2)
    assert_allclose(model1.mu, model2.mu, atol=0.01)
    assert_allclose(model1.Lambda, model2.Lambda, atol=0.01)
    assert_allclose(model1.eta, model2.eta, atol=0.01)
    assert_allclose(model1.A, model2.A, atol=0.02)
    assert_allclose(elbo[1], np.mean(model1.eval_llk(x)), rtol=1e-5)
    assert_allclose(elbo[3], np.mean(model1.eval_llk(x_val)), rtol=1e-4)
    assert_allclose(elbo[1], np.mean(model2.eval_llk(x)), rtol=1e-5)
    assert_allclose(elbo[3], np.mean(model2.eval_llk(x_val)), rtol=1e-4)
    

def test_plot():
    
    model1 = create_pdf()

    model1.plot1D()
    plt.savefig('./tests/data_out/plot_diag_normal_1D.pdf')
    plt.close()

    model1.plot2D()
    plt.savefig('./tests/data_out/plot_diag_normal_2D.pdf')
    plt.close()

    model1.plot3D()
    plt.savefig('./tests/data_out/plot_diag_normal_3D.pdf')
    plt.close()

    model1.plot3D_ellipsoid()
    plt.savefig('./tests/data_out/plot_diag_normal_3De.pdf')
    plt.close()




if __name__ == '__main__':
    pytest.main([__file__])


