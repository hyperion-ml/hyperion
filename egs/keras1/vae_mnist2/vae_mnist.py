'''
MNIST VAE
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras import initializations
from keras import optimizers
from keras.regularizers import l2

from scipy.stats import norm

from hyperion.keras.vae import VAE

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 500
nb_epoch = 200
l2_reg=0.001

def my_init(shape,name=None):
    return initializations.normal(shape, scale=0.1, name=name)

# enconder
x = Input(shape=(original_dim,))
h1 = Dense(intermediate_dim, activation='relu', init=my_init, W_regularizer=l2(l2_reg))(x)
h2 = Dense(intermediate_dim, activation='relu', init=my_init, W_regularizer=l2(l2_reg))(h1)
z_mean = Dense(latent_dim, init=my_init, W_regularizer=l2(l2_reg))(h2)
z_logvar = Dense(latent_dim, init=my_init, W_regularizer=l2(l2_reg))(h2)

encoder=Model(x,[z_mean, z_logvar])

# decoder
z=Input(shape=(latent_dim,))

decoder_h1 = Dense(intermediate_dim, activation='relu', init=my_init, W_regularizer=l2(l2_reg))
decoder_h2 = Dense(intermediate_dim, activation='relu', init=my_init, W_regularizer=l2(l2_reg))
decoder_mean = Dense(original_dim, activation='sigmoid', init=my_init, W_regularizer=l2(l2_reg))
h1_decoded = decoder_h1(z)
h2_decoded = decoder_h2(h1_decoded)
x_decoded_mean = decoder_mean(h2_decoded)

decoder=Model(z,x_decoded_mean)

# load MNIST digits data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# train VAE
vae=VAE(encoder, decoder, px_cond_form='bernoulli')
vae.build()
opt = optimizers.Adam(lr=0.001)
vae.fit(x_train,x_val=x_test,optimizer=opt,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size)

x_test_encoded = vae.compute_qz_x(x_test, batch_size=batch_size)[0]
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()
plt.savefig('vae_latent.pdf')

# display a 2D manifold of the digits
n = 20  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# we will sample n points within [-15, 15] standard deviations
grid_x = np.linspace(0, 1, n)
grid_y = np.linspace(0, 1, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[norm.ppf(xi), norm.ppf(yi)]])
        x_decoded = vae.compute_px_z(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()
plt.savefig('vae_manifold.pdf')
