'''
VAE for  GMM i-vector distribution
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras import initializations
from keras import optimizers
from keras.regularizers import l2

import h5py
from six.moves import xrange

from cronus.keras.vae.vae import VAE

NG=10
batch_size = 100
original_dim = 2
latent_dim = 1
intermediate_dim = 200
epochs = 100
l2_reg=0.0001
num_samples=10000

def my_init(shape,name=None):
    return initializations.normal(shape, scale=0.1, name=name)

# define encoder architecture
x = Input(shape=(original_dim,))
h1 = Dense(intermediate_dim, activation='relu', init=my_init, W_regularizer=l2(l2_reg))(x)
h2 = Dense(intermediate_dim, activation='relu', init=my_init, W_regularizer=l2(l2_reg))(h1)
z_mean = Dense(latent_dim, init=my_init, W_regularizer=l2(l2_reg))(h2)
z_logvar = Dense(latent_dim, init=my_init, W_regularizer=l2(l2_reg))(h2)

encoder=Model(x,[z_mean, z_logvar])

# define decoder architecture
z=Input(shape=(latent_dim,))
h1_dec = Dense(intermediate_dim, activation='relu', init=my_init, W_regularizer=l2(l2_reg))(z)
h2_dec = Dense(intermediate_dim, activation='relu', init=my_init, W_regularizer=l2(l2_reg))(h1_dec)
x_dec_mean = Dense(original_dim, init=my_init, W_regularizer=l2(l2_reg))(h2_dec)
x_dec_logvar = Dense(original_dim, init=my_init, W_regularizer=l2(l2_reg))(h2_dec)

decoder=Model(z,[x_dec_mean, x_dec_logvar])


# load data
f=h5py.File('data.h5','r')
x_train=np.asarray(f['x_train'],dtype='float32')
x_val=np.asarray(f['x_val'],dtype='float32')
y_train=np.asarray(f['y_train'],dtype='float32')
y_val=np.asarray(f['y_val'],dtype='float32')
t_train=np.asarray(f['t_train'],dtype='float32')
t_val=np.asarray(f['t_val'],dtype='float32')

# normalize data
mean_x=np.mean(x_train,axis=0)
std_x=np.std(x_train,axis=0)
x_train=(x_train-mean_x)/std_x
x_val=(x_val-mean_x)/std_x

M_val=int(np.max(t_val))

# train VAE
vae=VAE(encoder,decoder,'normal')
vae.build()
opt = optimizers.Adam(lr=0.001)
vae.train(x_train,x_val=x_val,optimizer=opt,
          shuffle=True,
          epochs=epochs,
          batch_size=batch_size)


#Colormap stuff
cm_norm_y = matplotlib.colors.Normalize(0,NG-1)
cm_norm_t = matplotlib.colors.Normalize(0,M_val)

# choose a colormap
c_m = matplotlib.cm.jet

# create a ScalarMappable and initialize a data structure
s_m_y = matplotlib.cm.ScalarMappable(cmap=c_m, norm=cm_norm_y)
s_m_y.set_array([])
s_m_t = matplotlib.cm.ScalarMappable(cmap=c_m, norm=cm_norm_t)
s_m_t.set_array([])


# plot the latent space
z_val = vae.compute_qz_x(x_val, batch_size=batch_size)[0]
plt.figure(figsize=(18, 12))
plt.subplot(2,3,1)
plt.scatter(x_val[:, 0], x_val[:, 1], c=y_val,cmap=c_m,norm=cm_norm_y)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('X val clusters')
plt.subplot(2,3,4)
plt.scatter(x_val[:, 0], x_val[:, 1], c=t_val,cmap=c_m,norm=cm_norm_t)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('X val segments')

#plt.colorbar()
if latent_dim==2:
    # 2D scatter plot
    plt.subplot(1,2,2)
    plt.scatter(z_val[:, 0], z_val[:, 1], c=y_val)
    plt.colorbar()
else:
    nbins=int(x_val.shape[0]/50)
    plt.subplot(2,3,2)
    plt.hist(z_val.ravel(),nbins,normed=True,color='k',histtype='step')
    plt.xlim(-3,3)
    plt.title('Z val distribution')
    plt.subplot(2,3,3)
    plt.hold(True)
    nbins=int(nbins/NG)
    for i in xrange(NG):
        plt.hist(z_val[y_val==i,:].ravel(),nbins,
                 normed=True,color=s_m_y.to_rgba(i),histtype='step')
    plt.xlim(-3,3)
    plt.colorbar(s_m_y)
    plt.title('Z val distribution per cluster')
    plt.subplot(2,3,6)
    plt.hold(True)
    nbins=int(nbins/M_val)
    for i in xrange(M_val):
        plt.hist(z_val[t_val==i,:].ravel(),nbins,
                 normed=True,color=s_m_t.to_rgba(i),histtype='step')
    plt.xlim(-3,3)
    plt.colorbar(s_m_t)
    plt.title('Z val distribution per segment')

plt.show()
plt.savefig('vae_z_val.pdf')

# decode z_val
x_val_dec = vae.decode_z(z_val, batch_size=batch_size)
plt.figure(figsize=(12, 12))
plt.subplot(2,2,1)
plt.scatter(x_val[:, 0], x_val[:, 1], c=y_val)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('X val clusters')
plt.subplot(2,2,3)
plt.scatter(x_val[:, 0], x_val[:, 1], c=t_val)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('X val segments')

plt.subplot(2,2,2)
plt.scatter(x_val_dec[:, 0], x_val_dec[:, 1], c=y_val)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('X val enc-dec clusters')
plt.colorbar()
plt.subplot(2,2,4)
plt.scatter(x_val_dec[:, 0], x_val_dec[:, 1], c=t_val)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('X val enc-dec segments')
plt.colorbar()
plt.show()
plt.savefig('vae_x_val_dec.pdf')


# Sample x from VAE
x_sample = vae.sample_x(num_samples, batch_size=batch_size)
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.scatter(x_val[:, 0], x_val[:, 1], c=y_val)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('X val')
plt.colorbar()
plt.subplot(1,2,2)
plt.scatter(x_sample[:, 0], x_sample[:, 1], c='b')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('X sample')
plt.show()
plt.savefig('vae_sample.pdf')
