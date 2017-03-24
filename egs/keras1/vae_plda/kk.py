'''
VAE for  GMM i-vector distribution
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, Merge
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras import initializations
from keras import optimizers
from keras.regularizers import l2

import h5py
from six.moves import xrange

from cronus.keras.vae.tied_vae_qyqzgy import TiedVAE_qYqZgY as TVAE

D=2
NG=10
batch_size = 10
original_dim = 2
latent_dim = 1
intermediate_dim = 200
nb_epoch = 500
l2_reg=0.0001
nb_samples=1000
N_i=1000

def my_init(shape,name=None):
    return initializations.normal(shape, scale=0.1, name=name)

# define encoder architecture
# define q(y|x)
x = Input(shape=(N_i,original_dim,))
h1_y = TimeDistributed(Dense(intermediate_dim, activation='relu',
                           init=my_init, W_regularizer=l2(l2_reg)))(x)
h2_y = TimeDistributed(Dense(int(intermediate_dim/2), activation='relu',
                           init=my_init, W_regularizer=l2(l2_reg)))(h1_y)
h2_y_pool = GlobalAveragePooling1D()(h2_y)
y_mean = Dense(latent_dim, init=my_init, W_regularizer=l2(l2_reg))(h2_y_pool)
y_logvar = Dense(latent_dim, init=my_init, W_regularizer=l2(l2_reg))(h2_y_pool)
qy=Model(x,[y_mean, y_logvar])

# define q(z|x,y)
y1=Input(shape=(N_i, latent_dim,))
xy = Merge(mode='concat', concat_axis=-1)([x, y1])
h1_z = TimeDistributed(Dense(intermediate_dim, activation='relu',
                           init=my_init, W_regularizer=l2(l2_reg)))(xy)
h2_z = TimeDistributed(Dense(int(intermediate_dim/2), activation='relu',
                           init=my_init, W_regularizer=l2(l2_reg)))(h1_z)
z_mean = TimeDistributed(Dense(latent_dim, init=my_init, W_regularizer=l2(l2_reg)))(h2_z)
z_logvar = TimeDistributed(Dense(latent_dim, init=my_init, W_regularizer=l2(l2_reg)))(h2_z)
qz=Model([x, y1],[z_mean, z_logvar])



# define decoder architecture
y=Input(shape=(N_i, latent_dim,))
z=Input(shape=(N_i, latent_dim,))
yz = Merge(mode='concat', concat_axis=-1)([y, z])
h1_dec = TimeDistributed(Dense(intermediate_dim, activation='relu',
                               init=my_init, W_regularizer=l2(l2_reg)))(yz)
h2_dec = TimeDistributed(Dense(intermediate_dim, activation='relu',
                               init=my_init, W_regularizer=l2(l2_reg)))(h1_dec)
x_dec_mean = TimeDistributed(Dense(original_dim, init=my_init, W_regularizer=l2(l2_reg)))(h2_dec)
x_dec_logvar = TimeDistributed(Dense(original_dim, init=my_init, W_regularizer=l2(l2_reg)))(h2_dec)

decoder=Model([y, z],[x_dec_mean, x_dec_logvar])


# load data
f=h5py.File('data.h5','r')

x_train=np.asarray(f['x_train'],dtype='float32')
x_val=np.asarray(f['x_val'],dtype='float32')
k_train=np.asarray(f['y_train'],dtype='float32')
k_val=np.asarray(f['y_val'],dtype='float32')
t_train=np.asarray(f['t_train'],dtype='float32')
t_val=np.asarray(f['t_val'],dtype='float32')

# normalize data
mean_x=np.mean(x_train,axis=0)
std_x=np.std(x_train,axis=0)
x_train=(x_train-mean_x)/std_x
x_val=(x_val-mean_x)/std_x

# 2D to 3D tensor
x_train=np.reshape(x_train,(-1,N_i,D))
x_val=np.reshape(x_val,(-1,N_i,D))

M_val=int(np.max(t_val))+1

# train VAE
vae=TVAE(qy,qz,decoder,'normal')
vae.build()
opt = optimizers.Adam(lr=0.001)
vae.train(x_train,x_val=x_val,optimizer=opt,
          shuffle=True,
          nb_epoch=nb_epoch,
          batch_size=batch_size)


#Colormap stuff
cm_norm_k = matplotlib.colors.Normalize(0,NG-1)
cm_norm_t = matplotlib.colors.Normalize(0,M_val)

# choose a colormap
c_m = matplotlib.cm.jet

# create a ScalarMappable and initialize a data structure
s_m_k = matplotlib.cm.ScalarMappable(cmap=c_m, norm=cm_norm_k)
s_m_k.set_array([])
s_m_t = matplotlib.cm.ScalarMappable(cmap=c_m, norm=cm_norm_t)
s_m_t.set_array([])


# plot the latent space
yz = vae.compute_qyz_x(x_val, batch_size=batch_size)
y_val=yz[0]
z_val=yz[2]

x_val2=np.reshape(x_val,(-1,D))
y_val2=np.reshape(y_val,(-1,1))
z_val2=np.reshape(z_val,(-1,1))

plt.figure(figsize=(18, 12))
plt.subplot(2,3,1)
plt.scatter(x_val2[:, 0], x_val2[:, 1], c=k_val, cmap=c_m,norm=cm_norm_k)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('X val clusters')
plt.subplot(2,3,4)
plt.scatter(x_val2[:, 0], x_val2[:, 1], c=t_val, cmap=c_m,norm=cm_norm_t)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('X val segments')

#plt.colorbar()
if latent_dim==2:
    # 2D scatter plot
    plt.subplot(1,2,2)
    plt.scatter(z_val2[:, 0], z_val2[:, 1], c=k_val)
    plt.colorbar()
else:
    nbins=int(x_val2.shape[0]/50)
    plt.subplot(2,3,2)
    plt.hist(z_val2.ravel(),nbins,normed=True,color='k',histtype='step')
    plt.xlim(-3,3)
    plt.title('Z val distribution')
    plt.subplot(2,3,3)
    plt.hold(True)
    nbins_i=int(nbins/NG)
    for i in xrange(NG):
        plt.hist(z_val2[k_val==i,:].ravel(),nbins_i,
                 normed=True,color=s_m_k.to_rgba(i),histtype='step')
    plt.xlim(-3,3)
    plt.colorbar(s_m_k)
    plt.title('Z val distribution per cluster')
    plt.subplot(2,3,6)
    plt.hold(True)
    nbins_i=int(nbins_i/M_val)
    for i in xrange(M_val):
        plt.hist(z_val2[t_val==i,:].ravel(),nbins_i,
                 normed=True,color=s_m_t.to_rgba(i),histtype='step')
    plt.xlim(-3,3)
    plt.colorbar(s_m_t)
    plt.title('Z val distribution per segment')

plt.show()
plt.savefig('tied_vae_qyqzgy_z_val.pdf')


plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.scatter(x_val2[:, 0], x_val2[:, 1], c=t_val,cmap=c_m,norm=cm_norm_t)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('X val segments')

nbins=5
plt.subplot(1,2,2)
plt.hist(y_val2.ravel(),nbins,normed=True,color='k',histtype='step')
plt.xlim(-3,3)
plt.title('Y val distribution')
plt.show()
plt.savefig('tied_vae_qyqzgy_y_val.pdf')


# decode z_val
x_val_dec = vae.decode_yz(y_val, z_val, batch_size=batch_size)
x_val_dec2=np.reshape(x_val_dec, (-1, D))
plt.figure(figsize=(12, 12))
plt.subplot(2,2,1)
plt.scatter(x_val2[:, 0], x_val2[:, 1], c=k_val)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('X val clusters')
plt.subplot(2,2,3)
plt.scatter(x_val2[:, 0], x_val2[:, 1], c=t_val)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('X val segments')

plt.subplot(2,2,2)
plt.scatter(x_val_dec2[:, 0], x_val_dec2[:, 1], c=k_val)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('X val enc-dec clusters')
plt.colorbar()
plt.subplot(2,2,4)
plt.scatter(x_val_dec2[:, 0], x_val_dec2[:, 1], c=t_val)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('X val enc-dec segments')
plt.colorbar()
plt.show()
plt.savefig('tied_vae_qyqzgy_x_val_dec.pdf')


# Sample x from VAE

x_sample = vae.sample_x(M_val, nb_samples, batch_size=batch_size)
x_sample2 = np.reshape(x_sample, (-1,D))
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.scatter(x_val2[:, 0], x_val2[:, 1], c=k_val)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('X val')
plt.colorbar()
plt.subplot(1,2,2)
plt.scatter(x_sample2[:, 0], x_sample2[:, 1], c='b')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('X sample')
plt.show()
plt.savefig('tied_vae_qyqzgy_sample.pdf')


y_sample=np.expand_dims(np.expand_dims([-2,0,2], axis=1), axis=1)
x_sample = vae.sample_x_g_y(y_sample, nb_samples, batch_size=batch_size)
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.scatter(x_val[0,:, 0], x_val[0, :, 1], c='b')
plt.scatter(x_val[1,:, 0], x_val[1, :, 1], c='r')
plt.scatter(x_val[4,:, 0], x_val[4, :, 1], c='g')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('X val')
plt.subplot(1,2,2)
plt.scatter(x_sample[0, :, 0], x_sample[0, :, 1], c='b')
plt.scatter(x_sample[1, :, 0], x_sample[1, :, 1], c='r')
plt.scatter(x_sample[2, :, 0], x_sample[2, :, 1], c='g')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title('X sample')
plt.show()
plt.savefig('tied_vae_qyqzgy_sample_t3.pdf')
