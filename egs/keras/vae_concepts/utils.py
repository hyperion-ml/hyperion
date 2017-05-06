
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

from keras import initializers
from keras.callbacks import *

from hyperion.keras.callbacks import *

def my_init(shape, dtype=None):
    return K.random_normal(shape, stddev=0.1, dtype=dtype)
    

def my_callbacks():
    lr_dict = { 100: 0.0001 }
    lrs = LearningRateSteps(lr_dict)
    eo = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min')
    return [lrs, eo]

def normalize_x(x):
    n = len(x)
    mean_x=np.mean(x[0],axis=0)
    std_x=np.std(x[0],axis=0)

    for i in xrange(n):
        x[i]=(x[i]-mean_x)/std_x
    return x

def save_xr(f, x, r):
    f.create_dataset('x_train', data=x[0])
    f.create_dataset('x_val', data=x[1])
    f.create_dataset('x_test', data=x[2])
    f.create_dataset('r_train', data=r[0])
    f.create_dataset('r_val', data=r[1])
    f.create_dataset('r_test', data=r[2])

def save_xrt(f, x, r, t):
    f.create_dataset('x_train', data=x[0])
    f.create_dataset('x_val', data=x[1])
    f.create_dataset('x_test', data=x[2])
    f.create_dataset('r_train', data=r[0])
    f.create_dataset('r_val', data=r[1])
    f.create_dataset('r_test', data=r[2])
    f.create_dataset('t_train', data=t[0])
    f.create_dataset('t_val', data=t[1])
    f.create_dataset('t_test', data=t[2])

    
def plot_xr(x, r, file_path):

    plt.figure(figsize=(6, 6))
    plt.scatter(x[:, 0], x[:, 1], c=r)
    plt.colorbar()
    plt.show()
    plt.savefig(file_path)

    
def plot_xzr(x, z, r, file_path):

    K=int(np.max(r))+1
    
    #Colormap stuff
    cm_norm = matplotlib.colors.Normalize(0,K-1)

    # choose a colormap
    c_m = matplotlib.cm.jet

    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=cm_norm)
    s_m.set_array([])

    plt.figure(figsize=(18, 6))
    plt.subplot(1,3,1)
    plt.scatter(x[:, 0], x[:, 1], c=r, cmap=c_m, norm=cm_norm)
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.title('X val')

    nbins=int(x.shape[0]/25)
    plt.subplot(1,3,2)
    plt.hist(z.ravel(), nbins, normed=True, color='k', histtype='step')
    plt.xlim(-3,3)
    plt.title('Z val distribution')
    plt.subplot(1,3,3)
    
    plt.hold(True)
    nbins=int(nbins/K)
    for i in xrange(K):
        plt.hist(z[r==i,:].ravel(), nbins,
                 normed=True, color=s_m.to_rgba(i), histtype='step')
    plt.xlim(-3,3)
    plt.colorbar(s_m)
    plt.title('Z val distribution per component')
    plt.show()
    plt.savefig(file_path)

def plot_xzrt(x, z, r, t, file_path):

    K=int(np.max(r))+1
    M=int(np.max(t))+1

    #Colormap stuff
    cm_norm_r = matplotlib.colors.Normalize(0,K-1)
    cm_norm_t = matplotlib.colors.Normalize(0,M)

    # choose a colormap
    c_m = matplotlib.cm.jet

    # create a ScalarMappable and initialize a data structure
    s_m_r = matplotlib.cm.ScalarMappable(cmap=c_m, norm=cm_norm_r)
    s_m_r.set_array([])
    s_m_t = matplotlib.cm.ScalarMappable(cmap=c_m, norm=cm_norm_t)
    s_m_t.set_array([])

    plt.figure(figsize=(18, 12))
    plt.subplot(2,3,1)
    plt.scatter(x[:, 0], x[:, 1], c=r, cmap=c_m,norm=cm_norm_r)
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.title('X val clusters')
    plt.subplot(2,3,4)
    plt.scatter(x[:, 0], x[:, 1], c=t, cmap=c_m,norm=cm_norm_t)
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.title('X val segments')
    
    nbins=int(x.shape[0]/50)
    plt.subplot(2,3,2)
    plt.hist(z.ravel(), nbins, normed=True, color='k', histtype='step')
    plt.xlim(-3,3)
    plt.title('Z val distribution')
    plt.subplot(2,3,3)
    plt.hold(True)
    nbins_i=int(nbins/K)
    for i in xrange(K):
        plt.hist(z[r==i,:].ravel(), nbins_i,
                 normed=True, color=s_m_r.to_rgba(i), histtype='step')
    plt.xlim(-3,3)
    plt.colorbar(s_m_r)
    plt.title('Z val distribution per cluster')
    plt.subplot(2,3,6)
    plt.hold(True)
    nbins_i=int(nbins_i/M)
    for i in xrange(M):
        plt.hist(z[t==i,:].ravel(), nbins_i,
                 normed=True, color=s_m_t.to_rgba(i), histtype='step')
    plt.xlim(-3,3)
    plt.colorbar(s_m_t)
    plt.title('Z val distribution per segment')

    plt.show()
    plt.savefig(file_path)


def plot_xyt(x, y, t, file_path):
    
    M=int(np.max(t))+1
    
    #Colormap stuff
    cm_norm_t = matplotlib.colors.Normalize(0,M)

    # choose a colormap
    c_m = matplotlib.cm.jet

    # create a ScalarMappable and initialize a data structure
    s_m_t = matplotlib.cm.ScalarMappable(cmap=c_m, norm=cm_norm_t)
    s_m_t.set_array([])

    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.scatter(x[:, 0], x[:, 1], c=t, cmap=c_m, norm=cm_norm_t)
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.title('X val segments')

    nbins=5
    plt.subplot(1,2,2)
    plt.hist(y.ravel(), nbins, normed=True, color='k', histtype='step')
    plt.xlim(-3,3)
    plt.title('Y val distribution')
    plt.show()
    plt.savefig(file_path)
    
def plot_xdecr(x, x_dec, r, file_path):
    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.scatter(x[:, 0], x[:, 1], c=r)
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.title('X val')
    plt.subplot(1,2,2)
    plt.scatter(x_dec[:, 0], x_dec[:, 1], c=r)
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.title('X val encode-decode')
    plt.colorbar()
    plt.show()
    plt.savefig(file_path)


def plot_xdecrt(x, x_dec, r, t, file_path):
    plt.figure(figsize=(12, 12))
    plt.subplot(2,2,1)
    plt.scatter(x[:, 0], x[:, 1], c=r)
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.title('X val clusters')
    plt.subplot(2,2,3)
    plt.scatter(x[:, 0], x[:, 1], c=t)
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.title('X val segments')
    
    plt.subplot(2,2,2)
    plt.scatter(x_dec[:, 0], x_dec[:, 1], c=r)
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.title('X val enc-dec clusters')
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.scatter(x_dec[:, 0], x_dec[:, 1], c=t)
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.title('X val enc-dec segments')
    plt.colorbar()
    plt.show()
    plt.savefig(file_path)


    
def plot_xsample(x, x_s, r, file_path):
    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.scatter(x[:, 0], x[:, 1], c=r)
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.title('X val')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.scatter(x_s[:, 0], x_s[:, 1], c='b')
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.title('X sample')
    plt.show()
    plt.savefig(file_path)

    
def plot_xcsample(x, x_s, r, file_path):
    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.scatter(x[:, 0], x[:, 1], c=r)
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.title('X val')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.scatter(x_s[:, 0], x_s[:, 1], c=r)
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.title('X sample')
    plt.show()
    plt.savefig(file_path)

    
def plot_xsample_t(x, x_sample, file_path):
    cc=['b','r','g','m','c']
    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    for i in xrange(x.shape[0]):
        plt.scatter(x[i,:, 0], x[i, :, 1], c=cc[i])
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.title('X val')
    plt.subplot(1,2,2)
    for i in xrange(x_sample.shape[0]):
        plt.scatter(x_sample[i,:, 0], x_sample[i, :, 1], c=cc[i])
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.title('X sample')
    plt.show()
    plt.savefig(file_path)

def load_xr(file_path):
    with h5py.File(file_path,'r') as f:
        x_train = np.asarray(f['x_train'], dtype='float32')
        x_val = np.asarray(f['x_val'], dtype='float32')
        r_train = np.asarray(f['r_train'], dtype='float32')
        r_val = np.asarray(f['r_val'], dtype='float32')
        
        K = np.asarray(f['K'],dtype='int32')

        return x_train, r_train, x_val, r_val, K


def load_xrt(file_path):
    with h5py.File(file_path,'r') as f:
        x_train = np.asarray(f['x_train'], dtype='float32')
        x_val = np.asarray(f['x_val'], dtype='float32')
        r_train = np.asarray(f['r_train'], dtype='float32')
        r_val = np.asarray(f['r_val'], dtype='float32')
        if 't_train' in f:
            t_train = np.asarray(f['t_train'], dtype='float32')
            t_val = np.asarray(f['t_val'], dtype='float32')
            M = np.asarray(f['M'], dtype='int32')
        else:
            t_train = t_val = None
            M = 0
        
        K = np.asarray(f['K'], dtype='int32')

        return x_train, r_train, t_train, x_val, r_val, t_val, K, M

    
def to_onehot(r, K):
    rr=np.zeros((len(r), K), dtype='float32')
    rr[np.arange(rr.shape[0]), r.astype('int32')]=1
    return rr


def x_2dto3d(x, M):
    N = int(x.shape[0]/M)
    D = x.shape[1]
    return np.reshape(x, (-1, N, D))

def x_3dto2d(x):
    D = x.shape[-1]
    return np.reshape(x, (-1, D))

def save_hist(file_path, h, label):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('label', data=np.asarray(label, dtype='S'))
        f.create_dataset('loss', data=h['loss'])
        f.create_dataset('val_loss', data=h['val_loss'])

def load_hist(file_path):
    with h5py.File(file_path, 'r') as f:
        label = np.asarray(f['label']).astype('U')
        loss = np.asarray(f['loss'], dtype='float32')
        val_loss = np.asarray(f['val_loss'], dtype='float32')
        return label, loss, val_loss
