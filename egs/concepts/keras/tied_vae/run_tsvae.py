#!/usr/bin/env python

'''
Run Tied conditional VAE with Q(y,Z)=Q(y)\prod Q(z_i|y)
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from keras.layers import Input, Dense

from hyperion.utils.math import int2onehot
from hyperion.keras.helpers import OptimizerFactory as KOF
from hyperion.keras.helpers import CallbacksFactory as KCF
from hyperion.keras.archs import TDNNV1TransposeWithEmbedInputV1, TDNNV1WithEmbedInputV1, TDNNV1, FFNetV1
from hyperion.keras.embed import SeqEmbed
from hyperion.keras.vae import TiedSupVAE_QYQZgY
from hyperion.transforms import PCA

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
markers = ['x', 'o', '+', '*', 's', 'h', 'D', '^', 'v', 'p', '8']

num_classes = 10
num_seqs_pc = 20
seq_length = 100
x_dim = 10
y_dim = 2
z_dim = 2
pool_dim = 50
h_dim = 20
kernel = 1


def generate_data():

    rng = np.random.RandomState(seed=1024)
    
    G = TDNNV1TransposeWithEmbedInputV1(
        2, 1, x_dim, h_dim, h_dim, z_dim, y_dim,
        kernel, 1,
        use_batchnorm = False,
        cat_embed_to_all_fc_layers = True,
        name='gen-1')
    weights = G.get_weights()
    
    #print(weights)
    for i in xrange(len(weights)):
        print(weights[i].ndim)
        if weights[i].ndim > 1:
            weights[i] = rng.normal(size=weights[i].shape).astype('float32')
            idx = np.abs(weights[i]) < np.mean(np.abs(weights[i]))
            weights[i][idx] = 0
            weights[i] = 0.1*weights[i]
        
    print(weights)
    
    #G.compile('adam')


    z = 0.1*rng.normal(size=(
        num_seqs_pc*num_classes, seq_length, z_dim)).astype('float32')
    z_val = 0.1*rng.normal(size=(
        num_seqs_pc*num_classes, seq_length, z_dim)).astype('float32')
    y = rng.normal(size=(num_classes, y_dim)).astype('float32')
    y = np.repeat(y, num_seqs_pc, axis=0)
    y = np.expand_dims(y, axis=1)
    y = np.repeat(y, seq_length, axis=1)

    # V1 = rng.normal(size=(2,10)).astype('float32')
    # V2 = 0.5*rng.normal(size=(2,10)).astype('float32')
    # V3 = 0.25*rng.normal(size=(2,10)).astype('float32')
    # U1 = rng.normal(size=(2,10)).astype('float32')
    # U2 = 0.5*rng.normal(size=(2,10)).astype('float32')
    # U3 = 0.25*rng.normal(size=(2,10)).astype('float32')
    # x = (np.dot(y,V1) + np.dot(z,U1) +
    #      np.dot(y*y,V2) + np.dot(z*z,U2) +
    #      np.dot(y*y*y,V3) + np.dot(z*z*z,U3))

    # x_val = (np.dot(y,V1) + np.dot(z_val,U1) +
    #          np.dot(y*y,V2) + np.dot(z_val*z_val,U2) +
    #          np.dot(y*y*y,V3) + np.dot(z_val*z_val*z_val,U3))

    x = G.predict([z, y], batch_size=100)
    x_val = G.predict([z_val, y], batch_size=100)

    t = np.arange(num_classes, dtype=int)
    t = np.repeat(t, num_seqs_pc)
    t = int2onehot(t, num_classes)

    return x, y, z, t, x_val, z_val


def plot_gen_data(x, y, t, nc=10):

    if not os.path.exists('./exp/gen_data'):
        os.makedirs('./exp/gen_data')
    color_marker = [(c,m) for m in markers for c in colors]
    plt.figure()
    for c in xrange(nc):
        idx = t[:,c]==1
        plt.scatter(y[idx,0,0], y[idx,0,1],
                    c=color_marker[c][0], marker=color_marker[c][1])
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.grid(True)
    plt.show()
    plt.savefig('./exp/gen_data/y.pdf')
    plt.close()

    plt.figure()
    for c in xrange(nc):
        idx = t[:,c]==1
        plt.scatter(x[idx,:,0], x[idx,:,1],
                    c=color_marker[c][0], marker=color_marker[c][1])
    #plt.xlim(-2,2)
    #plt.ylim(-2,2)
    plt.grid(True)
    plt.show()
    plt.savefig('./exp/gen_data/x.pdf')
    plt.close()

    pca = PCA(pca_dim=2)
    xa = np.reshape(x, (-1, x_dim))
    pca.fit(xa)
    x_pca = pca.predict(xa)
    x_pca = np.reshape(x_pca, (-1, seq_length,2))

    plt.figure()
    for c in xrange(nc):
        idx = t[:,c]==1
        plt.scatter(x_pca[idx,:,0], x_pca[idx,:,1],
                    c=color_marker[c][0], marker=color_marker[c][1])
    #plt.xlim(-2,2)
    #plt.ylim(-2,2)
    plt.grid(True)
    plt.show()
    plt.savefig('./exp/gen_data/x_pca.pdf')
    plt.close()

    return

    xa = xa - np.mean(xa, axis=0, keepdims=True)
    xa = xa/np.std(xa, axis=0, keepdims=True)
    tsne =TSNE(n_components=2,
               perplexity=30, early_exaggeration=12,
               learning_rate=200, n_iter=1000, init='pca', verbose=1)
    x_tsne = tsne.fit_transform(xa)
    x_tsne = np.reshape(x_tsne, (-1, seq_length,2))


    plt.figure()
    for c in xrange(nc):
        idx = t[:,c]==1
        plt.scatter(x_tsne[idx,:,0], x_tsne[idx,:,1],
                    c=color_marker[c][0], marker=color_marker[c][1])
    #plt.xlim(-2,2)
    #plt.ylim(-2,2)
    plt.grid(True)
    plt.show()
    plt.savefig('./exp/gen_data/x_tsne.pdf')
    plt.close()


    
def train_xvector(x, t, x_val, t_val):

    net1, context = TDNNV1(3,1, pool_dim, h_dim, h_dim, x_dim, kernel,
                           use_batchnorm = True, name='emb-1', return_context=True)
    net2 = FFNetV1(3, num_classes, y_dim, pool_dim*2, output_activation='softmax',
                   use_batchnorm = True, name='emb-2')

    seq_length = x.shape[1]
    model = SeqEmbed(net1, net2, left_context=context, rigth_context=context)
    model.build(seq_length)

    cb = KCF.create_callbacks(model, './exp/train_xvector', monitor='val_loss', patience=20, min_delta=1e-3)
    opt = KOF.create_optimizer('adam', lr=0.001, beta_1=0.9, beta_2=0.99, amsgrad=False)
    model.compile(metrics=['accuracy'], optimizer=opt)
    model.fit(x=x, y=t, validation_data=(x_val, t_val), batch_size=64, epochs=1000,callbacks=cb)

    return model


def train_tvae(x, t, x_val, t_val, w, lr):

    qy_net, context = TDNNV1(3, 1, [y_dim, y_dim], h_dim, pool_dim, x_dim, kernel,
                             use_batchnorm=True, name='qy', return_context=True)
    qz_net = TDNNV1WithEmbedInputV1(1, 1, [z_dim, z_dim], h_dim, h_dim, x_dim, y_dim, 1,
                                    padding='same',
                                    use_batchnorm=True, name='qz')

    
    pt_net = FFNetV1(3, num_classes, y_dim, y_dim, output_activation='softmax',
                     use_batchnorm=True, name='pt')
    px_net = TDNNV1TransposeWithEmbedInputV1(3, 1, [x_dim, x_dim], h_dim, h_dim, z_dim, y_dim, kernel,
                                             padding='same', 
                                             use_batchnorm=True, name='px')
    
    seq_length = x.shape[1]
    model = TiedSupVAE_QYQZgY(px_net, qy_net, qz_net, pt_net,
                              qy_pool_in_fmt='mean+logitvar',
                              qz_fmt='mean+logitvar',
                              frame_corr_penalty=0.2,
                              left_context=context, rigth_context=context,
                              px_weight=w, pt_weight=1,
                              kl_qy_weight=w, kl_qz_weight=w)
    model.build(seq_length)

    model_dir='./exp/train_tvae_pxw%.7f_lr%.7f' % (w,lr)
    cb = KCF.create_callbacks(model, model_dir, monitor='val_loss', patience=30, min_delta=1e-3)
    opt = KOF.create_optimizer('adam', lr=lr, beta_1=0.9, beta_2=0.99, amsgrad=False)
    model.compile(optimizer=opt)
    model.fit(x, t=t, x_val=x_val, t_val=t_val, batch_size=64, epochs=2000, callbacks=cb)
    
    return model

    

def run():

    x, y, z, t, x_val, z_val = generate_data()
    plot_gen_data(x, y, t)
    xv_model = train_xvector(x, t, x_val, t)
    tvae_models1 = []
    for i in xrange(8):
        tvae_model_i = train_tvae(x, t, x_val, t, w=10**(-i), lr=0.001)
        tvae_models1.append(tvae_model_i)
    tvae_models2 = []
    for i in xrange(7):
        tvae_model_i = train_tvae(x, t, x_val, t, w=10**(-i), lr=0.0001)
        tvae_models2.append(tvae_model_i)
    tvae_models3 = []
    for i in xrange(6):
        tvae_model_i = train_tvae(x, t, x_val, t, w=10**(-i), lr=0.00001)
        tvae_models3.append(tvae_model_i)
    tvae_models4 = []
    for i in xrange(5):
        tvae_model_i = train_tvae(x, t, x_val, t, w=10**(-i), lr=0.000001)
        tvae_models4.append(tvae_model_i)
    tvae_models5 = []
    for i in xrange(4):
        tvae_model_i = train_tvae(x, t, x_val, t, w=10**(-i), lr=0.0000001)
        tvae_models5.append(tvae_model_i)

    # tvae_model2 = train_tvae(x, t, x_val, t, w=0.1, lr=0.001)
    # tvae_model3 = train_tvae(x, t, x_val, t, w=0., lr=0.001)
    # tvae_model3 = train_tvae(x, t, x_val, t, w=0.0000001, lr=0.001)
    


if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        description='Runs VAE')

    # parser.add_argument('--exp', dest='exp', required=True)
    # args=parser.parse_args()

    #run(args.exp)
    run()
