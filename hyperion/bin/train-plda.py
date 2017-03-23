#!/usr/bin/env python

"""
Trains Centering and whitening
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import sys
import os
import argparse
import time

import numpy as np

from hyperion.io import HypDataReader
from hyperion.utils.scp_list import SCPList
from hyperion.transforms import TransformList
from hyperion.distributions.plda import *


def load_data(iv_file, train_file, preproc):

    utt2spk= SCPList.load(train_file, sep='=')
    
    hr = HypDataReader(iv_file)
    x = hr.read(utt2spk.file_path, '.ivec')
    if preproc is not None:
        x = preproc.predict(x)

    _, _, class_ids=np.unique(utt2spk.key,
                              return_index=True, return_inverse=True)

    return x, class_ids


def train_plda(iv_file, train_list, val_list, preproc_file, y_dim, z_dim,
               plda_type, epochs, md_epochs,
               name, out_path, **kwargs):
    
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    x, class_ids = load_data(iv_file, train_list, preproc)
    x_val = None
    class_ids_val = None
    if val_list is not None:
        x_val, class_ids_val = load_data(iv_file, val_list, preproc)

    t1 = time.time()

    if plda_type == 'frplda':
        model = FRPLDA()
    elif plda_type == 'splda':
        model = SPLDA(y_dim = y_dim)
    elbos = model.fit(x, class_ids, x_val=x_val, class_ids_val=class_ids_val,
                      epochs=epochs, md_epochs=md_epochs)

    print('Elapsed time: %.2f s.' % (time.time()-t1))
    
    model.save(out_path)

    if len(elbos)==2:
        elbo = np.vstack(elbos)
    num = np.arange(epochs)
    elbo = np.vstack((num, elbo)).T
    elbo_path=os.path.splitext(out_path)[0] + '.csv'
    np.savetxt(elbo_path, elbo, delimiter=',')
    
    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        description='Train LDA')

    parser.add_argument('--iv-file', dest='iv_file', required=True)
    parser.add_argument('--train-list', dest='train_list', required=True)
    parser.add_argument('--val-list', dest='val_list', default=None)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)
    parser.add_argument('--out-path', dest='out_path', required=True)
    parser.add_argument('--y-dim', dest='y_dim', type=int,
                        default=150)
    parser.add_argument('--z-dim', dest='z_dim', type=int,
                        default=None)
    parser.add_argument('--plda-type', dest='plda_type', default='splda')
    parser.add_argument('--epochs', dest='epochs', default=20, type=int)
    parser.add_argument('--md-epochs', dest='md_epochs', default=[1, 9],
                         type=int, nargs = '+')
    parser.add_argument('--name', dest='name', default='plda')
    
    args=parser.parse_args()
    
    train_plda(**vars(args))

            
