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
from hyperion.distributions.core import Normal
from hyperion.transforms import TransformList, CentWhiten, LNorm
from hyperion.utils.scp_list import SCPList



def load_data(iv_file, train_file, preproc):

    train_utt= SCPList.load(train_file, sep='=')
    
    hr = HypDataReader(iv_file)
    x = hr.read(train_utt.file_path, '.ivec')
    if preproc is not None:
        x = preproc.predict(x)

    return x


def train_cw(iv_file, train_list, preproc_file, with_lnorm, batch_size,
             tname, save_tlist, append_tlist, out_path, **kwargs):
    
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    x = load_data(iv_file, train_list, preproc)

    t1 = time.time()

    gauss = Normal(x_dim=x.shape[1])
    gauss.fit(x_train=x)

    if with_lnorm:
        model = LNorm(name=tname)
    else:
        model = CentWhiten(name=tname)

    model.fit(mu=gauss.mu, C=gauss.Sigma)

    print('Elapsed time: %.2f s.' % (time.time()-t1))
    
    x = model.predict(x)
    
    gauss.fit(x_train=x)
    print(gauss.mu[:4])
    print(gauss.Sigma[:4,:4])

    if save_tlist:
        if append_tlist and preproc is not None:
            preproc.append(model)
            model = preproc
        else:
            model = TransformList(model)

    model.save(out_path)
        
    
    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        description='Train Centering+Whitening')

    parser.add_argument('--iv-file', dest='iv_file', required=True)
    parser.add_argument('--train-list', dest='train_list', required=True)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)
    parser.add_argument('--out-path', dest='out_path', required=True)
    parser.add_argument('--with-lnorm', dest='with_lnorm', type=bool,
                        default=True)
    parser.add_argument('--save-tlist', dest='save_tlist', type=bool,
                        default=True)
    parser.add_argument('--append-tlist', dest='append_tlist', type=bool,
                        default=True)
    parser.add_argument('--batch-size',dest='batch_size',default=1024,type=int,
                        help=('Batch size (default: %(default)s)'))
    parser.add_argument('--tname', dest='tname', default='lnorm')
    
    args=parser.parse_args()
    
    train_cw(**vars(args))

            
