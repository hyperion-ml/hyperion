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
from hyperion.helpers import VectorReader as VR
from hyperion.pdfs.core import Normal
from hyperion.transforms import TransformList, CentWhiten, LNorm
from hyperion.utils.scp_list import SCPList



def load_data(iv_file, train_file, preproc, scp_sep, v_field):

    train_utt= SCPList.load(train_file, sep='=')
    
    hr = HypDataReader(iv_file)
    x = hr.read(train_utt.file_path, '', return_tensor=True)
    if preproc is not None:
        x = preproc.predict(x)

    return x


def train_cw(iv_file, train_list, preproc_file, with_lnorm,
             scp_sep, v_field,
             name, save_tlist, append_tlist, out_path, **kwargs):
    
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    vr = VR(iv_file, train_list, preproc, scp_sep=scp_sep, v_field=v_field)
    x = vr.read()
    # x = load_data(iv_file, train_list, preproc, scp_sep=scp_sep, v_field=v_filed)
    print(x.shape)
    t1 = time.time()

    gauss = Normal(x_dim=x.shape[1])
    gauss.fit(x=x)

    if with_lnorm:
        model = LNorm(name=name)
    else:
        model = CentWhiten(name=name)

    model.fit(mu=gauss.mu, C=gauss.Sigma)

    print('Elapsed time: %.2f s.' % (time.time()-t1))
    
    x = model.predict(x)
    
    gauss.fit(x=x)
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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Train Centering+Whitening')

    parser.add_argument('--iv-file', dest='iv_file', required=True)
    parser.add_argument('--train-list', dest='train_list', required=True)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)
    
    VR.add_argparse_args(parser)

    parser.add_argument('--out-path', dest='out_path', required=True)
    parser.add_argument('--with-lnorm', dest='with_lnorm', type=bool,
                        default=True)
    parser.add_argument('--save-tlist', dest='save_tlist', type=bool,
                        default=True)
    parser.add_argument('--append-tlist', dest='append_tlist', type=bool,
                        default=True)
    parser.add_argument('--name', dest='name', default='lnorm')
    
    args=parser.parse_args()
    
    train_cw(**vars(args))

            
