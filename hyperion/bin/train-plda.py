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
from hyperion.helpers import VectorClassReader as VCR
from hyperion.helpers import PLDAFactory as F
from hyperion.transforms import TransformList



def load_data(iv_file, train_file, preproc):

    utt2spk= SCPList.load(train_file, sep='=')
    
    hr = HypDataReader(iv_file)
    x = hr.read(utt2spk.file_path, '.ivec', return_tensor=True)
    if preproc is not None:
        x = preproc.predict(x)

    _, _, class_ids=np.unique(utt2spk.key,
                              return_index=True, return_inverse=True)

    return x, class_ids


def train_plda(iv_file, train_list, val_list, preproc_file,
               scp_sep, v_field,
               min_spc, max_spc, spc_pruning_mode,
               csplit_min_spc, csplit_max_spc, csplit_mode,
               csplit_overlap, vcr_seed, 
               plda_type, y_dim, z_dim,
               fullcov_W,
               update_mu, update_V, update_B, update_W,
               name, epochs, ml_md, md_epochs,
               out_path, **kwargs):
    
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    # x, class_ids = load_data(iv_file, train_list, preproc)
    # if val_list is not None:
    #     x_val, class_ids_val = load_data(iv_file, val_list, preproc)

    vcr_train = VCR(iv_file, train_list, preproc,
                    scp_sep=scp_sep, v_field=v_field,
                    min_spc=min_spc, max_spc=max_spc, spc_pruning_mode=spc_pruning_mode,
                    csplit_min_spc=csplit_min_spc, csplit_max_spc=csplit_max_spc,
                    csplit_mode=csplit_mode,
                    csplit_overlap=csplit_overlap, seed=vcr_seed)
    x, class_ids = vcr_train.read()

    x_val = None
    class_ids_val = None
    if val_list is not None:
        vcr_val = VCR(iv_file, val_list, preproc,
                      scp_sep=scp_sep, v_field=v_field,
                      min_spc=min_spc, max_spc=max_spc, spc_pruning_mode=spc_pruning_mode,
                      csplit_min_spc=csplit_min_spc, csplit_max_spc=csplit_max_spc,
                      csplit_mode=csplit_mode,
                      csplit_overlap=csplit_overlap, seed=vcr_seed)
        x_val, class_ids_val = vcr_val.read()
        
    t1 = time.time()

    # if plda_type == 'frplda':
    #     model = FRPLDA()
    # elif plda_type == 'splda':
    #     model = SPLDA(y_dim = y_dim)
    model = F.create_plda(plda_type, y_dim=y_dim, z_dim=z_dim, fullcov_W=fullcov_W,
                          update_mu=update_mu, update_V=update_V,
                          update_B=update_B, update_W=update_W,
                          name=name)
    elbos = model.fit(x, class_ids, x_val=x_val, class_ids_val=class_ids_val,
                      epochs=epochs, ml_md=ml_md, md_epochs=md_epochs)

    print('Elapsed time: %.2f s.' % (time.time()-t1))
    
    model.save(out_path)

    elbo = np.vstack(elbos)
    num = np.arange(epochs)
    elbo = np.vstack((num, elbo)).T
    elbo_path=os.path.splitext(out_path)[0] + '.csv'
    np.savetxt(elbo_path, elbo, delimiter=',')
    
    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Train PLDA')

    parser.add_argument('--iv-file', dest='iv_file', required=True)
    parser.add_argument('--train-list', dest='train_list', required=True)
    parser.add_argument('--val-list', dest='val_list', default=None)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)

    VCR.add_argparse_args(parser)
    F.add_argparse_train_args(parser)

    parser.add_argument('--out-path', dest='out_path', required=True)
    
    args=parser.parse_args()
    
    train_plda(**vars(args))

            
