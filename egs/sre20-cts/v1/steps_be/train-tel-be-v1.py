#!/usr/bin/env python
"""
Trains Backend for SRE20 tel condition
"""

import sys
import os
import argparse
import time

import numpy as np

from hyperion.helpers import VectorClassReader as VCR
from hyperion.transforms import TransformList, LDA, LNorm
from hyperion.helpers import PLDAFactory as F
from hyperion.utils.utt2info import Utt2Info


def train_be(iv_file, train_list,
             lda_dim,
             plda_type, y_dim, z_dim,
             epochs, ml_md, md_epochs,
             output_path, **kwargs):

    # Read data
    vcr_args = VCR.filter_args(**kwargs)
    vcr = VCR(iv_file, train_list, None, **vcr_args)
    x, class_ids = vcr.read()

    # Train LDA
    t1 = time.time()

    lda = LDA(lda_dim=lda_dim, name='lda')
    lda.fit(x, class_ids)

    x_lda = lda.predict(x)
    print('LDA Elapsed time: %.2f s.' % (time.time()-t1))

    # Train centering and whitening
    t1 = time.time()
    lnorm = LNorm(name='lnorm')    
    lnorm.fit(x_lda)

    x_ln = lnorm.predict(x_lda)
    print('LNorm Elapsed time: %.2f s.' % (time.time()-t1))
    
    # Train PLDA
    t1 = time.time()

    plda = F.create_plda(plda_type, y_dim=y_dim, z_dim=z_dim,
                         name='plda')
    elbo = plda.fit(x_ln, class_ids, 
                    epochs=epochs, ml_md=ml_md, md_epochs=md_epochs)

    print('PLDA Elapsed time: %.2f s.' % (time.time()-t1))

    # Save models
    preproc = TransformList(lda)
    preproc.append(lnorm)

    if not os.path.exists(output_path):
        os.makedirs(ouput_path)

    preproc.save(output_path + '/lda_lnorm.h5')
    plda.save(output_path + '/plda.h5')

    num = np.arange(epochs)
    elbo = np.vstack((num, elbo)).T
    np.savetxt(output_path + '/elbo.csv', elbo, delimiter=',')
 
    
if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Train Back-end for SRE20 telephone condition')

    parser.add_argument('--iv-file', dest='iv_file', required=True)
    parser.add_argument('--train-list', dest='train_list', required=True)
    
    VCR.add_argparse_args(parser)
    F.add_argparse_train_args(parser)
    
    parser.add_argument('--output-path', dest='output_path', required=True)
    parser.add_argument('--lda-dim', dest='lda_dim', type=int,
                        default=150)
    args=parser.parse_args()
    
    train_be(**vars(args))

            
