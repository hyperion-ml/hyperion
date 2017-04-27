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
from hyperion.transforms import TransformList, MVN, SbSw
from hyperion.utils.scp_list import SCPList

class_ids=[]

def load_data(iv_file, train_file, preproc):

    train_utt= SCPList.load(train_file, sep='=')
    
    hr = HypDataReader(iv_file)
    x = hr.read(train_utt.file_path, '.ivec', return_tensor=True)
    if preproc is not None:
        x = preproc.predict(x)

    global class_ids
    _, _, class_ids=np.unique(train_utt.key,
                              return_index=True, return_inverse=True)
        
    return x


def train_mvn(iv_file, train_list, preproc_file,
              scp_sep, v_field,
              name, save_tlist, append_tlist, out_path, **kwargs):
    
    if preproc_file is not None:
        preproc = TransformList.load(preproc_file)
    else:
        preproc = None

    vr = VR(iv_file, train_list, preproc, scp_sep=scp_sep, v_field=v_field)
    x = vr.read()
    # x = load_data(iv_file, train_list, preproc)

    t1 = time.time()

    model = MVN(name=name)

    model.fit(x)

    print('Elapsed time: %.2f s.' % (time.time()-t1))
    
    x = model.predict(x)

    s_mat = SbSw()
    s_mat.fit(x, class_ids)
    print(s_mat.Sb[:4,:4])
    print(s_mat.Sw[:4,:4])

    
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
        description='Train Global Mean and Variance Normalization')

    parser.add_argument('--iv-file', dest='iv_file', required=True)
    parser.add_argument('--train-list', dest='train_list', required=True)
    parser.add_argument('--preproc-file', dest='preproc_file', default=None)
    VR.add_argparse_args(parser)

    parser.add_argument('--out-path', dest='out_path', required=True)
    parser.add_argument('--save-tlist', dest='save_tlist', type=bool,
                        default=True)
    parser.add_argument('--append-tlist', dest='append_tlist', type=bool,
                        default=True)
    parser.add_argument('--name', dest='name', default='mvn')
    
    args=parser.parse_args()
    
    train_mvn(**vars(args))

            
