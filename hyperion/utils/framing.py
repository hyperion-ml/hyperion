"""
Functions to create and manipulate frames
"""

from __future__ import absolute_import
from __future__ import print_function

from six.moves import xrange

import numpy as np

from ..hyp_defs import float_cpu

def create_frames(x,frame_length, frame_shift=1, padding_mode=None, padding_side='symmetric', **kwargs):

    if padding_mode is not None:
        x=apply_padding(x, frame_length, frame_shift, padding_mode, padding_side, **kwargs)

    if x.ndim==1:
        n_samples=x.shape[0]
        in_dim=1
    else:
        n_samples=x.shape[0]
        in_dim=x.shape[1]

    n_out_frames=int(np.floor((n_samples-frame_length)/frame_shift+1))
        
    vec_x=x.ravel()
    out_dim=frame_length*in_dim
    X=np.zeros((n_out_frames, out_dim), dtype=float_cpu())

    start=0
    stop=out_dim
    shift=in_dim*frame_shift
    for i in xrange(n_out_frames):
        X[i,:]=vec_x[start:stop]
        start+=shift
        stop+=shift
    return X


def apply_padding(x, frame_length, frame_shift=1,
                  padding_mode=None, padding_side='symmetric', **kwargs):
    
    if padding_side=='symmetric':
        pad_spec=(int(np.ceil((frame_length-frame_shift)/2)),
                  int(np.floor((frame_length-frame_shift)/2)))
    elif padding_side=='left':
        pad_spec=(int(frame_length-frame_shift), 0)
    elif padding_side=='right':
        pad_spec=(0, int(frame_length-frame_shift))
    else:
        raise Exception('Unknown padding_side=%s' % padding_side)

    if x.ndim==2:
        pad_spec=(pad_spec, (0,0))
    return np.pad(x, pad_spec, mode=padding_mode, **kwargs)
