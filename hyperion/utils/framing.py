"""
Functions to create frames
"""

from __future__ import absolute_import
from __future__ import print_function
from six.moves import xrange

import numpy as np

from ..hyp_defs import float_cpu


class Framing(object):

    def __init__(self, frame_length, frame_shift=1,
                 pad_mode=None, pad_side='symmetric', **kwargs):
        self.frame_length = frame_lenght
        self.frame_shift = frame_shift
        self.pad_mode = pad_mode
        self.pad_width = None
        if self.pad_mode is not None:
            self.pad_width = self.create_pad_width(
                pad_side, frame_length, frame_shift)
        self.pad_kwargs = **kwargs


        
    @static
    def create_pad_width(pad_side, frame_length, frame_shift):
        overlap = frame_length - frame_shift
        if pad_side=='symmetric':
            pad_width=(int(np.ceil(overlap/2)),
                      int(np.floor(overlap/2)))
        elif pad_side=='left':
            pad_width=(int(overlap), 0)
        elif pad_side=='right':
            pad_width=(0, int(overlap))
        else:
            raise Exception('Unknown pad_side=%s' % pad_side)
        

        
    def create_frames(self, x):
        
        if self.pad_mode is not None:
            x=self.apply_padding(x)

        if x.ndim==1:
            num_samples=x.shape[0]
            in_dim=1
        else:
            num_samples=x.shape[0]
            in_dim=x.shape[1]

        num_out_frames=int(np.floor((num_samples-frame_length)/frame_shift+1))
        
        vec_x=x.ravel()
        out_dim=frame_length*in_dim
        X=np.zeros((num_out_frames, out_dim), dtype=float_cpu())
        
        start=0
        stop=out_dim
        shift=in_dim*frame_shift
        for i in xrange(num_out_frames):
            X[i,:]=vec_x[start:stop]
            start+=shift
            stop+=shift
            
        return X


    
    def apply_padding(self, x):
        pad_width = self.pad_width
        if x.ndim==2:
            pad_width=(pad_width, (0,0))
        return np.pad(x, pad_width, mode=self.pad_mode, **self.pad_kwargs)
