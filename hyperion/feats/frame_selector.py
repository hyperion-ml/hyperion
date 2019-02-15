"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import logging

import numpy as np

class FrameSelector(object):
    """Class to select speech frames.

    Attributes:
       tol_num_frames: maximum tolerated error between number of feature frames and VAD frames.
    """
    
    def __init__(self, tol_num_frames=3):
        self.tol_num_frames = tol_num_frames

        
    def select(self, x, sel):
        """Select speech frames.
        
        Args:
          x: feature matrix.
          sel: binary selector vector.

        Returns:
          Feature matrix with selected frames.
        """
        num_frames = x.shape[0]
        num_frames_vad = sel.shape[0]
        if num_frames == num_frames_vad:
            return x[sel,:]
        elif num_frames > num_frames_vad:
            if num_frames - num_frames_vad <= self.tol_num_frames:
                return x[:num_frames_vad,:][sel,:]
            else:
                raise Exception('num_frames (%d) > num_frames_vad (%d) + tol (%d)'
                                % (num_frames, num_frames_vad, self.tol_num_frames))                
        else:
            if num_frames_vad - num_frames <= self.tol_num_frames:
                return x[sel[:num_frames],:]
            else:
                raise Exception('num_frames_vad (%d) > num_frames (%d) + tol (%d)'
                                % (num_frames_vad, num_frames, self.tol_num_frames))                
                            

            
    @staticmethod
    def filter_args(prefix=None, **kwargs):
        """Filters frame selector args from arguments dictionary.
           
           Args:
             prefix: Options prefix.
             kwargs: Arguments dictionary.
           
           Returns:
             Dictionary with frame-selector options.
        """
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('tol_num_frames')

        d = dict((k, kwargs[p+k])
                 for k in valid_args if p+k in kwargs)

        return d

    
        
    @staticmethod
    def add_argparse_args(parser, prefix=None):
        """Adds frame-selector options to parser.
           
           Args:
             parser: Arguments parser
             prefix: Options prefix.
        """

        if prefix is None:
            p1 = '--'
            p2 = ''
        else:
            p1 = '--' + prefix + '-'
            p2 = prefix + '_'

        parser.add_argument(p1+'tol-num-frames', dest=(p2+'tol_num_frames'), type=int,
                            default=3,
                            help='maximum tolerated error between number of feature frames and VAD frames.')

         
