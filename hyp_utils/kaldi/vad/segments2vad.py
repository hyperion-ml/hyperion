#!/usr/bin/env python
"""
 Copyright 2018   Johns Hopkins University (Jesus Villalba) 
 Apache 2.0
"""

from __future__ import absolute_import
from __future__ import print_function
from six.moves import xrange

import sys
import os
import argparse
import time
import re
import numpy as np
import pandas as pd

frame_shift=0.01

def write_vad(f, file_id, vad):
    f.write('%s [ ' % (file_id))
    for i in xrange(len(vad)):
        f.write('%d ' % vad[i])
    f.write(']\n')

    
def segments2vad_file(file_id, marks, num_frames, fvad):

    tbeg = np.round(np.array(marks.tbeg, ndmin=1)/frame_shift).astype('int')
    tend = np.round(np.array(marks.tend, ndmin=1)/frame_shift).astype('int')
    tend[-1] = min(tend[-1], num_frames)
    
    vad = np.zeros((num_frames,), dtype=int)
    for j in xrange(len(tbeg)):
        vad[tbeg[j]:tend[j]+1] = 1
    write_vad(fvad, file_id, vad)

    

def segments2vad(segments_file, num_frames_file, vad_file):

    df_segments = pd.read_csv(segments_file, sep='\s+', header=None,
                              names=['segments_id','file_id','tbeg','tend'])
    df_segments.index = df_segments.file_id
    
    df_num_frames = pd.read_csv(num_frames_file, sep='\s+', header=None,
                             names=['file_id','num_frames'])
    df_num_frames.index = df_num_frames.file_id

    with open(vad_file, 'w') as fvad:
        for file_id in df_num_frames.file_id:
            print(file_id)
            num_frames_i = int(df_num_frames.num_frames.loc[file_id])
            if file_id in df_segments.index:
                df_segments_i = df_segments.loc[file_id]
                #print(df_segments_i)
                segments2vad_file(file_id, df_segments_i, num_frames_i, fvad)
            else:
                print('Empty file %s' % file_id)
    

if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Converts Vimal VAD segments to kaldi VAD files')

    parser.add_argument('--segments',dest='segments_file', required=True)
    parser.add_argument('--num-frames', dest='num_frames_file', required=True)
    parser.add_argument('--vad-file', dest='vad_file', required=True)
    args=parser.parse_args()
    
    segments2vad(**vars(args))
                            
