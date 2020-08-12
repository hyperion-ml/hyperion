"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
# from __future__ import absolute_import

import logging
import numpy as np

from ..hyp_defs import float_cpu
from .vad_reader import VADReader
from .data_rw_factory import RandomAccessDataReaderFactory as DRF

class BinVADReader(VADReader):

    def __init__(self, rspecifier, path_prefix=None, scp_sep=' '):

        r = DRF.create(rspecifier, path_prefix, scp_sep=scp_sep)
        super(BinVADReader, self).__init__(r.file_path, r.permissive)
        self.r = r

    def read_num_frames(self, keys):
        return self.r.read_dims(keys, assert_same_dim=False)


    def read(self, keys, squeeze=False, offset=0, num_frames=0):
        if isinstance(keys, str):
            keys = [keys]

        vad = self.r.read(keys)

        offset_is_list = (isinstance(offset, list) or
                              isinstance(offset, np.ndarray))
        num_frames_is_list = (isinstance(num_frames, list) or
                            isinstance(num_frames, np.ndarray))
        if offset_is_list:
            assert len(offset) == len(keys)
        if num_frames_is_list:
            assert len(num_frames) == len(keys)

        output_vad = []
        for i in range(len(keys)):
            vad_i = vad[i].astype(np.bool, copy=False)
            offset_i = offset[i] if offset_is_list else offset
            num_frames_i = num_frames[i] if num_frames_is_list else num_frames
            if offset_i > 0:
                vad_i = vad_i[offset:]

            if num_frames_i > 0:
                n_i = len(vad_i)
                if n_i > num_frames_i:
                    vad_i = vad_i[:num_frames_i]
                elif n_i < num_frames_i:
                    new_vad_i = np.zeros((num_frames_i,), dtype=np.bool)
                    new_vad_i[:n_i] = vad_i
                    vad_i = new_vad_i
            
            output_vad.append(vad_i)
        
        if squeeze:
            output_vad = self.r._squeeeze(output_vad, self.permissive)

        return output_vad

