"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import numpy as np

from ..hyp_defs import float_cpu


class VADReader(object):
    """Abstract base class to read vad files.
    
        Attributes:
           file_path: h5, ark or scp file to read.
           permissive: If True, if the data that we want to read is not in the file 
                       it returns an empty matrix, if False it raises an exception.
      
    """
    def __init__(self, file_path, permissive=False):

        self.file_path = file_path
        self.permissive = permissive


    def __enter__(self):
        """Function required when entering contructions of type

           with VADReader('file.h5') as f:
              keys, data = f.read()
        """
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        """Function required when exiting from contructions of type

           with VADReader('file.h5') as f:
              keys, data = f.read()
        """
        self.close()

    def close(self):
        """Closes input file."""
        pass

    @staticmethod
    def _get_bin_vad_slice(vad, offset, num_frames):
        if offset > 0:
            vad = vad[offset:]

        if num_frames > 0:
            n = len(vad)
            if n > num_frames:
                vad = vad[:num_frames]
            elif n < num_frames:
                new_vad = np.zeros((num_frames,), dtype=np.bool)
                new_vad[:n] = vad
                vad = new_vad
        
        return vad
