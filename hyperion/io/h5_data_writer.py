"""
Class to write data to hdf5 files.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange
from six import string_types

import sys
import numpy as np
import h5py

from ..hyp_defs import float_save
from ..utils.kaldi_io_funcs import is_token
from .data_writer import DataWriter


class H5DataWriter(DataWriter):

    def __init__(self, archive_path, script_path=None, flush=False,
                 compress=False, compression_method='auto'):
        
        super(H5DataWriter, self).__init__(
            archive_path, script_path, flush, compress, compression_method):

        self.f = h5py.File(archive_path, 'w')
        if script_path is None:
            self.f_script = None
        else:
            self.f_script = open(script_path, 'w')


    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

        
    def close(self):
        self.f.close()
        if self.f_script is not None:
            self.f_script.close()


    def flush(self):
        self.f.flush()
        if self.f_script is not None:
            self.f_script.flush()
        
        
    def write(self, keys, data):
        
        if isinstance(keys, string_types):
            keys = [keys]
            data = [data]
            
        for i, key_i in enumerate(keys):
            assert is_token(key_i), 'Token %s not valid' % key_i
            self.f.create_dataset(key_i, data=data[i].astype(float_save()),
                                  copy=False)
            if self.f_script is not None:
                self.f_script.write('%s %s\n' % (
                    key_i, self.wspecifier.archive))
                
