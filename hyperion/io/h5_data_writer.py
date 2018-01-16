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
from ..utils.scp_list import SCPList
from ..utils.kaldi_matrix import KaldiMatrix, KaldiCompressedMatrix
from ..utils.kaldi_io_funcs import is_token
from .data_writer import DataWriter


class H5DataWriter(DataWriter):

    def __init__(self, archive_path, script_path=None, **kwargs):
        
        super(H5DataWriter, self).__init__(
            archive_path, script_path, **kwargs)

        self.f = h5py.File(archive_path, 'w')
        if script_path is None:
            self.f_script = None
        else:
            self.f_script = open(script_path, 'w')


            
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


        
    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None
        if self.f_script is not None:
            self.f_script.close()


            
    def flush(self):
        self.f.flush()
        if self.f_script is not None:
            self.f_script.flush()
        


    def _convert_data(self, data):
        
        if isinstance(data, np.ndarray):
            if self.compress:
                mat = KaldiCompressedMatrix.compress(
                    data, self.compression_method)
                return mat.get_data_attrs()
            else:
                data = data.astype(float_save(), copy=False)
                return data, None
        else:
            raise ValueError('Data is not ndarray')


        
    def write(self, keys, data):
        
        if isinstance(keys, string_types):
            keys = [keys]
            data = [data]
            
        for i, key_i in enumerate(keys):
            assert is_token(key_i), 'Token %s not valid' % key_i
            data_i, attrs = self._convert_data(data[i])
            dset = self.f.create_dataset(key_i, data=data_i)
            if attrs is not None:
                for k, v in attrs.items():
                    dset.attrs[k] = v
                
            if self.f_script is not None:
                self.f_script.write('%s%s%s\n' % (
                    key_i, self.scp_sep, self.archive_path))

            if self._flush:
                self.flush()
