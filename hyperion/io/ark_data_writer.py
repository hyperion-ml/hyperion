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

from ..hyp_defs import float_save
from ..utils.scp_list import SCPList
from ..utils.kaldi_io_funcs import is_token, write_token, init_kaldi_output_stream
from ..utils.kaldi_matrix import KaldiMatrix, KaldiCompressedMatrix
from .data_writer import DataWriter



class ArkDataWriter(DataWriter):

    def __init__(self, archive_path, script_path=None,
                 binary=True, **kwargs):
        super(ArkDataWriter, self).__init__(
            archive_path, script_path, **kwargs)
        self.binary = binary

        if binary:
            self.f = open(archive_path, 'wb')
        else:
            self.f = open(archive_path, 'w')

        if script_path is not None:
            self.f_script = open(script_path, 'w')
        else:
            self.f_script = None
            
            

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


            
    def _convert_data(self, data):
        if isinstance(data, np.ndarray):
            data = data.astype(float_save(), copy=False)
            if self.compress:
                return KaldiCompressedMatrix.compress(
                    data, self.compression_method)
            return KaldiMatrix(data)
        
        if isinstance(data, KaldiMatrix):
            if self.compress:
                return KaldiCompressedMatrix.compress(
                    data, self.compression_method)
            return data

        if isinstance(data, KaldiCompressedMatrix):
            if not self.compress:
                return data.to_matrix()
            return data

        raise ValueError('Data is not ndarray or KaldiMatrix')


    
    def write(self, keys, data):
        
        if isinstance(keys, string_types):
            keys = [keys]
            data = [data]
            
        for i, key_i in enumerate(keys):
            assert is_token(key_i), 'Token %s not valid' % key_i
            write_token(self.f, self.binary, key_i)

            pos = self.f.tell()
            data_i = self._convert_data(data[i])
        
            init_kaldi_output_stream(self.f, self.binary)
            data_i.write(self.f, self.binary)

            if self.f_script is not None:
                self.f_script.write('%s%s%s:%d\n' % (
                    key_i, self.scp_sep, self.archive_path, pos))
            
            if self._flush:
                self.flush()
