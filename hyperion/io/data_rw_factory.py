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

# from ..hyp_defs import float_save
# from ..utils.kaldi_io_funcs import is_token

from .rw_specifiers import ArchiveType, WSpecifier, RSpecifier, WSpecType, RSpecType
from .h5_data_writer import H5DataWriter as H5DW
from .ark_data_writer import ArkDataWriter as ADW
from .ark_data_reader import SequentialArkFileDataReader as SAFDR
from .ark_data_reader import SequentialArkScriptDataReader as SASDR
from .ark_data_reader import RandomAccessArkDataReader as RADR



class DataWriterFactory(object):

    @staticmethod
    def create(wspecifier, compress, compression_method):
        if isinstance(wspecifier, string_types):
            wspecifier = WSpecifier.create(wspecifier)

        if (wspecifier.spec_type ==  WSpecType.ARCHIVE or
            wspecifier.spec_type == WSpecType.BOTH):
            
            if wspecifier.archive_type == ArchiveType.H5:
                return H5DW(wspecifier.archive, wspecifier.script,
                            wspecifier.flush,
                            compress, compression_method)
            else:
                return ADW(wspecifier.archive, wspecifier.script,
                           wspecifier.binary, wspecifier.flush,
                           compress, compression_method)


            
class SequentialDataReaderFactory(object):

    @staticmethod
    def create(rspecifier, path_prefix=None):
        if isinstance(rspecifier, string_types):
            rspecifier = RSpecifier.create(rspecifier)
            
        if rspecifier.spec_type ==  WSpecType.ARCHIVE
            if rspecifier.archive_type == ArchiveType.H5:
                pass
                # return SequentialH5FileDataReader(rspecifier)
            else:
                return SAFDR(rspecifier.archive)
        else:
            if rspecifier.archive_type == ArchiveType.H5:
                pass
                # return SequentialH5ScriptDataReader(rspecifier)
            else:
                return SASDR(rspecifier.script, path_prefix)


            
class RandomAccessDataReaderFactory(object):

    @staticmethod
    def create(rspecifier, path_prefix=None):
        
        if isinstance(rspecifier, string_types):
            rspecifier = RSpecifier.create(rspecifier)
            
        if rspecifier.spec_type ==  WSpecType.ARCHIVE
            if rspecifier.archive_type == ArchiveType.H5:
                pass
                # return RH5FDR(rspecifier)
            else:
                raise ValueError(
                    'Random access to Ark file %s needs a script file' %
                    rspecifier.archive)
        else:
            if rspecifier.archive_type == ArchiveType.H5:
                pass
                # return RH5SDR(rspecifier)
            else:
                return RADR(rspecifier.script, path_prefix, rspecifier.permissive)



