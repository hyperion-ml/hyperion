"""
Classes to read data from kaldi ark files.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange
from six import string_types

import sys
import numpy as np
import h5py

from ..hyp_defs import float_cpu
from ..utils.scp_list import SCPList
from ..utils.kaldi_matrix import KaldiMatrix, KaldiCompressedMatrix
from ..utils.kaldi_io_funcs import is_token, read_token, peek,
from .data_reader import SequentialDataReader, RandomAccessDataReader


class SequentialArkDataReader(SequentialDataReader):

    def __init__(self, file_path, permissive = False):
        super(SequentialArkDataReader, self).__init__(file_path, permissive)
        self.f = None
        self.cur_file = None

    
    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None



    def _seek(self, offset):
        cur_pos = self.f.tell()
        delta = offset - cur_pos
        self.f.seek(delta, 1)
        

        
    def _open_archive(self, file_path, offset=0):
        if self.f is None or file_path != self.cur_file:
            self.close()
            self.cur_file = file_path
            self.f = open(file_path, 'rb')
            # read_token(self.f, True)
            # binary = init_kaldi_input_stream(self.f)
            # if binary:
            #     self.f.seek(0,0)
            # else:
            #     self.f.close()
            #     self.f = open(file_path, 'r')

        if offset > 0:
            self._seek(offset)
        

            
    def read_num_rows(self, num_records=0, assert_same_dim=True):
        shapes = self.read_shapes(num_records, assert_same_dim)
        num_rows = np.array([s[0] if len(s)==2 else 1 for s in shapes], dtype=np.int32)
        return num_rows


    
    def read_dims(self, num_records=0, assert_same_dim=True):
        shapes = self.read_shapes(num_records, False)
        dims = np.array([s[-1] for s in shapes], dtype=np.int32)
        if assert_same_dim:
            assert np.all(dims==dims[0])
        return dims




    
class SequentialArkFileDataReader(SequentialArkDataReader):

    def __init__(self, file_path):
        super(SequentialArkFileDataReader, self).__init__(file_path, False)
        self._open_archive(self.file_path)
        self._eof = False

        
    def reset(self):
        if self.f is not None:
            self.f.seek(0, 0)
            self._eof = False
            
        
    def eof(self):
        return self._eof or self.f is None

    

    def read_shapes(self, num_records=0, assert_same_dim=True):
        keys = []
        shapes = []
        count = 0
        binary = False
        while num_records==0 or count < num_records:

            key_i = read_token(self.f, binary)
            if key_i == '':
                self._eof = True
                break

            binary = init_kaldi_input_stream(self.f)
            shape_i = KaldiMatrix.read_shape(
                self.f, binary, sequential_mode=True)

            assert num_rows_i == 0 or data_i.shape[0] == num_rows_i
            
            key = keys.append(key)
            shapes.append(shapes_i)
            self.cur_item += 1
            count += 1

        if assert_same_dim:
            dims = np.array([s[-1] for s in shapes], dtype=np.int32)
            assert np.all(dims == dims[0])
            
        return keys, shapes



    
    def read(self, num_records=0, squeeze=False, row_offset=0, num_rows=0):
        
        keys = []
        data = []
        count = 0
        binary = False
        while num_records==0 or count < num_records:

            key_i = read_token(self.f, binary)
            if key_i == '':
                self._eof = True
                break

            row_offset_i = 0 if row_offset == 0 else row_offset[i]
            num_rows_i = 0 if num_rows == 0 else num_rows[i]
            
            binary = init_kaldi_input_stream(self.f)
            data_i = KaldiMatrix.read(
                self.f, binary, row_offset_i, num_rows_i,
                sequential_mode=True).to_ndarray()

            assert num_rows_i == 0 or data_i.shape[0] == num_rows_i
            
            key = keys.append(key)
            data.append(data_i)
            self.cur_item += 1
            count += 1

        if squeeze:
            data = self._squeeze(data)
            
        return keys, data
        
        



class SequentialArkScriptDataReader(SequentialArkDataReader):

    def __init__(self, file_path, path_prefix=None):
        super(SequentialArkScriptDataReader, self).__init__(file_path, False)
        self.scp = SCPList.load(self.file_path)
        if path_prefix is not None:
            self.scp.add_prefix_to_filepath(path_prefix)
            
        self.cur_item = 0
        

        
    def reset(self):
        self.close()
        self.cur_item = 0


        
    def eof(self):
        return self.cur_item == len(scp)


    
    def read_shapes(self, num_records=0, assert_same_dim=True):
        if num_records == 0:
            num_records = len(self.scp) - self.cur_item
        
        keys = []
        shapes = []
        for i in xrange(num_records):
            key, file_path, offset, range_spec = self.scp[self.cur_item]

            row_offset_i, num_rows_i = self._combine_ranges(range_spec, 0, 0)

            self._open_archive(file_path, offset)
            binary = init_kaldi_input_stream(self.f)
            shape_i = KaldiMatrix.read_shape(
                self.f, binary, sequential_mode=True)

            shape_i = self._apply_range_to_shape(shape_i)
            
            key = keys.append(key)
            shapes.append(shape_i)
            self.cur_item += 1

        if assert_same_dim:
            dims = np.array([s[-1] for s in shapes], dtype=np.int32)
            assert np.all(dims == dims[0])
            
        return keys, shapes

    

    
    def read(self, num_records=0, squeeze=False, row_offset=0, num_rows=0):

        if num_records == 0:
            num_records = len(self.scp) - self.cur_item
        
        keys = []
        data = []
        for i in xrange(num_records):
            key, file_path, offset, range_spec = self.scp[self.cur_item]

            row_offset_i = 0 if row_offset == 0 else row_offset[i]
            num_rows_i = 0 if num_rows == 0 else num_rows[i]

            row_offset_i, num_rows_i = self._combine_ranges(
                range_spec, offset_i, num_rows_i)
            
            self._open_archive(file_path, offset)
            binary = init_kaldi_input_stream(self.f)
            data_i = KaldiMatrix.read(
                self.f, binary, row_offset_i, num_rows_i,
                sequential_mode=True).to_ndarray()

            assert num_rows_i == 0 or data_i.shape[0] == num_rows_i
            
            key = keys.append(key)
            data.append(data_i)
            self.cur_item += 1

        if squeeze:
            data = self._squeeze(data)
            
        return keys, data




class RandomAccessArkDataReader(RandomAccessDataReader):

    def __init__(self, file_path, path_prefix=None, permissive=False):
        super(RandomAccessArkDataReader, self).__init__(file_path, permissive)
        
        self.scp = SCPList.load(script)
        if path_prefix is not None:
            self.scp.add_prefix_to_filepath(path_prefix)

        archives, archive_idx = np.unique(
            self.scp.file_path, return_inverse=True)
        self.archives = archives
        self.archive_idx = archive_idx
        self.f = [None] * len(self.archives)
        

        
    def close(self):
        for f in self.f:
            if f is not None:
                f.close()
        self.f = [None] * len(self.f)



    def _open_archive(self, key_idx, offset=0):
        archive_idx = self.archive_idx[key_idx]
        if self.f[archive_idx] is None:
            self.f[archive_idx] = open(self.archive[archive_idx], 'rb')

        f = self.f[archive_idx]
        f.seek(offset, 0)
        return f


    
    def read_num_rows(self, keys, assert_same_dim=True):
        shapes = self.read_shapes(keys, assert_same_dim)
        num_rows = np.array([s[0] if len(s)==2 else 1 for s in shapes],
                            dtype=np.int32)
        return num_rows


    
    def read_dims(self, keys, assert_same_dim=True):
        shapes = self.read_shapes(keys, False)
        dims = np.array([s[-1] for s in shapes], dtype=np.int32)
        if assert_same_dim:
            assert np.all(dims==dims[0])
        return dims


    
    def read_shapes(self, keys, assert_same_dim=True):
        
        if isinstance(keys, string_types):
            keys = [keys]

        shapes = []
        for key in keys:
            
            if not (key in self.scp):
                if self.permissive:
                    shapes.append((0,))
                    continue
                else:
                    raise Exception('Key %s not found' % key)

            index = self.scp.get_index(key)
            _, file_path, offset, range_spec = self.scp[index]

            row_offset_i, num_rows_i = self._combine_ranges(
                range_spec, 0, 0)
            
            f = self._open_archive(index, offset)
            binary = init_kaldi_input_stream(f)
            shape_i = KaldiMatrix.read_shape(
                f, binary, sequential_mode=False)

            shape_i = self._apply_range_to_shape(shape_i)
            
            shapes.append(shape_i)

        if assert_same_dim:
            dims = np.array([s[-1] for s in shapes], dtype=np.int32)
            assert np.all(dims == dims[0])
            
        return shapes

    

    def read(self, keys, squeeze=False, row_offset=0, num_cols=0):
        if isinstance(keys, string_types):
            keys = [keys]

        for i,key in enumerate(keys):
            
            if not (key in self.scp):
                if self.permissive:
                    data.append(np.array([], dtype=float_cpu()))
                    continue
                else:
                    raise Exception('Key %s not found' % key)

            index = self.scp.get_index(key)
            _, file_path, offset, range_spec = self.scp[index]

            row_offset_i = 0 if row_offset == 0 else row_offset[i]
            num_rows_i = 0 if num_rows == 0 else num_rows[i]

            row_offset_i, num_rows_i = self._combine_ranges(
                range_spec, offset_i, num_rows_i)
            
            f = self._open_archive(index, offset)
            binary = init_kaldi_input_stream(f)
            data_i = KaldiMatrix.read(
                f, binary, row_offset_i, num_rows_i,
                sequential_mode=False).to_ndarray()

            assert num_rows_i == 0 or data_i.shape[0] == num_rows_i
            
            data.append(data_i)

        if squeeze:
            data = self._squeeze(data)
            
        return data
            
