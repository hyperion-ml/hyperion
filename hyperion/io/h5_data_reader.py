"""
Classes to read data from hdf5 files.
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
from ..utils.list_utils import split_list, split_list_group_by_key
from ..utils.scp_list import SCPList
from ..utils.kaldi_matrix import KaldiMatrix, KaldiCompressedMatrix
from ..utils.kaldi_io_funcs import is_token
from .data_reader import SequentialDataReader, RandomAccessDataReader



# def _read_h5_shape(dset, row_offset=0, num_rows=0):
    
#     if row_offset == 0 and num_rows == 0:
#         shape = dset.shape
#     else:
#         shape = list(dset.shape)
#         if num_rows == 0:
#             shape[0] -= row_offset_i
#             assert shape[0] > 0
#         else:
#             assert num_rows <= shape[0]
#             shape[0] = num_rows
#         shape = tuple(shape)
        
#     return shape



def _read_h5_data(dset, row_offset=0, num_rows=0, transform=None):
    if row_offset > 0:
        if num_rows == 0:
            data = dset[row_offset:]
        else:
            data = dset[row_offset:row_offset+num_rows]
    elif num_rows > 0:
        data = dset[:num_rows]
    else:
        data = dset

    if 'data_format' in dset.attrs:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        data = KaldiCompressedMatrix.build_from_data_attrs(
            data, dset.attrs).to_ndarray()

    assert num_rows == 0 or data.shape[0] == num_rows

    data = np.asarray(data, dtype=float_cpu())
    if transform is not None:
        data = transform.predict(data)
    return data
    




class SequentialH5DataReader(SequentialDataReader):

    def __init__(self, file_path, **kwargs):
        super(SequentialH5DataReader, self).__init__(file_path, **kwargs)
        self.f = None
        self.cur_file = None
        self.cur_item = 0

        
    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None


        
    def _open_archive(self, file_path):
        if self.f is None or file_path != self.cur_file:
            self.close()
            self.cur_file = file_path
            self.f = h5py.File(file_path, 'r')
        

            
    def read_num_rows(self, num_records=0, assert_same_dim=True):
        keys, shapes = self.read_shapes(num_records, assert_same_dim)
        num_rows = np.array([s[0] if len(s)==2 else 1 for s in shapes], dtype=int)
        return keys, num_rows


    
    def read_dims(self, num_records=0, assert_same_dim=True):
        keys, shapes = self.read_shapes(num_records, False)
        dims = np.array([s[-1] for s in shapes], dtype=np.int32)
        if assert_same_dim and len(dims)>0:
            assert np.all(dims==dims[0])
        return keys, dims



    
class SequentialH5FileDataReader(SequentialH5DataReader):

    def __init__(self, file_path, **kwargs):
        super(SequentialH5FileDataReader, self).__init__(
            file_path, permissive=False, **kwargs)
        self._open_archive(self.file_path)
        self._keys = list(self.f.keys())
        if self.num_parts > 1:
            if self.split_by_key:
                self._keys, _ = split_list_group_by_key(
                    self._keys, self.part_idx, self.num_parts)
            else:
                self._keys, _ = split_list(self._keys, self.part_idx, self.num_parts)


        
    def reset(self):
        if self.f is not None:
            self.cur_item = 0
            

            
    def eof(self):
        return self.cur_item == len(self._keys)

    

    def read_shapes(self, num_records=0, assert_same_dim=True):
        if num_records == 0:
            num_records = len(self._keys) - self.cur_item

        keys = []
        shapes = []
        for i in xrange(num_records):
            if self.eof():
                break
            key = self._keys[self.cur_item]
            keys.append(key)
            shapes.append(self.f[key].shape)
            self.cur_item += 1
                          
        if assert_same_dim and len(shapes)>0:
            dims = np.array([s[-1] for s in shapes], dtype=np.int32)
            assert np.all(dims == dims[0])
            
        return keys, shapes



    
    def read(self, num_records=0, squeeze=False, row_offset=0, num_rows=0):

        if num_records == 0:
            num_records = len(self._keys) - self.cur_item

        row_offset_is_list = (isinstance(row_offset, list) or
                              isinstance(row_offset, np.ndarray))
        num_rows_is_list = (isinstance(num_rows, list) or
                            isinstance(num_rows, np.ndarray))
        keys = []
        data = []
        for i in xrange(num_records):
            if self.eof():
                break
            key_i = self._keys[self.cur_item]

            row_offset_i = row_offset[i] if row_offset_is_list else row_offset
            num_rows_i = num_rows[i] if num_rows_is_list else num_rows

            dset_i = self.f[key_i]
            data_i = _read_h5_data(dset_i, row_offset_i, num_rows_i, self.transform)
            
            keys.append(key_i)
            data.append(data_i)
            self.cur_item += 1
            
        if squeeze:
            data = self._squeeze(data)
            
        return keys, data




class SequentialH5ScriptDataReader(SequentialH5DataReader):

    def __init__(self, file_path, path_prefix=None, scp_sep=' ', **kwargs):
        super(SequentialH5ScriptDataReader, self).__init__(
            file_path, permissive=False,  **kwargs)
                      
        self.scp = SCPList.load(self.file_path, sep=scp_sep)
        if self.num_parts > 1:
            self.scp = self.scp.split(self.part_idx, self.num_parts,
                                      group_by_key=self.split_by_key)
        if path_prefix is not None:
            self.scp.add_prefix_to_filepath(path_prefix)
            

        
    def reset(self):
        self.close()
        self.cur_item = 0


        
    def eof(self):
        return self.cur_item == len(self.scp)


    
    def read_shapes(self, num_records=0, assert_same_dim=True):
        if num_records == 0:
            num_records = len(self.scp) - self.cur_item
        
        keys = []
        shapes = []
        for i in xrange(num_records):
            if self.eof():
                break

            key, file_path, offset, range_spec = self.scp[self.cur_item]

            row_offset_i, num_rows_i = self._combine_ranges(range_spec, 0, 0)

            self._open_archive(file_path)

            shape_i = self.f[key].shape
            shape_i = self._apply_range_to_shape(
                shape_i, row_offset_i, num_rows_i)
                    
            keys.append(key)
            shapes.append(shape_i)
            self.cur_item += 1
            
        if assert_same_dim:
            dims = np.array([s[-1] for s in shapes], dtype=np.int32)
            assert np.all(dims == dims[0])
            
        return keys, shapes

    

    
    def read(self, num_records=0, squeeze=False, row_offset=0, num_rows=0):

        if num_records == 0:
            num_records = len(self.scp) - self.cur_item

        row_offset_is_list = (isinstance(row_offset, list) or
                              isinstance(row_offset, np.ndarray))
        num_rows_is_list = (isinstance(num_rows, list) or
                            isinstance(num_rows, np.ndarray))

        keys = []
        data = []
        for i in xrange(num_records):
            if self.eof():
                break
            key, file_path, offset, range_spec = self.scp[self.cur_item]

            row_offset_i = row_offset[i] if row_offset_is_list else row_offset
            num_rows_i = num_rows[i] if num_rows_is_list else num_rows
            row_offset_i, num_rows_i = self._combine_ranges(
                range_spec, row_offset_i, num_rows_i)
            
            self._open_archive(file_path)
            
            dset_i = self.f[key]
            data_i = _read_h5_data(dset_i, row_offset_i, num_rows_i, self.transform)
            
            key = keys.append(key)
            data.append(data_i)
            self.cur_item += 1

        if squeeze:
            data = self._squeeze(data)
            
        return keys, data




class RandomAccessH5DataReader(RandomAccessDataReader):

    def __init__(self, file_path, transform=None, permissive = False):
        super(RandomAccessH5DataReader, self).__init__(file_path, transform, permissive)
        self.f = None


        
    def read_num_rows(self, keys, assert_same_dim=True):
        shapes = self.read_shapes(keys, assert_same_dim)
        num_rows = np.array([s[0] if len(s)==2 else 1 for s in shapes],
                            dtype=int)
        return num_rows


    
    def read_dims(self, keys, assert_same_dim=True):
        shapes = self.read_shapes(keys, False)
        dims = np.array([s[-1] for s in shapes], dtype=np.int32)
        if assert_same_dim:
            assert np.all(dims==dims[0])
        return dims



    
class RandomAccessH5FileDataReader(RandomAccessH5DataReader):

    def __init__(self, file_path, **kwargs):
        super(RandomAccessH5FileDataReader, self).__init__(file_path, **kwargs)
        self._open_archive(file_path)



    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None

            
        
    def _open_archive(self, file_path):
        if self.f is None:
            self.close()
            self.f = h5py.File(file_path, 'r')
        

            
    def read_shapes(self, keys, assert_same_dim=True):
        
        if isinstance(keys, string_types):
            keys = [keys]

        shapes = []
        for key in keys:
            
            if not (key in self.f):
                if self.permissive:
                    shapes.append((0,))
                    continue
                else:
                    raise Exception('Key %s not found' % key)

            shape_i = self.f[key].shape
            shapes.append(shape_i)

        if assert_same_dim:
            dims = np.array([s[-1] for s in shapes], dtype=np.int32)
            assert np.all(dims == dims[0])
            
        return shapes

    

    def read(self, keys, squeeze=False, row_offset=0, num_rows=0):
        if isinstance(keys, string_types):
            keys = [keys]


        row_offset_is_list = (isinstance(row_offset, list) or
                              isinstance(row_offset, np.ndarray))
        num_rows_is_list = (isinstance(num_rows, list) or
                            isinstance(num_rows, np.ndarray))
        if row_offset_is_list:
            assert len(row_offset) == len(keys)
        if num_rows_is_list:
            assert len(num_rows) == len(keys)

        data = []
        for i,key in enumerate(keys):
            
            if not (key in self.f):
                if self.permissive:
                    data.append(np.array([], dtype=float_cpu()))
                    continue
                else:
                    raise Exception('Key %s not found' % key)

            row_offset_i = row_offset[i] if row_offset_is_list else row_offset
            num_rows_i = num_rows[i] if num_rows_is_list else num_rows

            dset_i = self.f[key]
            data_i = _read_h5_data(dset_i, row_offset_i, num_rows_i, self.transform)
            data.append(data_i)

        if squeeze:
            data = self._squeeze(data, self.permissive)
            
        return data
            



class RandomAccessH5ScriptDataReader(RandomAccessH5DataReader):

    def __init__(self, file_path, path_prefix=None, scp_sep=' ', **kwargs):
        super(RandomAccessH5DataReader, self).__init__(
            file_path, **kwargs)
        
        self.scp = SCPList.load(self.file_path, sep=scp_sep)
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



    def _open_archive(self, key_idx):
        archive_idx = self.archive_idx[key_idx]
        if self.f[archive_idx] is None:
            self.f[archive_idx] = h5py.File(self.archives[archive_idx], 'r')

        return self.f[archive_idx]


    
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
            
            f = self._open_archive(index)
            if not (key in f):
                if self.permissive:
                    shapes.append((0,))
                    continue
                else:
                    raise Exception('Key %s not found' % key)

            shape_i = f[key].shape
            shape_i = self._apply_range_to_shape(
                shape_i, row_offset_i, num_rows_i)
            
            shapes.append(shape_i)

        if assert_same_dim:
            dims = np.array([s[-1] for s in shapes], dtype=np.int32)
            assert np.all(dims == dims[0])
            
        return shapes

    

    def read(self, keys, squeeze=False, row_offset=0, num_rows=0):
        if isinstance(keys, string_types):
            keys = [keys]


        row_offset_is_list = (isinstance(row_offset, list) or
                              isinstance(row_offset, np.ndarray))
        num_rows_is_list = (isinstance(num_rows, list) or
                            isinstance(num_rows, np.ndarray))
        if row_offset_is_list:
            assert len(row_offset) == len(keys)
        if num_rows_is_list:
            assert len(num_rows) == len(keys)

        data = []
        for i,key in enumerate(keys):
            
            if not (key in self.scp):
                if self.permissive:
                    data.append(np.array([], dtype=float_cpu()))
                    continue
                else:
                    raise Exception('Key %s not found' % key)

            index = self.scp.get_index(key)
            _, file_path, offset, range_spec = self.scp[index]

            row_offset_i = row_offset[i] if row_offset_is_list else row_offset
            num_rows_i = num_rows[i] if num_rows_is_list else num_rows
            row_offset_i, num_rows_i = self._combine_ranges(
                range_spec, row_offset_i, num_rows_i)
            
            f = self._open_archive(index)
            if not (key in f):
                if self.permissive:
                    data.append(np.array([], dtype=float_cpu()))
                    continue
                else:
                    raise Exception('Key %s not found' % key)

            dset_i = f[key]
            data_i = _read_h5_data(dset_i, row_offset_i, num_rows_i, self.transform)
            data.append(data_i)

        if squeeze:
            data = self._squeeze(data, self.permissive)
            
        return data
            
