"""
Base class to write ark or h5 files
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange
from six import string_types

from abc import ABCMeta, abstractmethod
import numpy as np

from ..hyp_defs import float_cpu
from ..utils.scp_list import SCPList
from ..transforms import TransformList


class DataReader(object):
    __metaclass__ = ABCMeta
    def __init__(self, file_path, transform=None, permissive=False):
        self.file_path = file_path
        self.permissive = permissive
        if isinstance(transform, string_types):
            self.transform = TransformList.load(transform)
        else:
            self.transform = transform
        

    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


    @abstractmethod
    def close(self):
        pass


    @staticmethod
    def _squeeze(data, permissive=False):

        ndim = data[0].ndim
        shape = data[0].shape
        for i in xrange(len(data)):
            if len(data[i]) == 0:
                if permissive:
                    data[i] = np.zeros((1,)+shape, dtype=float_cpu())
                continue
            assert ndim == data[i].ndim
            assert shape[-1] == data[i].shape[-1]
            data[i] = np.expand_dims(data[i], axis=0)
        
        return np.concatenate(tuple(data), axis=0)
            
            
        
            
    @staticmethod
    def _combine_ranges(read_range, row_offset, num_rows):
        if read_range is None:
            return row_offset, num_rows

        row_offset = row_offset + read_range[0]
            
        if num_rows == 0:
            num_rows = read_range[1]
        else:
            if read_range[1] > 0:
                assert read_range[1] >= num_rows

        return row_offset, num_rows

    
    @staticmethod
    def _apply_range_to_shape(shape, row_offset, num_rows):
        if row_offset > 0 or num_rows > 0:
            shape = list(shape)
            shape[0] -= row_offset
            if num_rows > 0:
                assert shape[0] >= num_rows
                shape[0] = num_rows
            shape = tuple(shape)
        return shape



            
class SequentialDataReader(DataReader):
    __metaclass__ = ABCMeta

    def __init__(self, file_path, transform=None, permissive=False,
                 part_idx=1, num_parts=1, split_by_key=False):
        super(SequentialDataReader, self).__init__(
            file_path, transform, permissive)
        self.part_idx = part_idx
        self.num_parts = num_parts
        self.split_by_key = split_by_key
        

        
    def __iter__(self):
        return self


    def __next__(self):
        key, data = self.read(1)
        if len(key)==0:
            raise StopIteration
        return key[0], data[0]
    

    def next(self):
        return self.__next__()

    
    @abstractmethod
    def reset(self):
        pass
    
    
    @abstractmethod
    def eof(self):
        return False
    

    @abstractmethod
    def read_num_rows(self, num_records=0, assert_same_dim=True):
        pass

    @abstractmethod
    def read_dims(self, num_records=0, assert_same_dim=True):
        pass

    @abstractmethod
    def read_shapes(self, num_records=0, assert_same_dim=True):
        pass


    @abstractmethod
    def read(self, num_records=0, squeeze=False, offset=0, num_cols=0):
        pass




class RandomAccessDataReader(DataReader):
    __metaclass__ = ABCMeta

    def __init__(self, file_path, transform=None, permissive=False):
        super(RandomAccessDataReader, self).__init__(
            file_path, transform, permissive)

        
    @abstractmethod
    def read_num_rows(self, keys=None, assert_same_dim=True):
        pass

    @abstractmethod
    def read_dims(self, keys=None, assert_same_dim=True):
        pass

    @abstractmethod
    def read_shapes(self, keys=None, assert_same_dim=True):
        pass

    @abstractmethod
    def read(self, keys, squeeze=False, offset=0, num_cols=0):
        pass
