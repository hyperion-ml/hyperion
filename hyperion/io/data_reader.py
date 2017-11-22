"""
Base class to write ark or h5 files
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

from abc import ABCMeta, abstractmethod

from ..utils.scp_list import SCPList


class DataReader(object):
    __metaclass__ = ABCMeta
    def __init__(self, file_path, permissive=False):
        self.file_path = file_path
        self.permissive = permissive

    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


    @abstractmethod
    def close(self):
        pass


    @staticmethod
    def _squeeze(data):

        ndim = data[0].ndim
        shape = data[0].shape
        for i, data_i in xrange(len(data)):
            if len(data_i) == 0:
                if self.permissive:
                    data[i] = np.zeros((1, shape[0]), dtype=float_cpu())
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
        if row_offset_i > 0 or num_rows_i > 0:
            shape = list(shape)
            shape[0] -= row_offset_i
            if num_rows_i > 0:
                assert shape[0] >= num_rows_i
                shape[0] = num_rows_i
            shape = tuple(shape)
        return shape



            
class SequentialDataReader(DataReader):
    __metaclass__ = ABCMeta

    def __init__(self, file_path, permissive=False):
        super(SequentialDataReader, self).__init__(file_path, permissive)
        

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

    def __init__(self, rspecifier):
        super(RandomAccessDataReader, self).__init__(rspecifier)

        
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
