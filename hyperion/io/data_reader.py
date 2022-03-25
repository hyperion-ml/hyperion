"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from abc import ABCMeta, abstractmethod
import numpy as np
import multiprocessing

from ..hyp_defs import float_cpu
from ..utils.scp_list import SCPList
from ..np.transforms import TransformList


class DataReader(object):
    __metaclass__ = ABCMeta

    def __init__(self, file_path, transform=None, permissive=False):
        """Abstract base class to read Ark or hdf5 feature files.

        Attributes:
           file_path: h5, ark or scp file to read.
           transform: TransformList object, applies a transformation to the
                      features after reading them from disk.
           permissive: If True, if the data that we want to read is not in the file
                       it returns an empty matrix, if False it raises an exception.

        """
        self.file_path = file_path
        self.permissive = permissive
        if isinstance(transform, str):
            self.transform = TransformList.load(transform)
        else:
            self.transform = transform

    def __enter__(self):
        """Function required when entering contructions of type

        with DataReader('file.h5') as f:
           keys, data = f.read()
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Function required when exiting from contructions of type

        with DataReader('file.h5') as f:
           keys, data = f.read()
        """
        self.close()

    @abstractmethod
    def close(self):
        """Closes input file."""
        pass

    @staticmethod
    def _squeeze(data, permissive=False):
        """Converts list of matrices to 3D numpy array or
           list of vectors to 2D numpy array.

        Args:
          data: List of matrices or vectors.
          permissive: If True, if one of the matrices/vectors in data is empty,
                      it substitutes it by matrix/vector with all zeros.
                      If false, it raises exception.

        Returns:
          2D or 3D numpy array.
        """
        ndim = data[0].ndim
        shape = data[0].shape
        for i in range(len(data)):
            if len(data[i]) == 0:
                if permissive:
                    data[i] = np.zeros((1,) + shape, dtype=float_cpu())
                continue
            assert ndim == data[i].ndim
            assert shape[-1] == data[i].shape[-1]
            data[i] = np.expand_dims(data[i], axis=0)

        return np.concatenate(tuple(data), axis=0)

    @staticmethod
    def _combine_ranges(read_range, row_offset, num_rows):
        """Combines two frame ranges.
           One is the range in the scp file, e.g, in the scp file
              recording1  file1.ark:34[3:40]
              recording2  file1.ark:100[5:20]

              [3:40] and [5:20] are frame ranges.

           The user can decide to just read a submatrix of that, e.g.,
           read 10 rows starting in row_offset 1.
           If we combine that with the range [3:40], the function returns.
           row_offset=4 (3+1) and num_rows=10.

        Args:
          read_range: Frame range from scp file. It is a tuple with the
             first row and number of rows to read.
          row_offset: User defined row_offset.
          num_rows: User defined number of rows to read, it it is 0, we read
                    all the rows defined in the scp read_range.

        Returns:
          Combined row_offset, first row of the recording to read.
          Combined number of rows (frames) to read.
        """
        if read_range is None:
            return row_offset, num_rows

        if num_rows == 0:
            num_rows = read_range[1]
        else:
            if read_range[1] > 0:
                assert read_range[1] - row_offset >= num_rows

        row_offset = row_offset + read_range[0]
        return row_offset, num_rows

    @staticmethod
    def _apply_range_to_shape(shape, row_offset, num_rows):
        """Modifies shape given the user defined row_offset and num_rows to read.
           If we are reading a matrix of shape (100,4) and row_offset=10, num_rows=20,
           it returns (20,4).
           If row_offset=20, num_rows=0, it returns (80,4).

        Args:
          shape: Original shape of the feature matrix.
          row_offset: User defined row_offset, first frame to read.
          num_rows: User defined num_rows, number of frames to read.

        Returns:
           2D tuple with modified shape.
        """
        if row_offset > 0 or num_rows > 0:
            shape = list(shape)
            shape[0] -= row_offset
            if num_rows > 0:
                assert shape[0] >= num_rows
                shape[0] = num_rows
            shape = tuple(shape)
        return shape


class SequentialDataReader(DataReader):
    """Abstract base class to read Ark or hdf5 feature files in
    sequential order.

     Attributes:
        file_path: h5, ark or scp file to read.
        transform: TransformList object, applies a transformation to the
                   features after reading them from disk.
        permissive: If True, if the data that we want to read is not in the file
                    it returns an empty matrix, if False it raises an exception.
        part_idx: It splits the input into num_parts and writes only
                  part part_idx, where part_idx=1,...,num_parts.
        num_parts: Number of parts to split the input data.
        split_by_key: If True, all the elements with the same key go to the same part.
    """

    __metaclass__ = ABCMeta

    def __init__(
        self,
        file_path,
        transform=None,
        permissive=False,
        part_idx=1,
        num_parts=1,
        split_by_key=False,
    ):
        super().__init__(file_path, transform, permissive)
        self.lock = multiprocessing.Lock()
        self.part_idx = part_idx
        self.num_parts = num_parts
        self.split_by_key = split_by_key

    def __iter__(self):
        """Needed to build an iterator, e.g.:
        r = SequentialDataReader(...)
        for key, data in r:
           print(key, data)
        """
        return self

    def __next__(self):
        """Needed to build an iterator, e.g.:
        r = SequentialDataReader(...)
        for key, data in r:
           print(key, data)
        """
        key, data = self.read(1)
        if len(key) == 0:
            raise StopIteration
        return key[0], data[0]

    def next(self):
        """__next__ for Python 2"""
        return self.__next__()

    @abstractmethod
    def reset(self):
        """Returns the file pointer to the begining of the dataset,
        then we can start reading the features again.
        """
        pass

    @abstractmethod
    def eof(self):
        """End of file.

        Returns:
          True, when we have read all the recordings in the dataset.
        """
        return False

    @abstractmethod
    def read_num_rows(self, num_records=0, assert_same_dim=True):
        """Reads the number of rows in the feature matrices of the dataset.

        Args:
          num_records: How many matrices shapes to read, if num_records=0 it
                       reads al the matrices in the dataset.
          assert_same_dim: If True, it raise exception in not all the matrices have
                           the same number of columns.

        Returns:
          List of num_records recording names.
          Integer numpy array with num_records number of rows.
        """
        pass

    @abstractmethod
    def read_dims(self, num_records=0, assert_same_dim=True):
        """Reads the number of columns in the feature matrices of the dataset.

        Args:
          num_records: How many matrices shapes to read, if num_records=0 it
                       reads al the matrices in the dataset.
          assert_same_dim: If True, it raise exception in not all the matrices have
                           the same number of columns.

        Returns:
          List of num_records recording names.
          Integer numpy array with num_records number of columns.
        """
        pass

    @abstractmethod
    def read_shapes(self, num_records=0, assert_same_dim=True):
        """Reads the shapes in the feature matrices of the dataset.

        Args:
          num_records: How many matrices shapes to read, if num_records=0 it
                       reads al the matrices in the dataset.
          assert_same_dim: If True, it raise exception in not all the matrices have
                           the same number of columns.

        Returns:
          List of num_records recording names.
          List of tuples with num_records shapes.
        """
        pass

    @abstractmethod
    def read(self, num_records=0, squeeze=False, offset=0, num_rows=0):
        """Reads next num_records feature matrices/vectors.

        Args:
          num_records: Number of feature matrices to read.
          squeeze: If True, it converts the list of
                   matrices/vectors to 3D/2D numpy array.
                   All matrices need to have same number of rows.
          offset: List of integers or numpy array of with the first row to
                  read from each feature matrix.
          num_rows: List of integers or numpy array of with the
                    number of rows to read from each feature matrix.
                    If 0 it reads all the rows.

        Returns:
          key: List of recording names.
          data: List of feature matrices/vectors or 3D/2D numpy array.
        """
        pass


class RandomAccessDataReader(DataReader):
    __metaclass__ = ABCMeta

    def __init__(self, file_path, transform=None, permissive=False):
        """Abstract base class to read Ark or hdf5 feature files in
           random order.

        Attributes:
           file_path: h5 or scp file to read.
           transform: TransformList object, applies a transformation to the
                      features after reading them from disk.
           permissive: If True, if the data that we want to read is not in the file
                       it returns an empty matrix, if False it raises an exception.
        """

        super().__init__(file_path, transform, permissive)

    @abstractmethod
    def read_num_rows(self, keys=None, assert_same_dim=True):
        """Reads the number of rows in the feature matrices of the dataset.

        Args:
          keys: List of recording names from which we want to retrieve the
                number of rows.
          assert_same_dim: If True, it raise exception in not all the matrices have
                           the same number of columns.

        Returns:
          Integer numpy array with the number of rows for the recordings in keys.
        """
        pass

    @abstractmethod
    def read_dims(self, keys=None, assert_same_dim=True):
        """Reads the number of columns in the feature matrices of the dataset.

        Args:
          keys: List of recording names from which we want to retrieve the
                number of columns.
          assert_same_dim: If True, it raise exception in not all the matrices have
                           the same number of columns.

        Returns:
          Integer numpy array with the number of columns for the recordings in keys
        """
        pass

    @abstractmethod
    def read_shapes(self, keys=None, assert_same_dim=True):
        """Reads the shapes in the feature matrices of the dataset.

        Args:
          keys: List of recording names from which we want to retrieve the
                shapes.
          assert_same_dim: If True, it raise exception in not all the matrices have
                           the same number of columns.

        Returns:
          List of tuples with the shapes for the recordings in keys.
        """
        pass

    @abstractmethod
    def read(self, keys, squeeze=False, offset=0, num_rows=0):
        """Reads the feature matrices/vectors for the recordings in keys.

        Args:
          keys: List of recording names from which we want to retrieve the
                feature matrices/vectors.
          squeeze: If True, it converts the list of
                   matrices/vectors to 3D/2D numpy array.
                   All matrices need to have same number of rows.
          offset: List of integers or numpy array of with the first row to
                  read from each feature matrix.
          num_rows: List of integers or numpy array of with the
                    number of rows to read from each feature matrix.
                    If 0 it reads all the rows.

        Returns:
          data: List of feature matrices/vectors or 3D/2D numpy array.
        """
        pass
