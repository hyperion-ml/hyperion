"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

 Classes to read data from hdf5 files.
"""

import sys
import time
import numpy as np
import h5py
import multiprocessing

from ..hyp_defs import float_cpu
from ..utils.list_utils import split_list, split_list_group_by_key
from ..utils.scp_list import SCPList
from ..utils.kaldi_matrix import KaldiMatrix, KaldiCompressedMatrix
from ..utils.kaldi_io_funcs import is_token
from .data_reader import SequentialDataReader, RandomAccessDataReader


def _read_h5_data(dset, row_offset=0, num_rows=0, transform=None):
    """Auxiliary function to read the feature matrix from hdf5 dataset.
       It decompresses the data if it was compressed.

    Args:
      dset: hdf5 dataset correspoding to a feature matrix/vector.
      row_offset: First row to read from each feature matrix.
      num_rows: Number of rows to read from the feature matrix.
                If 0 it reads all the rows.
      transform: TransformList object, applies a transformation to the
                 features after reading them from disk.

    Returns:
      Numpy array with feature matrix/vector.
    """
    if row_offset > 0:
        if num_rows == 0:
            data = dset[row_offset:]
        else:
            data = dset[row_offset : row_offset + num_rows]
    elif num_rows > 0:
        data = dset[:num_rows]
    else:
        data = dset

    if "data_format" in dset.attrs:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        data = KaldiCompressedMatrix.build_from_data_attrs(
            data, dset.attrs
        ).to_ndarray()

    assert num_rows == 0 or data.shape[0] == num_rows

    data = np.asarray(data, dtype=float_cpu())
    if transform is not None:
        data = transform.predict(data)
    return data


class SequentialH5DataReader(SequentialDataReader):
    """Abstract base class to read hdf5 feature files in
    sequential order.

     Attributes:
        file_path: ark or scp file to read.
        transform: TransformList object, applies a transformation to the
                   features after reading them from disk.
        part_idx: It splits the input into num_parts and writes only
                  part part_idx, where part_idx=1,...,num_parts.
        num_parts: Number of parts to split the input data.
        split_by_key: If True, all the elements with the same key go to the same part.
    """

    def __init__(self, file_path, **kwargs):
        super().__init__(file_path, **kwargs)
        self.f = None
        self.cur_file = None
        self.cur_item = 0

    def close(self):
        """Closes current hdf5 file."""
        if self.f is not None:
            self.f.close()
            self.f = None

    def _open_archive(self, file_path):
        """Opens the hdf5 file where the next matrix/vector is
        if it is not open.
        If there was another hdf5 file open, it closes it.
        """
        if self.f is None or file_path != self.cur_file:
            self.close()
            self.cur_file = file_path
            self.f = h5py.File(file_path, "r")

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
        keys, shapes = self.read_shapes(num_records, assert_same_dim)
        num_rows = np.array([s[0] if len(s) == 2 else 1 for s in shapes], dtype=int)
        return keys, num_rows

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
        keys, shapes = self.read_shapes(num_records, False)
        dims = np.array([s[-1] for s in shapes], dtype=np.int32)
        if assert_same_dim and len(dims) > 0:
            assert np.all(dims == dims[0])
        return keys, dims


class SequentialH5FileDataReader(SequentialH5DataReader):
    """Class to read feature matrices/vectors in
    sequential order from a single hdf5 file.

     Attributes:
        file_path: Ark file to read.
        transform: TransformList object, applies a transformation to the
                   features after reading them from disk.
        part_idx: It splits the input into num_parts and writes only
                  part part_idx, where part_idx=1,...,num_parts.
        num_parts: Number of parts to split the input data.
        split_by_key: If True, all the elements with the same key go to the same part.
    """

    def __init__(self, file_path, **kwargs):
        super().__init__(file_path, permissive=False, **kwargs)
        self._open_archive(self.file_path)
        self._keys = list(self.f.keys())
        if self.num_parts > 1:
            if self.split_by_key:
                self._keys, _ = split_list_group_by_key(
                    self._keys, self.part_idx, self.num_parts
                )
            else:
                self._keys, _ = split_list(self._keys, self.part_idx, self.num_parts)

    @property
    def keys(self):
        return self._keys

    def reset(self):
        """Puts the file pointer back to the begining of the file"""
        if self.f is not None:
            self.cur_item = 0

    def eof(self):
        """Returns True when it reaches the end of the ark file."""
        return self.cur_item == len(self._keys)

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
        if num_records == 0:
            num_records = len(self._keys) - self.cur_item

        keys = []
        shapes = []
        for i in range(num_records):
            if self.eof():
                break
            key = self._keys[self.cur_item]
            keys.append(key)
            shapes.append(self.f[key].shape)
            self.cur_item += 1

        if assert_same_dim and len(shapes) > 0:
            dims = np.array([s[-1] for s in shapes], dtype=np.int32)
            assert np.all(dims == dims[0])

        return keys, shapes

    def read(self, num_records=0, squeeze=False, row_offset=0, num_rows=0):
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
        if num_records == 0:
            num_records = len(self._keys) - self.cur_item

        row_offset_is_list = isinstance(row_offset, list) or isinstance(
            row_offset, np.ndarray
        )
        num_rows_is_list = isinstance(num_rows, list) or isinstance(
            num_rows, np.ndarray
        )
        keys = []
        data = []
        with self.lock:
            for i in range(num_records):
                if self.eof():
                    break

                key_i = self._keys[self.cur_item]

                row_offset_i = row_offset[i] if row_offset_is_list else row_offset
                num_rows_i = num_rows[i] if num_rows_is_list else num_rows

                dset_i = self.f[key_i]
                data_i = _read_h5_data(dset_i, row_offset_i, num_rows_i, self.transform)

                self.cur_item += 1

                keys.append(key_i)
                data.append(data_i)

        if squeeze:
            data = self._squeeze(data)

        return keys, data


class SequentialH5ScriptDataReader(SequentialH5DataReader):
    """Class to read features from multiple hdf5 files where a scp file
    indicates which hdf5 file contains each feature matrix.

     Attributes:
        file_path: scp file to read.
        path_prefix: If input_spec is a scp file, it pre-appends
                     path_prefix string to the second column of
                     the scp file. This is useful when data
                     is read from a different directory of that
                     it was created.
        scp_sep: Separator for scp files (default ' ').
        transform: TransformList object, applies a transformation to the
                   features after reading them from disk.
        part_idx: It splits the input into num_parts and writes only
                  part part_idx, where part_idx=1,...,num_parts.
        num_parts: Number of parts to split the input data.
        split_by_key: If True, all the elements with the same key go to the same part.
    """

    def __init__(self, file_path, path_prefix=None, scp_sep=" ", **kwargs):
        super().__init__(file_path, permissive=False, **kwargs)

        self.scp = SCPList.load(self.file_path, sep=scp_sep)
        if self.num_parts > 1:
            self.scp = self.scp.split(
                self.part_idx, self.num_parts, group_by_key=self.split_by_key
            )
        if path_prefix is not None:
            self.scp.add_prefix_to_filepath(path_prefix)

    @property
    def keys(self):
        return self.scp.key

    def reset(self):
        """Closes all the open hdf5 files and puts the read pointer pointing
        to the first element in the scp file."""
        self.close()
        self.cur_item = 0

    def eof(self):
        """Returns True when all the elements in the scp have been read."""
        return self.cur_item == len(self.scp)

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
        if num_records == 0:
            num_records = len(self.scp) - self.cur_item

        keys = []
        shapes = []
        for i in range(num_records):
            if self.eof():
                break

            key, file_path, offset, range_spec = self.scp[self.cur_item]

            row_offset_i, num_rows_i = self._combine_ranges(range_spec, 0, 0)

            self._open_archive(file_path)

            shape_i = self.f[key].shape
            shape_i = self._apply_range_to_shape(shape_i, row_offset_i, num_rows_i)

            keys.append(key)
            shapes.append(shape_i)
            self.cur_item += 1

        if assert_same_dim:
            dims = np.array([s[-1] for s in shapes], dtype=np.int32)
            assert np.all(dims == dims[0])

        return keys, shapes

    def read(self, num_records=0, squeeze=False, row_offset=0, num_rows=0):
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
        if num_records == 0:
            num_records = len(self.scp) - self.cur_item

        row_offset_is_list = isinstance(row_offset, list) or isinstance(
            row_offset, np.ndarray
        )
        num_rows_is_list = isinstance(num_rows, list) or isinstance(
            num_rows, np.ndarray
        )

        keys = []
        data = []
        with self.lock:
            for i in range(num_records):
                if self.eof():
                    break

                key, file_path, offset, range_spec = self.scp[self.cur_item]

                row_offset_i = row_offset[i] if row_offset_is_list else row_offset
                num_rows_i = num_rows[i] if num_rows_is_list else num_rows
                row_offset_i, num_rows_i = self._combine_ranges(
                    range_spec, row_offset_i, num_rows_i
                )

                self._open_archive(file_path)

                dset_i = self.f[key]
                data_i = _read_h5_data(dset_i, row_offset_i, num_rows_i, self.transform)
                self.cur_item += 1

                key = keys.append(key)
                data.append(data_i)

        if squeeze:
            data = self._squeeze(data)

        return keys, data


class RandomAccessH5DataReader(RandomAccessDataReader):
    """Abstract base class to read hdf5 feature files in
    random order.

     Attributes:
        file_path: hdf5 or scp file to read.
        transform: TransformList object, applies a transformation to the
                   features after reading them from disk.
        permissive: If True, if the data that we want to read is not in the file
                    it returns an empty matrix, if False it raises an exception.
    """

    def __init__(self, file_path, transform=None, permissive=False):
        super().__init__(file_path, transform, permissive)
        self.f = None

    def read_num_rows(self, keys, assert_same_dim=True):
        """Reads the number of rows in the feature matrices of the dataset.

        Args:
          keys: List of recording names from which we want to retrieve the
                number of rows.
          assert_same_dim: If True, it raise exception in not all the matrices have
                           the same number of columns.

        Returns:
          Integer numpy array with the number of rows for the recordings in keys.
        """
        shapes = self.read_shapes(keys, assert_same_dim)
        num_rows = np.array([s[0] if len(s) == 2 else 1 for s in shapes], dtype=int)
        return num_rows

    def read_dims(self, keys, assert_same_dim=True):
        """Reads the number of columns in the feature matrices of the dataset.

        Args:
          keys: List of recording names from which we want to retrieve the
                number of columns.
          assert_same_dim: If True, it raise exception in not all the matrices have
                           the same number of columns.

        Returns:
          Integer numpy array with the number of columns for the recordings in keys
        """
        shapes = self.read_shapes(keys, False)
        dims = np.array([s[-1] for s in shapes], dtype=np.int32)
        if assert_same_dim:
            assert np.all(dims == dims[0])
        return dims


class RandomAccessH5FileDataReader(RandomAccessH5DataReader):
    """Class to read from a single hdf5 file in random order

    Attributes:
       file_path: scp file to read.
       transform: TransformList object, applies a transformation to the
                  features after reading them from disk.
       permissive: If True, if the data that we want to read is not in the file
                   it returns an empty matrix, if False it raises an exception.
    """

    def __init__(self, file_path, **kwargs):
        super().__init__(file_path, **kwargs)
        self.lock = multiprocessing.Lock()
        self._open_archive(file_path)

    def close(self):
        """Closes the hdf5 files."""
        if self.f is not None:
            self.f.close()
            self.f = None

    def _open_archive(self, file_path):
        """Open the hdf5 file it it is not open."""
        if self.f is None:
            self.close()
            self.f = h5py.File(file_path, "r")

    @property
    def keys(self):
        return list(self.f.keys())

    def read_shapes(self, keys, assert_same_dim=True):
        """Reads the shapes in the feature matrices of the dataset.

        Args:
          keys: List of recording names from which we want to retrieve the
                shapes.
          assert_same_dim: If True, it raise exception in not all the matrices have
                           the same number of columns.

        Returns:
          List of tuples with the shapes for the recordings in keys.
        """
        if isinstance(keys, str):
            keys = [keys]

        shapes = []
        for key in keys:

            if not (key in self.f):
                if self.permissive:
                    shapes.append((0,))
                    continue
                else:
                    raise Exception("Key %s not found" % key)

            shape_i = self.f[key].shape
            shapes.append(shape_i)

        if assert_same_dim:
            dims = np.array([s[-1] for s in shapes], dtype=np.int32)
            assert np.all(dims == dims[0])

        return shapes

    def read(self, keys, squeeze=False, row_offset=0, num_rows=0):
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
        if isinstance(keys, str):
            keys = [keys]

        row_offset_is_list = isinstance(row_offset, list) or isinstance(
            row_offset, np.ndarray
        )
        num_rows_is_list = isinstance(num_rows, list) or isinstance(
            num_rows, np.ndarray
        )
        if row_offset_is_list:
            assert len(row_offset) == len(keys)
        if num_rows_is_list:
            assert len(num_rows) == len(keys)

        data = []
        for i, key in enumerate(keys):

            if not (key in self.f):
                if self.permissive:
                    data.append(np.array([], dtype=float_cpu()))
                    continue
                else:
                    raise Exception("Key %s not found" % key)

            row_offset_i = row_offset[i] if row_offset_is_list else row_offset
            num_rows_i = num_rows[i] if num_rows_is_list else num_rows

            with self.lock:
                dset_i = self.f[key]
                data_i = _read_h5_data(dset_i, row_offset_i, num_rows_i, self.transform)
            data.append(data_i)

        if squeeze:
            data = self._squeeze(data, self.permissive)

        return data


class RandomAccessH5ScriptDataReader(RandomAccessH5DataReader):
    """Class to read multiple hdf5 files in random order, where a scp file
    indicates which hdf5 file contains each feature matrix.

    Attributes:
        file_path: scp file to read.
        path_prefix: If input_spec is a scp file, it pre-appends
                     path_prefix string to the second column of
                     the scp file. This is useful when data
                     is read from a different directory of that
                     it was created.
        transform: TransformList object, applies a transformation to the
                   features after reading them from disk.
        permissive: If True, if the data that we want to read is not in the file
                    it returns an empty matrix, if False it raises an exception.
        scp_sep: Separator for scp files (default ' ').
    """

    def __init__(self, file_path, path_prefix=None, scp_sep=" ", **kwargs):
        super().__init__(file_path, **kwargs)

        self.scp = SCPList.load(self.file_path, sep=scp_sep)
        if path_prefix is not None:
            self.scp.add_prefix_to_filepath(path_prefix)

        archives, archive_idx = np.unique(self.scp.file_path, return_inverse=True)
        self.archives = archives
        self.archive_idx = archive_idx
        self.f = [None] * len(self.archives)
        self.locks = [multiprocessing.Lock() for i in range(len(self.archives))]

    def close(self):
        """Closes all the open hdf5 files."""
        for f in self.f:
            if f is not None:
                f.close()
        self.f = [None] * len(self.f)

    @property
    def keys(self):
        return self.scp.key

    def _open_archive(self, key_idx):
        """Opens the hdf5 file correspoding to a given feature/matrix
           if it is not already open.

        Args:
          key_idx: Integer position of the feature matrix in the scp file.

        Returns:
          Python file object.
        """
        archive_idx = self.archive_idx[key_idx]
        with self.locks[archive_idx]:
            if self.f[archive_idx] is None:
                self.f[archive_idx] = h5py.File(self.archives[archive_idx], "r")

        return self.f[archive_idx], self.locks[archive_idx]

    def read_shapes(self, keys, assert_same_dim=True):
        """Reads the shapes in the feature matrices of the dataset.

        Args:
          keys: List of recording names from which we want to retrieve the
                shapes.
          assert_same_dim: If True, it raise exception in not all the matrices have
                           the same number of columns.

        Returns:
          List of tuples with the shapes for the recordings in keys.
        """
        if isinstance(keys, str):
            keys = [keys]
        # t1 = time.time()
        shapes = []
        for key in keys:

            if not (key in self.scp):
                if self.permissive:
                    shapes.append((0,))
                    continue
                else:
                    raise Exception("Key %s not found" % key)

            index = self.scp.get_index(key)
            _, file_path, offset, range_spec = self.scp[index]

            row_offset_i, num_rows_i = self._combine_ranges(range_spec, 0, 0)

            f, lock = self._open_archive(index)
            if not (key in f):
                if self.permissive:
                    shapes.append((0,))
                    continue
                else:
                    raise Exception("Key %s not found" % key)

            with lock:
                shape_i = f[key].shape
            shape_i = self._apply_range_to_shape(shape_i, row_offset_i, num_rows_i)
            # print('%s %d %.2f' % (key,time.time()-t1, len(shapes)/len(keys)*100.))
            shapes.append(shape_i)

        if assert_same_dim:
            dims = np.array([s[-1] for s in shapes], dtype=np.int32)
            assert np.all(dims == dims[0])

        return shapes

    def read(self, keys, squeeze=False, row_offset=0, num_rows=0):
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
        if isinstance(keys, str):
            keys = [keys]

        row_offset_is_list = isinstance(row_offset, list) or isinstance(
            row_offset, np.ndarray
        )
        num_rows_is_list = isinstance(num_rows, list) or isinstance(
            num_rows, np.ndarray
        )
        if row_offset_is_list:
            assert len(row_offset) == len(keys)
        if num_rows_is_list:
            assert len(num_rows) == len(keys)

        data = []
        for i, key in enumerate(keys):

            if not (key in self.scp):
                if self.permissive:
                    data.append(np.array([], dtype=float_cpu()))
                    continue
                else:
                    raise Exception("Key %s not found" % key)

            index = self.scp.get_index(key)
            _, file_path, offset, range_spec = self.scp[index]

            row_offset_i = row_offset[i] if row_offset_is_list else row_offset
            num_rows_i = num_rows[i] if num_rows_is_list else num_rows
            row_offset_i, num_rows_i = self._combine_ranges(
                range_spec, row_offset_i, num_rows_i
            )

            f, lock = self._open_archive(index)
            with lock:
                if not (key in f):
                    if self.permissive:
                        data.append(np.array([], dtype=float_cpu()))
                        continue
                    else:
                        raise Exception("Key %s not found" % key)

                dset_i = f[key]
                data_i = _read_h5_data(dset_i, row_offset_i, num_rows_i, self.transform)

            data.append(data_i)

        if squeeze:
            data = self._squeeze(data, self.permissive)

        return data
