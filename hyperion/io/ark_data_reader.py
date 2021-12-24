"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import sys
import numpy as np
import multiprocessing as threading

from ..hyp_defs import float_cpu
from ..utils.scp_list import SCPList
from ..utils.kaldi_matrix import KaldiMatrix, KaldiCompressedMatrix
from ..utils.kaldi_io_funcs import is_token, read_token, peek, init_kaldi_input_stream
from .data_reader import SequentialDataReader, RandomAccessDataReader


class SequentialArkDataReader(SequentialDataReader):
    """Abstract base class to read Ark feature files in
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
        self.lock = threading.Lock()
        self.cur_file = None

    def close(self):
        """Closes input file."""
        if self.f is not None:
            self.f.close()
            self.f = None

    def _seek(self, offset):
        """Moves the pointer of the input file.

        Args:
          offset: Byte where we want to put the pointer.
        """
        cur_pos = self.f.tell()
        delta = offset - cur_pos
        self.f.seek(delta, 1)

    def _open_archive(self, file_path, offset=0):
        """Opens the current file if it is not open and moves the
           file pointer to a given position.
           Closes previous open Ark files.

        Args:
          file_path: File from which we want to read the next feature matrix.
          offset: Byte position where feature matrix is in the file.
        """
        if self.f is None or file_path != self.cur_file:
            self.close()
            self.cur_file = file_path
            self.f = open(file_path, "rb")

        if offset > 0:
            self._seek(offset)

    def read_num_rows(self, num_records=0, assert_same_dim=True):
        """Reads the number of rows in the feature matrices of the dataset.

        Args:
          num_records: How many matrices shapes to read, if num_records=0 it
                       reads all the matrices in the dataset.
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
        dims = np.array([s[-1] for s in shapes], dtype=int)
        if assert_same_dim and len(dims) > 0:
            assert np.all(dims == dims[0])
        return keys, dims


class SequentialArkFileDataReader(SequentialArkDataReader):
    """Class to read feature matrices/vectors in
    sequential order from a single Ark file.

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
        super(SequentialArkFileDataReader, self).__init__(
            file_path, permissive=False, **kwargs
        )
        self._open_archive(self.file_path)
        self._eof = False
        self._keys = None
        if self.num_parts > 1:
            raise NotImplementedError(
                "Dataset splitting not available for %s" % self.__class__.__name__
            )

    def reset(self):
        """Puts the file pointer back to the begining of the file"""
        if self.f is not None:
            self.f.seek(0, 0)
            self._eof = False

    def eof(self):
        """Returns True when it reaches the end of the ark file."""
        return self._eof or self.f is None

    @property
    def keys(self):
        if self._keys is None:
            self.reset()
            self._keys, _ = self.read_shapes()
            self.reset()

        return self._keys

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
        keys = []
        shapes = []
        count = 0
        binary = False
        while num_records == 0 or count < num_records:

            key_i = read_token(self.f, binary)
            if key_i == "":
                self._eof = True
                break

            binary = init_kaldi_input_stream(self.f)
            shape_i = KaldiMatrix.read_shape(self.f, binary, sequential_mode=True)

            keys.append(key_i)
            shapes.append(shape_i)
            count += 1

        if assert_same_dim and len(shapes) > 0:
            dims = np.array([s[-1] for s in shapes], dtype=int)
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
        row_offset_is_list = isinstance(row_offset, list) or isinstance(
            row_offset, np.ndarray
        )
        num_rows_is_list = isinstance(num_rows, list) or isinstance(
            num_rows, np.ndarray
        )
        keys = []
        data = []
        count = 0
        binary = False
        with self.lock:
            while num_records == 0 or count < num_records:

                key_i = read_token(self.f, binary)
                if key_i == "":
                    self._eof = True
                    break

                row_offset_i = row_offset[i] if row_offset_is_list else row_offset
                num_rows_i = num_rows[i] if num_rows_is_list else num_rows

                binary = init_kaldi_input_stream(self.f)
                data_i = KaldiMatrix.read(
                    self.f, binary, row_offset_i, num_rows_i, sequential_mode=True
                ).to_ndarray()

                assert num_rows_i == 0 or data_i.shape[0] == num_rows_i

                if self.transform is not None:
                    data_i = self.transform.predict(data_i)

                keys.append(key_i)
                data.append(data_i)
                count += 1

        if squeeze:
            data = self._squeeze(data)

        return keys, data


class SequentialArkScriptDataReader(SequentialArkDataReader):
    """Class to read Ark feature files indexed by a scp file in
    sequential order.

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
        super(SequentialArkScriptDataReader, self).__init__(
            file_path, permissive=False, **kwargs
        )
        self.scp = SCPList.load(self.file_path, sep=scp_sep)

        if self.num_parts > 1:
            self.scp = self.scp.split(
                self.part_idx, self.num_parts, group_by_key=self.split_by_key
            )

        if path_prefix is not None:
            self.scp.add_prefix_to_filepath(path_prefix)

        self.cur_item = 0

    @property
    def keys(self):
        return self.scp.key

    def reset(self):
        """Closes all the open Ark files and puts the read pointer pointing
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

            self._open_archive(file_path, offset)
            binary = init_kaldi_input_stream(self.f)
            shape_i = KaldiMatrix.read_shape(self.f, binary, sequential_mode=True)

            shape_i = self._apply_range_to_shape(shape_i, row_offset_i, num_rows_i)

            keys.append(key)
            shapes.append(shape_i)
            self.cur_item += 1

        if assert_same_dim:
            dims = np.array([s[-1] for s in shapes], dtype=int)
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

                self._open_archive(file_path, offset)
                binary = init_kaldi_input_stream(self.f)
                data_i = KaldiMatrix.read(
                    self.f, binary, row_offset_i, num_rows_i, sequential_mode=True
                ).to_ndarray()

                assert num_rows_i == 0 or data_i.shape[0] == num_rows_i

                if self.transform is not None:
                    data_i = self.transform.predict(data_i)

                keys.append(key)
                data.append(data_i)
                self.cur_item += 1

        if squeeze:
            data = self._squeeze(data)

        return keys, data


class RandomAccessArkDataReader(RandomAccessDataReader):
    """Class to read Ark files in random order, using scp file to
    index the Ark files.

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

    def __init__(
        self, file_path, path_prefix=None, transform=None, permissive=False, scp_sep=" "
    ):
        super(RandomAccessArkDataReader, self).__init__(
            file_path, transform, permissive
        )

        self.scp = SCPList.load(self.file_path, sep=scp_sep)
        if path_prefix is not None:
            self.scp.add_prefix_to_filepath(path_prefix)

        archives, archive_idx = np.unique(self.scp.file_path, return_inverse=True)
        self.archives = archives
        self.archive_idx = archive_idx
        self.f = [None] * len(self.archives)
        self.locks = [threading.Lock() for i in range(len(self.archives))]

    @property
    def keys(self):
        return self.scp.key

    def close(self):
        """Closes all the open Ark files."""
        for f in self.f:
            if f is not None:
                f.close()
        self.f = [None] * len(self.f)

    def _open_archive(self, key_idx, offset=0):
        """Opens the Ark file correspoding to a given feature/matrix
           if it is not already open and moves the file pointer to the
           point where we can read that feature matrix.

           If the file was already open, it only moves the file pointer.

        Args:
          key_idx: Integer position of the feature matrix in the scp file.
          offset: Byte where we can find the feature matrix in the Ark file.

        Returns:
          Python file object.
          threading.Lock object corresponding to the file
        """
        archive_idx = self.archive_idx[key_idx]
        with self.locks[archive_idx]:
            if self.f[archive_idx] is None:
                self.f[archive_idx] = open(self.archives[archive_idx], "rb")

            f = self.f[archive_idx]
            f.seek(offset, 0)

        return f, self.locks[archive_idx]

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
        num_rows = np.array([s[0] if len(s) == 2 else 1 for s in shapes], dtype=np.int)
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
        dims = np.array([s[-1] for s in shapes], dtype=np.int)
        if assert_same_dim:
            assert np.all(dims == dims[0])
        return dims

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
            with lock:
                f.seek(offset, 0)
                binary = init_kaldi_input_stream(f)
                shape_i = KaldiMatrix.read_shape(f, binary, sequential_mode=False)

            shape_i = self._apply_range_to_shape(shape_i, row_offset_i, num_rows_i)

            shapes.append(shape_i)

        if assert_same_dim:
            dims = np.array([s[-1] for s in shapes], dtype=np.int)
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
                f.seek(offset, 0)
                binary = init_kaldi_input_stream(f)
                data_i = KaldiMatrix.read(
                    f, binary, row_offset_i, num_rows_i, sequential_mode=False
                ).to_ndarray()

            assert num_rows_i == 0 or data_i.shape[0] == num_rows_i

            if self.transform is not None:
                data_i = self.transform.predict(data_i)

            data.append(data_i)

        if squeeze:
            data = self._squeeze(data, self.permissive)

        return data
