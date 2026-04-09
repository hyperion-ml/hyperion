"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

Classes to read data from hdf5 files.
"""

import multiprocessing
import time
from multiprocessing.synchronize import Lock as ProcessLock
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np

from ..hyp_defs import float_cpu

# from ..utils.scp_list import SCPList
from ..utils import FeatureSet, PathLike
from ..utils.kaldi_io_funcs import is_token
from ..utils.kaldi_matrix import KaldiCompressedMatrix, KaldiMatrix
from ..utils.list_utils import split_list, split_list_group_by_key
from .data_reader import (
    RandomAccessDataReader,
    ReadIndex,
    SequentialDataReader,
    TransformArg,
)


def _read_h5_data(
    dset,
    row_offset: int = 0,
    num_rows: int = 0,
    transform: TransformArg = None,
) -> np.ndarray:
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
        file_path: hdf5 or scp file to read.
        transform: TransformList object, applies a transformation to the
                   features after reading them from disk.
        part_idx: It splits the input into num_parts and writes only
                  part part_idx, where part_idx=1,...,num_parts.
        num_parts: Number of parts to split the input data.
        split_by_key: If True, all the elements with the same key go to the same part.
    """

    def __init__(self, file_path: PathLike, **kwargs) -> None:
        """Initialize a sequential HDF5 reader."""
        super().__init__(file_path, **kwargs)
        self.f = None
        self.cur_file = None
        self.cur_item = 0

    def __getstate__(self) -> dict:
        """Drop process-local runtime objects when pickling for spawn workers."""
        state = self.__dict__.copy()
        state["f"] = None
        state["lock"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Recreate process-local runtime objects after unpickling in workers."""
        self.__dict__.update(state)
        self.lock = multiprocessing.Lock()
        self.f = None
        if self.cur_file is not None:
            self._open_archive(self.cur_file)

    def close(self) -> None:
        """Closes current hdf5 file."""
        if self.f is not None:
            self.f.close()
            self.f = None

    def _open_archive(self, file_path: PathLike) -> None:
        """Opens the hdf5 file where the next matrix/vector is
        if it is not open.
        If there was another hdf5 file open, it closes it.
        """
        if self.f is None or file_path != self.cur_file:
            self.close()
            self.cur_file = file_path
            self.f = h5py.File(file_path, "r")

    def read_num_rows(
        self, num_records: int = 0, assert_same_dim: bool = True
    ) -> Tuple[List[str], np.ndarray]:
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
        num_rows = np.array(
            [
                s[0] if len(s) == 2 else (0 if len(s) == 1 and s[0] == 0 else 1)
                for s in shapes
            ],
            dtype=int,
        )
        return keys, num_rows

    def read_dims(
        self, num_records: int = 0, assert_same_dim: bool = True
    ) -> Tuple[List[str], np.ndarray]:
        """Reads the number of columns in the feature matrices of the dataset.

        Args:
          num_records: How many matrices shapes to read, if num_records=0 it
                       reads all the matrices in the dataset.
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
        file_path: HDF5 file to read.
        transform: TransformList object, applies a transformation to the
                   features after reading them from disk.
        part_idx: It splits the input into num_parts and writes only
                  part part_idx, where part_idx=1,...,num_parts.
        num_parts: Number of parts to split the input data.
        split_by_key: If True, all the elements with the same key go to the same part.

     Examples:
        >>> from hyperion.io.h5_data_reader import SequentialH5FileDataReader
        >>> with SequentialH5FileDataReader("feats.h5") as reader:
        ...     keys, data = reader.read(num_records=2)
        ...     keys2, shapes2 = reader.read_shapes(num_records=2)
        ...
        >>> reader = SequentialH5FileDataReader("feats.h5")
        >>> keys, stacked = reader.read(num_records=8, squeeze=True)
        >>> reader.close()
    """

    def __init__(self, file_path: PathLike, **kwargs) -> None:
        """Initialize a sequential reader over a single HDF5 file."""
        super().__init__(file_path, permissive=False, **kwargs)
        self._open_archive(self.file_path)
        self._keys = list(self.f.keys())
        if self.num_parts > 1:
            self._keys, _ = split_list(self._keys, self.part_idx, self.num_parts)

    @property
    def keys(self) -> List[str]:
        """List keys available in the HDF5 file."""
        return self._keys

    def reset(self) -> None:
        """Puts the file pointer back to the beginning of the file"""
        if self.f is not None:
            self.cur_item = 0

    def eof(self) -> bool:
        """Returns True when it reaches the end of the HDF5 file."""
        return self.cur_item == len(self._keys)

    def read_shapes(
        self, num_records: int = 0, assert_same_dim: bool = True
    ) -> Tuple[List[str], List[Tuple[int, ...]]]:
        """Reads the shapes in the feature matrices of the dataset.

        Args:
          num_records: How many matrices shapes to read, if num_records=0 it
                       reads all the matrices in the dataset.
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

    def read(
        self,
        num_records: int = 0,
        squeeze: bool = False,
        row_offset: ReadIndex = 0,
        num_rows: ReadIndex = 0,
    ) -> Tuple[List[str], Union[List[np.ndarray], np.ndarray]]:
        """Reads next num_records feature matrices/vectors.

        Args:
          num_records: Number of feature matrices to read.
          squeeze: If True, it converts the list of
                   matrices/vectors to 3D/2D numpy array.
                   All matrices need to have same number of rows.
          row_offset: List of integers or numpy array with the first row to
                  read from each feature matrix.
          num_rows: List of integers or numpy array with the
                    number of rows to read from each feature matrix.
                    If 0 it reads all the rows.

        Returns:
          key: List of recording names.
          data: List of feature matrices/vectors or 3D/2D numpy array.
        """
        if num_records == 0:
            num_records = len(self._keys) - self.cur_item
        else:
            num_records = min(num_records, len(self._keys) - self.cur_item)

        row_offset_is_list = isinstance(row_offset, (list, np.ndarray))
        num_rows_is_list = isinstance(num_rows, (list, np.ndarray))
        if isinstance(row_offset, np.ndarray) and row_offset.ndim == 0:
            row_offset_is_list = False
        if isinstance(num_rows, np.ndarray) and num_rows.ndim == 0:
            num_rows_is_list = False
        if row_offset_is_list and len(row_offset) < num_records:
            raise ValueError(
                f"row_offset has {len(row_offset)} items but {num_records} are required"
            )
        if num_rows_is_list and len(num_rows) < num_records:
            raise ValueError(
                f"num_rows has {len(num_rows)} items but {num_records} are required"
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
        transform: TransformList object, applies a transformation to the
                   features after reading them from disk.
        part_idx: It splits the input into num_parts and writes only
                  part part_idx, where part_idx=1,...,num_parts.
        num_parts: Number of parts to split the input data.
        split_by_key: If True, all the elements with the same key go to the same part.

     Examples:
        >>> from hyperion.io.h5_data_reader import SequentialH5ScriptDataReader
        >>> reader = SequentialH5ScriptDataReader("feats.scp")
        >>> keys, data = reader.read(num_records=4)
        >>> keys2, data2 = reader.read(num_records=2, row_offset=20, num_rows=80)
        >>> reader.reset()
        >>> keys3, shapes3 = reader.read_shapes(num_records=3)
        >>> reader.close()
    """

    def __init__(
        self, file_path: PathLike, path_prefix: Optional[PathLike] = None, **kwargs
    ) -> None:
        """Initialize a sequential reader over HDF5 files indexed by scp."""
        super().__init__(file_path, permissive=False, **kwargs)

        self.feature_set = FeatureSet.load(self.file_path)
        if self.num_parts > 1:
            self.feature_set = self.feature_set.split(self.part_idx, self.num_parts)
        if path_prefix is not None:
            self.feature_set.add_prefix_to_storage_path(path_prefix)

    @property
    def keys(self) -> np.ndarray:
        """Array of recording keys in read order."""
        return self.feature_set["id"].values

    def reset(self) -> None:
        """Closes all the open hdf5 files and puts the read pointer pointing
        to the first element in the scp file."""
        self.close()
        self.cur_item = 0

    def eof(self) -> bool:
        """Returns True when all the elements in the scp have been read."""
        return self.cur_item == len(self.feature_set)

    def read_shapes(
        self, num_records: int = 0, assert_same_dim: bool = True
    ) -> Tuple[List[str], List[Tuple[int, ...]]]:
        """Reads the shapes in the feature matrices of the dataset.

        Args:
          num_records: How many matrices shapes to read, if num_records=0 it
                       reads all the matrices in the dataset.
          assert_same_dim: If True, it raise exception in not all the matrices have
                           the same number of columns.

        Returns:
          List of num_records recording names.
          List of tuples with num_records shapes.
        """
        if num_records == 0:
            num_records = len(self.feature_set) - self.cur_item

        keys = []
        shapes = []
        for i in range(num_records):
            if self.eof():
                break

            feature_spec = self.feature_set.iloc[self.cur_item]
            key = feature_spec["id"]

            self._open_archive(feature_spec["storage_path"])
            shape_i = self.f[key].shape
            if "start" in feature_spec and "num_frames" in feature_spec:
                range_spec = [feature_spec["start"], feature_spec["num_frames"]]
                row_offset_i, num_rows_i = self._combine_ranges(range_spec, 0, 0)
                shape_i = self._apply_range_to_shape(shape_i, row_offset_i, num_rows_i)

            keys.append(key)
            shapes.append(shape_i)
            self.cur_item += 1

        if assert_same_dim and len(shapes) > 0:
            dims = np.array([s[-1] for s in shapes], dtype=np.int32)
            assert np.all(dims == dims[0])

        return keys, shapes

    def read(
        self,
        num_records: int = 0,
        squeeze: bool = False,
        row_offset: ReadIndex = 0,
        num_rows: ReadIndex = 0,
    ) -> Tuple[List[str], Union[List[np.ndarray], np.ndarray]]:
        """Reads next num_records feature matrices/vectors.

        Args:
          num_records: Number of feature matrices to read.
          squeeze: If True, it converts the list of
                   matrices/vectors to 3D/2D numpy array.
                   All matrices need to have same number of rows.
          row_offset: List of integers or numpy array with the first row to
                  read from each feature matrix.
          num_rows: List of integers or numpy array with the
                    number of rows to read from each feature matrix.
                    If 0 it reads all the rows.

        Returns:
          key: List of recording names.
          data: List of feature matrices/vectors or 3D/2D numpy array.
        """
        if num_records == 0:
            num_records = len(self.feature_set) - self.cur_item
        else:
            num_records = min(num_records, len(self.feature_set) - self.cur_item)

        row_offset_is_list = isinstance(row_offset, (list, np.ndarray))
        num_rows_is_list = isinstance(num_rows, (list, np.ndarray))
        if isinstance(row_offset, np.ndarray) and row_offset.ndim == 0:
            row_offset_is_list = False
        if isinstance(num_rows, np.ndarray) and num_rows.ndim == 0:
            num_rows_is_list = False
        if row_offset_is_list and len(row_offset) < num_records:
            raise ValueError(
                f"row_offset has {len(row_offset)} items but {num_records} are required"
            )
        if num_rows_is_list and len(num_rows) < num_records:
            raise ValueError(
                f"num_rows has {len(num_rows)} items but {num_records} are required"
            )

        keys = []
        data = []
        with self.lock:
            for i in range(num_records):
                if self.eof():
                    break

                feature_spec = self.feature_set.iloc[self.cur_item]
                key = feature_spec["id"]
                file_path = feature_spec["storage_path"]
                if "start" in feature_spec and "num_frames" in feature_spec:
                    range_spec = [feature_spec["start"], feature_spec["num_frames"]]
                else:
                    range_spec = None

                row_offset_i = row_offset[i] if row_offset_is_list else row_offset
                num_rows_i = num_rows[i] if num_rows_is_list else num_rows
                row_offset_i, num_rows_i = self._combine_ranges(
                    range_spec, row_offset_i, num_rows_i
                )

                self._open_archive(file_path)

                dset_i = self.f[key]
                data_i = _read_h5_data(dset_i, row_offset_i, num_rows_i, self.transform)
                self.cur_item += 1

                keys.append(key)
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

    def __init__(
        self,
        file_path: PathLike,
        transform: TransformArg = None,
        permissive: bool = False,
    ) -> None:
        """Initialize a random-access HDF5 reader."""
        super().__init__(file_path, transform, permissive)
        self.f = None

    def read_num_rows(
        self, keys: Union[str, List[str], np.ndarray], assert_same_dim: bool = True
    ) -> np.ndarray:
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
        num_rows = np.array(
            [
                s[0] if len(s) == 2 else (0 if len(s) == 1 and s[0] == 0 else 1)
                for s in shapes
            ],
            dtype=int,
        )
        return num_rows

    def read_dims(
        self, keys: Union[str, List[str], np.ndarray], assert_same_dim: bool = True
    ) -> np.ndarray:
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
        if assert_same_dim and len(dims) > 0:
            if self.permissive:
                dims_to_assert = np.array(
                    [s[-1] for s in shapes if not (len(s) == 1 and s[0] == 0)],
                    dtype=np.int32,
                )
            else:
                dims_to_assert = dims
            if len(dims_to_assert) > 0:
                assert np.all(dims_to_assert == dims_to_assert[0])
        return dims


class RandomAccessH5FileDataReader(RandomAccessH5DataReader):
    """Class to read from a single hdf5 file in random order

    Attributes:
       file_path: HDF5 file to read.
       transform: TransformList object, applies a transformation to the
                  features after reading them from disk.
       permissive: If True, if the data that we want to read is not in the file
                   it returns an empty matrix, if False it raises an exception.

    Examples:
       >>> from hyperion.io.h5_data_reader import RandomAccessH5FileDataReader
       >>> with RandomAccessH5FileDataReader("feats.h5") as reader:
       ...     data = reader.read(["utt1", "utt2"])
       ...     dims = reader.read_dims(["utt1", "utt2"])
       ...     sliced = reader.read(["utt1", "utt2"], row_offset=[0, 10], num_rows=[50, 40])
    """

    def __init__(self, file_path: PathLike, **kwargs) -> None:
        """Initialize random access to a single HDF5 file."""
        super().__init__(file_path, **kwargs)
        self.lock = multiprocessing.Lock()
        self._open_archive(file_path)

    def __getstate__(self) -> dict:
        """Drop process-local runtime objects when pickling for spawn workers."""
        state = self.__dict__.copy()
        state["f"] = None
        state["lock"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Recreate process-local runtime objects after unpickling in workers."""
        self.__dict__.update(state)
        self.lock = multiprocessing.Lock()
        self.f = None
        self._open_archive(self.file_path)

    def close(self) -> None:
        """Closes the hdf5 files."""
        if self.f is not None:
            self.f.close()
            self.f = None

    def _open_archive(self, file_path: PathLike) -> None:
        """Open the hdf5 file it it is not open."""
        if self.f is None:
            self.close()
            self.f = h5py.File(file_path, "r")

    @property
    def keys(self) -> List[str]:
        """List keys available in the HDF5 file."""
        return list(self.f.keys())

    def read_shapes(
        self, keys: Union[str, List[str], np.ndarray], assert_same_dim: bool = True
    ) -> List[Tuple[int, ...]]:
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

        if assert_same_dim and len(shapes) > 0:
            if self.permissive:
                dims = np.array(
                    [s[-1] for s in shapes if not (len(s) == 1 and s[0] == 0)],
                    dtype=np.int32,
                )
            else:
                dims = np.array([s[-1] for s in shapes], dtype=np.int32)
            if len(dims) > 0:
                assert np.all(dims == dims[0])

        return shapes

    def read(
        self,
        keys: Union[str, List[str], np.ndarray],
        squeeze: bool = False,
        row_offset: ReadIndex = 0,
        num_rows: ReadIndex = 0,
    ) -> Union[List[np.ndarray], np.ndarray]:
        """Reads the feature matrices/vectors for the recordings in keys.

        Args:
          keys: List of recording names from which we want to retrieve the
                feature matrices/vectors.
          squeeze: If True, it converts the list of
                   matrices/vectors to 3D/2D numpy array.
                   All matrices need to have same number of rows.
          row_offset: List of integers or numpy array with the first row to
                  read from each feature matrix.
          num_rows: List of integers or numpy array with the
                    number of rows to read from each feature matrix.
                    If 0 it reads all the rows.

        Returns:
          data: List of feature matrices/vectors or 3D/2D numpy array.
        """
        if isinstance(keys, str):
            keys = [keys]

        row_offset_is_list = isinstance(row_offset, (list, np.ndarray))
        num_rows_is_list = isinstance(num_rows, (list, np.ndarray))
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

    Examples:
        >>> from hyperion.io.h5_data_reader import RandomAccessH5ScriptDataReader
        >>> with RandomAccessH5ScriptDataReader("feats.scp") as reader:
        ...     data = reader.read(["utt1", "utt2"])
        ...     shapes = reader.read_shapes(["utt1", "utt2"])
        ...     data2 = reader.read(["missing", "utt2"], row_offset=0, num_rows=100)
    """

    def __init__(
        self, file_path: PathLike, path_prefix: Optional[PathLike] = None, **kwargs
    ) -> None:
        """Initialize random access over HDF5 files indexed by scp."""
        super().__init__(file_path, **kwargs)

        self.feature_set = FeatureSet.load(self.file_path)
        if path_prefix is not None:
            self.feature_set.add_prefix_to_storage_path(path_prefix)

        archives, archive_idx = np.unique(
            self.feature_set["storage_path"], return_inverse=True
        )
        self.archives = archives
        self.archive_idx = archive_idx
        self.f = [None] * len(self.archives)
        self.locks = [multiprocessing.Lock() for i in range(len(self.archives))]

    def __getstate__(self) -> dict:
        """Drop process-local runtime objects when pickling for spawn workers."""
        state = self.__dict__.copy()
        state["f"] = None
        state["locks"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Recreate process-local runtime objects after unpickling in workers."""
        self.__dict__.update(state)
        self.f = [None] * len(self.archives)
        self.locks = [multiprocessing.Lock() for _ in range(len(self.archives))]

    def close(self) -> None:
        """Closes all the open hdf5 files."""
        for f in self.f:
            if f is not None:
                f.close()
        self.f = [None] * len(self.f)

    @property
    def keys(self) -> np.ndarray:
        """Array of keys that can be queried."""
        return self.feature_set["id"].values

    def _open_archive(self, key_idx: int) -> Tuple[object, ProcessLock]:
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

    def read_shapes(
        self, keys: Union[str, List[str], np.ndarray], assert_same_dim: bool = True
    ) -> List[Tuple[int, ...]]:
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

            if not (key in self.feature_set.index):
                if self.permissive:
                    shapes.append((0,))
                    continue
                else:
                    raise Exception("Key %s not found" % key)

            index = self.feature_set.get_loc(key)
            feature_spec = self.feature_set.loc[key]
            f, lock = self._open_archive(index)
            if not (key in f):
                if self.permissive:
                    shapes.append((0,))
                    continue
                else:
                    raise Exception("Key %s not found" % key)

            with lock:
                shape_i = f[key].shape

            if "start" in feature_spec and "num_frames" in feature_spec:
                range_spec = [feature_spec["start"], feature_spec["num_frames"]]
                row_offset_i, num_rows_i = self._combine_ranges(range_spec, 0, 0)
                shape_i = self._apply_range_to_shape(shape_i, row_offset_i, num_rows_i)

            shapes.append(shape_i)

        if assert_same_dim and len(shapes) > 0:
            if self.permissive:
                dims = np.array(
                    [s[-1] for s in shapes if not (len(s) == 1 and s[0] == 0)],
                    dtype=np.int32,
                )
            else:
                dims = np.array([s[-1] for s in shapes], dtype=np.int32)
            if len(dims) > 0:
                assert np.all(dims == dims[0])

        return shapes

    def read(
        self,
        keys: Union[str, List[str], np.ndarray],
        squeeze: bool = False,
        row_offset: ReadIndex = 0,
        num_rows: ReadIndex = 0,
    ) -> Union[List[np.ndarray], np.ndarray]:
        """Reads the feature matrices/vectors for the recordings in keys.

        Args:
          keys: List of recording names from which we want to retrieve the
                feature matrices/vectors.
          squeeze: If True, it converts the list of
                   matrices/vectors to 3D/2D numpy array.
                   All matrices need to have same number of rows.
          row_offset: List of integers or numpy array with the first row to
                  read from each feature matrix.
          num_rows: List of integers or numpy array with the
                    number of rows to read from each feature matrix.
                    If 0 it reads all the rows.

        Returns:
          data: List of feature matrices/vectors or 3D/2D numpy array.
        """
        if isinstance(keys, str):
            keys = [keys]

        row_offset_is_list = isinstance(row_offset, (list, np.ndarray))
        num_rows_is_list = isinstance(num_rows, (list, np.ndarray))
        if row_offset_is_list:
            assert len(row_offset) == len(keys)
        if num_rows_is_list:
            assert len(num_rows) == len(keys)

        data = []
        for i, key in enumerate(keys):

            if not (key in self.feature_set.index):
                if self.permissive:
                    data.append(np.array([], dtype=float_cpu()))
                    continue
                else:
                    raise Exception("Key %s not found" % key)

            index = self.feature_set.get_loc(key)
            feature_spec = self.feature_set.loc[key]
            if "start" in feature_spec and "num_frames" in feature_spec:
                range_spec = [feature_spec["start"], feature_spec["num_frames"]]
            else:
                range_spec = None

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


# class SequentialH5DataReader(SequentialDataReader):
#     """Abstract base class to read hdf5 feature files in
#     sequential order.

#      Attributes:
#         file_path: hdf5 or scp file to read.
#         transform: TransformList object, applies a transformation to the
#                    features after reading them from disk.
#         part_idx: It splits the input into num_parts and writes only
#                   part part_idx, where part_idx=1,...,num_parts.
#         num_parts: Number of parts to split the input data.
#         split_by_key: If True, all the elements with the same key go to the same part.
#     """

#     def __init__(self, file_path: PathLike, **kwargs) -> None:
#         """Initialize a sequential HDF5 reader."""
#         super().__init__(file_path, **kwargs)
#         self.f = None
#         self.cur_file = None
#         self.cur_item = 0

#     def close(self) -> None:
#         """Closes current hdf5 file."""
#         if self.f is not None:
#             self.f.close()
#             self.f = None

#     def _open_archive(self, file_path: PathLike) -> None:
#         """Opens the hdf5 file where the next matrix/vector is
#         if it is not open.
#         If there was another hdf5 file open, it closes it.
#         """
#         if self.f is None or file_path != self.cur_file:
#             self.close()
#             self.cur_file = file_path
#             self.f = h5py.File(file_path, "r")

#     def read_num_rows(
#         self, num_records: int = 0, assert_same_dim: bool = True
#     ) -> Tuple[List[str], np.ndarray]:
#         """Reads the number of rows in the feature matrices of the dataset.

#         Args:
#           num_records: How many matrices shapes to read, if num_records=0 it
#                        reads all the matrices in the dataset.
#           assert_same_dim: If True, it raise exception in not all the matrices have
#                            the same number of columns.

#         Returns:
#           List of num_records recording names.
#           Integer numpy array with num_records number of rows.
#         """
#         keys, shapes = self.read_shapes(num_records, assert_same_dim)
#         num_rows = np.array(
#             [s[0] if len(s) == 2 else (0 if len(s) == 1 and s[0] == 0 else 1) for s in shapes],
#             dtype=int,
#         )
#         return keys, num_rows

#     def read_dims(
#         self, num_records: int = 0, assert_same_dim: bool = True
#     ) -> Tuple[List[str], np.ndarray]:
#         """Reads the number of columns in the feature matrices of the dataset.

#         Args:
#           num_records: How many matrices shapes to read, if num_records=0 it
#                        reads all the matrices in the dataset.
#           assert_same_dim: If True, it raise exception in not all the matrices have
#                            the same number of columns.

#         Returns:
#           List of num_records recording names.
#           Integer numpy array with num_records number of columns.
#         """
#         keys, shapes = self.read_shapes(num_records, False)
#         dims = np.array([s[-1] for s in shapes], dtype=np.int32)
#         if assert_same_dim and len(dims) > 0:
#             assert np.all(dims == dims[0])
#         return keys, dims


# class SequentialH5FileDataReader(SequentialH5DataReader):
#     """Class to read feature matrices/vectors in
#     sequential order from a single hdf5 file.

#      Attributes:
#         file_path: HDF5 file to read.
#         transform: TransformList object, applies a transformation to the
#                    features after reading them from disk.
#         part_idx: It splits the input into num_parts and writes only
#                   part part_idx, where part_idx=1,...,num_parts.
#         num_parts: Number of parts to split the input data.
#         split_by_key: If True, all the elements with the same key go to the same part.
#     """

#     def __init__(self, file_path: PathLike, **kwargs) -> None:
#         """Initialize a sequential reader over a single HDF5 file."""
#         super().__init__(file_path, permissive=False, **kwargs)
#         self._open_archive(self.file_path)
#         self._keys = list(self.f.keys())
#         if self.num_parts > 1:
#             self._keys, _ = split_list(self._keys, self.part_idx, self.num_parts)

#     @property
#     def keys(self) -> List[str]:
#         """List keys available in the HDF5 file."""
#         return self._keys

#     def reset(self) -> None:
#         """Puts the file pointer back to the beginning of the file"""
#         if self.f is not None:
#             self.cur_item = 0

#     def eof(self) -> bool:
#         """Returns True when it reaches the end of the HDF5 file."""
#         return self.cur_item == len(self._keys)

#     def read_shapes(
#         self, num_records: int = 0, assert_same_dim: bool = True
#     ) -> Tuple[List[str], List[Tuple[int, ...]]]:
#         """Reads the shapes in the feature matrices of the dataset.

#         Args:
#           num_records: How many matrices shapes to read, if num_records=0 it
#                        reads all the matrices in the dataset.
#           assert_same_dim: If True, it raise exception in not all the matrices have
#                            the same number of columns.

#         Returns:
#           List of num_records recording names.
#           List of tuples with num_records shapes.
#         """
#         if num_records == 0:
#             num_records = len(self._keys) - self.cur_item

#         keys = []
#         shapes = []
#         for i in range(num_records):
#             if self.eof():
#                 break
#             key = self._keys[self.cur_item]
#             keys.append(key)
#             shapes.append(self.f[key].shape)
#             self.cur_item += 1

#         if assert_same_dim and len(shapes) > 0:
#             dims = np.array([s[-1] for s in shapes], dtype=np.int32)
#             assert np.all(dims == dims[0])

#         return keys, shapes

#     def read(
#         self,
#         num_records: int = 0,
#         squeeze: bool = False,
#         row_offset: ReadIndex = 0,
#         num_rows: ReadIndex = 0,
#     ) -> Tuple[List[str], Union[List[np.ndarray], np.ndarray]]:
#         """Reads next num_records feature matrices/vectors.

#         Args:
#           num_records: Number of feature matrices to read.
#           squeeze: If True, it converts the list of
#                    matrices/vectors to 3D/2D numpy array.
#                    All matrices need to have same number of rows.
#           row_offset: List of integers or numpy array with the first row to
#                   read from each feature matrix.
#           num_rows: List of integers or numpy array with the
#                     number of rows to read from each feature matrix.
#                     If 0 it reads all the rows.

#         Returns:
#           key: List of recording names.
#           data: List of feature matrices/vectors or 3D/2D numpy array.
#         """
#         if num_records == 0:
#             num_records = len(self._keys) - self.cur_item
#         else:
#             num_records = min(num_records, len(self._keys) - self.cur_item)

#         row_offset_is_list = isinstance(row_offset, (list, np.ndarray))
#         num_rows_is_list = isinstance(num_rows, (list, np.ndarray))
#         if isinstance(row_offset, np.ndarray) and row_offset.ndim == 0:
#             row_offset_is_list = False
#         if isinstance(num_rows, np.ndarray) and num_rows.ndim == 0:
#             num_rows_is_list = False
#         if row_offset_is_list and len(row_offset) < num_records:
#             raise ValueError(
#                 f"row_offset has {len(row_offset)} items but {num_records} are required"
#             )
#         if num_rows_is_list and len(num_rows) < num_records:
#             raise ValueError(
#                 f"num_rows has {len(num_rows)} items but {num_records} are required"
#             )
#         keys = []
#         data = []
#         with self.lock:
#             for i in range(num_records):
#                 if self.eof():
#                     break

#                 key_i = self._keys[self.cur_item]

#                 row_offset_i = row_offset[i] if row_offset_is_list else row_offset
#                 num_rows_i = num_rows[i] if num_rows_is_list else num_rows

#                 dset_i = self.f[key_i]
#                 data_i = _read_h5_data(dset_i, row_offset_i, num_rows_i, self.transform)

#                 self.cur_item += 1

#                 keys.append(key_i)
#                 data.append(data_i)

#         if squeeze:
#             data = self._squeeze(data)

#         return keys, data


# class SequentialH5ScriptDataReader(SequentialH5DataReader):
#     """Class to read features from multiple hdf5 files where a scp file
#     indicates which hdf5 file contains each feature matrix.

#      Attributes:
#         file_path: scp file to read.
#         path_prefix: If input_spec is a scp file, it pre-appends
#                      path_prefix string to the second column of
#                      the scp file. This is useful when data
#                      is read from a different directory of that
#                      it was created.
#         transform: TransformList object, applies a transformation to the
#                    features after reading them from disk.
#         part_idx: It splits the input into num_parts and writes only
#                   part part_idx, where part_idx=1,...,num_parts.
#         num_parts: Number of parts to split the input data.
#         split_by_key: If True, all the elements with the same key go to the same part.
#     """

#     def __init__(
#         self, file_path: PathLike, path_prefix: Optional[PathLike] = None, **kwargs
#     ) -> None:
#         """Initialize a sequential reader over HDF5 files indexed by scp."""
#         super().__init__(file_path, permissive=False, **kwargs)

#         self.feature_set = FeatureSet.load(self.file_path)
#         if self.num_parts > 1:
#             self.feature_set = self.feature_set.split(self.part_idx, self.num_parts)
#         if path_prefix is not None:
#             self.feature_set.add_prefix_to_storage_path(path_prefix)

#     @property
#     def keys(self) -> np.ndarray:
#         """Array of recording keys in read order."""
#         return self.feature_set["id"].values

#     def reset(self) -> None:
#         """Closes all the open hdf5 files and puts the read pointer pointing
#         to the first element in the scp file."""
#         self.close()
#         self.cur_item = 0

#     def eof(self) -> bool:
#         """Returns True when all the elements in the scp have been read."""
#         return self.cur_item == len(self.feature_set)

#     def read_shapes(
#         self, num_records: int = 0, assert_same_dim: bool = True
#     ) -> Tuple[List[str], List[Tuple[int, ...]]]:
#         """Reads the shapes in the feature matrices of the dataset.

#         Args:
#           num_records: How many matrices shapes to read, if num_records=0 it
#                        reads all the matrices in the dataset.
#           assert_same_dim: If True, it raise exception in not all the matrices have
#                            the same number of columns.

#         Returns:
#           List of num_records recording names.
#           List of tuples with num_records shapes.
#         """
#         if num_records == 0:
#             num_records = len(self.feature_set) - self.cur_item

#         keys = []
#         shapes = []
#         for i in range(num_records):
#             if self.eof():
#                 break

#             feature_spec = self.feature_set.iloc[self.cur_item]
#             key = feature_spec["id"]

#             self._open_archive(feature_spec["storage_path"])
#             shape_i = self.f[key].shape
#             if "start" in feature_spec and "num_frames" in feature_spec:
#                 range_spec = [feature_spec["start"], feature_spec["num_frames"]]
#                 row_offset_i, num_rows_i = self._combine_ranges(range_spec, 0, 0)
#                 shape_i = self._apply_range_to_shape(shape_i, row_offset_i, num_rows_i)

#             keys.append(key)
#             shapes.append(shape_i)
#             self.cur_item += 1

#         if assert_same_dim and len(shapes) > 0:
#             dims = np.array([s[-1] for s in shapes], dtype=np.int32)
#             assert np.all(dims == dims[0])

#         return keys, shapes

#     def read(
#         self,
#         num_records: int = 0,
#         squeeze: bool = False,
#         row_offset: ReadIndex = 0,
#         num_rows: ReadIndex = 0,
#     ) -> Tuple[List[str], Union[List[np.ndarray], np.ndarray]]:
#         """Reads next num_records feature matrices/vectors.

#         Args:
#           num_records: Number of feature matrices to read.
#           squeeze: If True, it converts the list of
#                    matrices/vectors to 3D/2D numpy array.
#                    All matrices need to have same number of rows.
#           row_offset: List of integers or numpy array with the first row to
#                   read from each feature matrix.
#           num_rows: List of integers or numpy array with the
#                     number of rows to read from each feature matrix.
#                     If 0 it reads all the rows.

#         Returns:
#           key: List of recording names.
#           data: List of feature matrices/vectors or 3D/2D numpy array.
#         """
#         if num_records == 0:
#             num_records = len(self.feature_set) - self.cur_item
#         else:
#             num_records = min(num_records, len(self.feature_set) - self.cur_item)

#         row_offset_is_list = isinstance(row_offset, (list, np.ndarray))
#         num_rows_is_list = isinstance(num_rows, (list, np.ndarray))
#         if isinstance(row_offset, np.ndarray) and row_offset.ndim == 0:
#             row_offset_is_list = False
#         if isinstance(num_rows, np.ndarray) and num_rows.ndim == 0:
#             num_rows_is_list = False
#         if row_offset_is_list and len(row_offset) < num_records:
#             raise ValueError(
#                 f"row_offset has {len(row_offset)} items but {num_records} are required"
#             )
#         if num_rows_is_list and len(num_rows) < num_records:
#             raise ValueError(
#                 f"num_rows has {len(num_rows)} items but {num_records} are required"
#             )

#         keys = []
#         data = []
#         with self.lock:
#             for i in range(num_records):
#                 if self.eof():
#                     break

#                 feature_spec = self.feature_set.iloc[self.cur_item]
#                 key = feature_spec["id"]
#                 file_path = feature_spec["storage_path"]
#                 if "start" in feature_spec and "num_frames" in feature_spec:
#                     range_spec = [feature_spec["start"], feature_spec["num_frames"]]
#                 else:
#                     range_spec = None

#                 row_offset_i = row_offset[i] if row_offset_is_list else row_offset
#                 num_rows_i = num_rows[i] if num_rows_is_list else num_rows
#                 row_offset_i, num_rows_i = self._combine_ranges(
#                     range_spec, row_offset_i, num_rows_i
#                 )

#                 self._open_archive(file_path)

#                 dset_i = self.f[key]
#                 data_i = _read_h5_data(dset_i, row_offset_i, num_rows_i, self.transform)
#                 self.cur_item += 1

#                 keys.append(key)
#                 data.append(data_i)

#         if squeeze:
#             data = self._squeeze(data)

#         return keys, data


# class RandomAccessH5DataReader(RandomAccessDataReader):
#     """Abstract base class to read hdf5 feature files in
#     random order.

#      Attributes:
#         file_path: hdf5 or scp file to read.
#         transform: TransformList object, applies a transformation to the
#                    features after reading them from disk.
#         permissive: If True, if the data that we want to read is not in the file
#                     it returns an empty matrix, if False it raises an exception.
#     """

#     def __init__(
#         self,
#         file_path: PathLike,
#         transform: TransformArg = None,
#         permissive: bool = False,
#     ) -> None:
#         """Initialize a random-access HDF5 reader."""
#         super().__init__(file_path, transform, permissive)
#         self.f = None

#     def read_num_rows(
#         self, keys: Union[str, List[str], np.ndarray], assert_same_dim: bool = True
#     ) -> np.ndarray:
#         """Reads the number of rows in the feature matrices of the dataset.

#         Args:
#           keys: List of recording names from which we want to retrieve the
#                 number of rows.
#           assert_same_dim: If True, it raise exception in not all the matrices have
#                            the same number of columns.

#         Returns:
#           Integer numpy array with the number of rows for the recordings in keys.
#         """
#         shapes = self.read_shapes(keys, assert_same_dim)
#         num_rows = np.array(
#             [s[0] if len(s) == 2 else (0 if len(s) == 1 and s[0] == 0 else 1) for s in shapes],
#             dtype=int,
#         )
#         return num_rows

#     def read_dims(
#         self, keys: Union[str, List[str], np.ndarray], assert_same_dim: bool = True
#     ) -> np.ndarray:
#         """Reads the number of columns in the feature matrices of the dataset.

#         Args:
#           keys: List of recording names from which we want to retrieve the
#                 number of columns.
#           assert_same_dim: If True, it raise exception in not all the matrices have
#                            the same number of columns.

#         Returns:
#           Integer numpy array with the number of columns for the recordings in keys
#         """
#         shapes = self.read_shapes(keys, False)
#         dims = np.array([s[-1] for s in shapes], dtype=np.int32)
#         if assert_same_dim and len(dims) > 0:
#             if self.permissive:
#                 dims_to_assert = np.array(
#                     [s[-1] for s in shapes if not (len(s) == 1 and s[0] == 0)],
#                     dtype=np.int32,
#                 )
#             else:
#                 dims_to_assert = dims
#             if len(dims_to_assert) > 0:
#                 assert np.all(dims_to_assert == dims_to_assert[0])
#         return dims


# class RandomAccessH5FileDataReader(RandomAccessH5DataReader):
#     """Class to read from a single hdf5 file in random order

#     Attributes:
#        file_path: HDF5 file to read.
#        transform: TransformList object, applies a transformation to the
#                   features after reading them from disk.
#        permissive: If True, if the data that we want to read is not in the file
#                    it returns an empty matrix, if False it raises an exception.
#     """

#     def __init__(self, file_path: PathLike, **kwargs) -> None:
#         """Initialize random access to a single HDF5 file."""
#         super().__init__(file_path, **kwargs)
#         self.lock = multiprocessing.Lock()
#         self._open_archive(file_path)

#     def close(self) -> None:
#         """Closes the hdf5 files."""
#         if self.f is not None:
#             self.f.close()
#             self.f = None

#     def _open_archive(self, file_path: PathLike) -> None:
#         """Open the hdf5 file it it is not open."""
#         if self.f is None:
#             self.close()
#             self.f = h5py.File(file_path, "r")

#     @property
#     def keys(self) -> List[str]:
#         """List keys available in the HDF5 file."""
#         return list(self.f.keys())

#     def read_shapes(
#         self, keys: Union[str, List[str], np.ndarray], assert_same_dim: bool = True
#     ) -> List[Tuple[int, ...]]:
#         """Reads the shapes in the feature matrices of the dataset.

#         Args:
#           keys: List of recording names from which we want to retrieve the
#                 shapes.
#           assert_same_dim: If True, it raise exception in not all the matrices have
#                            the same number of columns.

#         Returns:
#           List of tuples with the shapes for the recordings in keys.
#         """
#         if isinstance(keys, str):
#             keys = [keys]

#         shapes = []
#         for key in keys:

#             if not (key in self.f):
#                 if self.permissive:
#                     shapes.append((0,))
#                     continue
#                 else:
#                     raise Exception("Key %s not found" % key)

#             shape_i = self.f[key].shape
#             shapes.append(shape_i)

#         if assert_same_dim and len(shapes) > 0:
#             if self.permissive:
#                 dims = np.array(
#                     [s[-1] for s in shapes if not (len(s) == 1 and s[0] == 0)],
#                     dtype=np.int32,
#                 )
#             else:
#                 dims = np.array([s[-1] for s in shapes], dtype=np.int32)
#             if len(dims) > 0:
#                 assert np.all(dims == dims[0])

#         return shapes

#     def read(
#         self,
#         keys: Union[str, List[str], np.ndarray],
#         squeeze: bool = False,
#         row_offset: ReadIndex = 0,
#         num_rows: ReadIndex = 0,
#     ) -> Union[List[np.ndarray], np.ndarray]:
#         """Reads the feature matrices/vectors for the recordings in keys.

#         Args:
#           keys: List of recording names from which we want to retrieve the
#                 feature matrices/vectors.
#           squeeze: If True, it converts the list of
#                    matrices/vectors to 3D/2D numpy array.
#                    All matrices need to have same number of rows.
#           row_offset: List of integers or numpy array with the first row to
#                   read from each feature matrix.
#           num_rows: List of integers or numpy array with the
#                     number of rows to read from each feature matrix.
#                     If 0 it reads all the rows.

#         Returns:
#           data: List of feature matrices/vectors or 3D/2D numpy array.
#         """
#         if isinstance(keys, str):
#             keys = [keys]

#         row_offset_is_list = isinstance(row_offset, (list, np.ndarray))
#         num_rows_is_list = isinstance(num_rows, (list, np.ndarray))
#         if row_offset_is_list:
#             assert len(row_offset) == len(keys)
#         if num_rows_is_list:
#             assert len(num_rows) == len(keys)

#         data = []
#         for i, key in enumerate(keys):

#             if not (key in self.f):
#                 if self.permissive:
#                     data.append(np.array([], dtype=float_cpu()))
#                     continue
#                 else:
#                     raise Exception("Key %s not found" % key)

#             row_offset_i = row_offset[i] if row_offset_is_list else row_offset
#             num_rows_i = num_rows[i] if num_rows_is_list else num_rows

#             with self.lock:
#                 dset_i = self.f[key]
#                 data_i = _read_h5_data(dset_i, row_offset_i, num_rows_i, self.transform)
#             data.append(data_i)

#         if squeeze:
#             data = self._squeeze(data, self.permissive)

#         return data


# class RandomAccessH5ScriptDataReader(RandomAccessH5DataReader):
#     """Class to read multiple hdf5 files in random order, where a scp file
#     indicates which hdf5 file contains each feature matrix.

#     Attributes:
#         file_path: scp file to read.
#         path_prefix: If input_spec is a scp file, it pre-appends
#                      path_prefix string to the second column of
#                      the scp file. This is useful when data
#                      is read from a different directory of that
#                      it was created.
#         transform: TransformList object, applies a transformation to the
#                    features after reading them from disk.
#         permissive: If True, if the data that we want to read is not in the file
#                     it returns an empty matrix, if False it raises an exception.
#     """

#     def __init__(
#         self, file_path: PathLike, path_prefix: Optional[PathLike] = None, **kwargs
#     ) -> None:
#         """Initialize random access over HDF5 files indexed by scp."""
#         super().__init__(file_path, **kwargs)

#         self.feature_set = FeatureSet.load(self.file_path)
#         if path_prefix is not None:
#             self.feature_set.add_prefix_to_storage_path(path_prefix)

#         archives, archive_idx = np.unique(
#             self.feature_set["storage_path"], return_inverse=True
#         )
#         self.archives = archives
#         self.archive_idx = archive_idx
#         self.f = [None] * len(self.archives)
#         self.locks = [multiprocessing.Lock() for i in range(len(self.archives))]

#     def close(self) -> None:
#         """Closes all the open hdf5 files."""
#         for f in self.f:
#             if f is not None:
#                 f.close()
#         self.f = [None] * len(self.f)

#     @property
#     def keys(self) -> np.ndarray:
#         """Array of keys that can be queried."""
#         return self.feature_set["id"].values

#     def _open_archive(self, key_idx: int) -> Tuple[object, ProcessLock]:
#         """Opens the hdf5 file correspoding to a given feature/matrix
#            if it is not already open.

#         Args:
#           key_idx: Integer position of the feature matrix in the scp file.

#         Returns:
#           Python file object.
#         """
#         archive_idx = self.archive_idx[key_idx]
#         with self.locks[archive_idx]:
#             if self.f[archive_idx] is None:
#                 self.f[archive_idx] = h5py.File(self.archives[archive_idx], "r")

#         return self.f[archive_idx], self.locks[archive_idx]

#     def read_shapes(
#         self, keys: Union[str, List[str], np.ndarray], assert_same_dim: bool = True
#     ) -> List[Tuple[int, ...]]:
#         """Reads the shapes in the feature matrices of the dataset.

#         Args:
#           keys: List of recording names from which we want to retrieve the
#                 shapes.
#           assert_same_dim: If True, it raise exception in not all the matrices have
#                            the same number of columns.

#         Returns:
#           List of tuples with the shapes for the recordings in keys.
#         """
#         if isinstance(keys, str):
#             keys = [keys]
#         # t1 = time.time()
#         shapes = []
#         for key in keys:

#             if not (key in self.feature_set.index):
#                 if self.permissive:
#                     shapes.append((0,))
#                     continue
#                 else:
#                     raise Exception("Key %s not found" % key)

#             index = self.feature_set.get_loc(key)
#             feature_spec = self.feature_set.loc[key]
#             f, lock = self._open_archive(index)
#             if not (key in f):
#                 if self.permissive:
#                     shapes.append((0,))
#                     continue
#                 else:
#                     raise Exception("Key %s not found" % key)

#             with lock:
#                 shape_i = f[key].shape

#             if "start" in feature_spec and "num_frames" in feature_spec:
#                 range_spec = [feature_spec["start"], feature_spec["num_frames"]]
#                 row_offset_i, num_rows_i = self._combine_ranges(range_spec, 0, 0)
#                 shape_i = self._apply_range_to_shape(shape_i, row_offset_i, num_rows_i)

#             shapes.append(shape_i)

#         if assert_same_dim and len(shapes) > 0:
#             if self.permissive:
#                 dims = np.array(
#                     [s[-1] for s in shapes if not (len(s) == 1 and s[0] == 0)],
#                     dtype=np.int32,
#                 )
#             else:
#                 dims = np.array([s[-1] for s in shapes], dtype=np.int32)
#             if len(dims) > 0:
#                 assert np.all(dims == dims[0])

#         return shapes

#     def read(
#         self,
#         keys: Union[str, List[str], np.ndarray],
#         squeeze: bool = False,
#         row_offset: ReadIndex = 0,
#         num_rows: ReadIndex = 0,
#     ) -> Union[List[np.ndarray], np.ndarray]:
#         """Reads the feature matrices/vectors for the recordings in keys.

#         Args:
#           keys: List of recording names from which we want to retrieve the
#                 feature matrices/vectors.
#           squeeze: If True, it converts the list of
#                    matrices/vectors to 3D/2D numpy array.
#                    All matrices need to have same number of rows.
#           row_offset: List of integers or numpy array with the first row to
#                   read from each feature matrix.
#           num_rows: List of integers or numpy array with the
#                     number of rows to read from each feature matrix.
#                     If 0 it reads all the rows.

#         Returns:
#           data: List of feature matrices/vectors or 3D/2D numpy array.
#         """
#         if isinstance(keys, str):
#             keys = [keys]

#         row_offset_is_list = isinstance(row_offset, (list, np.ndarray))
#         num_rows_is_list = isinstance(num_rows, (list, np.ndarray))
#         if row_offset_is_list:
#             assert len(row_offset) == len(keys)
#         if num_rows_is_list:
#             assert len(num_rows) == len(keys)

#         data = []
#         for i, key in enumerate(keys):

#             if not (key in self.feature_set.index):
#                 if self.permissive:
#                     data.append(np.array([], dtype=float_cpu()))
#                     continue
#                 else:
#                     raise Exception("Key %s not found" % key)

#             index = self.feature_set.get_loc(key)
#             feature_spec = self.feature_set.loc[key]
#             if "start" in feature_spec and "num_frames" in feature_spec:
#                 range_spec = [feature_spec["start"], feature_spec["num_frames"]]
#             else:
#                 range_spec = None

#             row_offset_i = row_offset[i] if row_offset_is_list else row_offset
#             num_rows_i = num_rows[i] if num_rows_is_list else num_rows
#             row_offset_i, num_rows_i = self._combine_ranges(
#                 range_spec, row_offset_i, num_rows_i
#             )

#             f, lock = self._open_archive(index)
#             with lock:
#                 if not (key in f):
#                     if self.permissive:
#                         data.append(np.array([], dtype=float_cpu()))
#                         continue
#                     else:
#                         raise Exception("Key %s not found" % key)

#                 dset_i = f[key]
#                 data_i = _read_h5_data(dset_i, row_offset_i, num_rows_i, self.transform)

#             data.append(data_i)

#         if squeeze:
#             data = self._squeeze(data, self.permissive)

#         return data
