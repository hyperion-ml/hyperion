"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import multiprocessing as threading
from multiprocessing.synchronize import Lock as ProcessLock
from typing import List, Optional, Tuple, Union

import numpy as np

from ..hyp_defs import float_cpu
from ..utils import FeatureSet, PathLike
from ..utils.kaldi_io_funcs import init_kaldi_input_stream, is_token, peek, read_token
from ..utils.kaldi_matrix import KaldiCompressedMatrix, KaldiMatrix
from .data_reader import (
    RandomAccessDataReader,
    ReadIndex,
    SequentialDataReader,
    TransformArg,
)


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
    """

    def __init__(self, file_path: PathLike, **kwargs) -> None:
        """Initialize a sequential reader for a single Ark stream."""
        super().__init__(file_path, **kwargs)
        self.f = None
        self.lock = threading.Lock()
        self.cur_file = None

    def __getstate__(self) -> dict:
        """Drop process-local runtime objects when pickling for spawn workers."""
        state = self.__dict__.copy()
        f_pos = None
        if self.f is not None:
            try:
                f_pos = self.f.tell()
            except Exception:
                f_pos = None
        state["_f_pos"] = f_pos
        state["f"] = None
        state["lock"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Recreate process-local runtime objects after unpickling in workers."""
        f_pos = state.pop("_f_pos", None)
        self.__dict__.update(state)
        self.lock = threading.Lock()
        self.f = None
        if self.cur_file is not None:
            self._open_archive(self.cur_file)
            if f_pos is not None and f_pos > 0:
                self._seek(f_pos)

    def close(self) -> None:
        """Closes input file."""
        if self.f is not None:
            self.f.close()
            self.f = None

    def _seek(self, offset: int) -> None:
        """Moves the pointer of the input file.

        Args:
          offset: Byte where we want to put the pointer.
        """
        cur_pos = self.f.tell()
        delta = offset - cur_pos
        self.f.seek(delta, 1)

    def _open_archive(self, file_path: PathLike, offset: int = 0) -> None:
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

     Examples:
        >>> from hyperion.io.ark_data_reader import SequentialArkFileDataReader
        >>> with SequentialArkFileDataReader("feats.ark") as reader:
        ...     keys, data = reader.read(num_records=2)
        ...     keys2, shapes = reader.read_shapes(num_records=2)
        ...
        >>> reader = SequentialArkFileDataReader("feats.ark")
        >>> for key, mat in reader:
        ...     print(key, mat.shape)
        ...     break
        >>> reader.close()
    """

    def __init__(self, file_path: PathLike, **kwargs) -> None:
        """Initialize a sequential reader over one Ark file."""
        super().__init__(file_path, permissive=False, **kwargs)
        self._open_archive(self.file_path)
        self._eof = False
        self._keys = None
        if self.num_parts > 1:
            raise NotImplementedError(
                "Dataset splitting not available for %s" % self.__class__.__name__
            )

    def reset(self) -> None:
        """Puts the file pointer back to the beginning of the file"""
        if self.f is not None:
            self.f.seek(0, 0)
            self._eof = False

    def eof(self) -> bool:
        """Returns True when it reaches the end of the ark file."""
        return self._eof or self.f is None

    @property
    def keys(self) -> List[str]:
        """List recording keys available in the Ark file."""
        if self._keys is None:
            self.reset()
            self._keys, _ = self.read_shapes()
            self.reset()

        return self._keys

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
        row_offset_is_list = isinstance(row_offset, (list, np.ndarray))
        num_rows_is_list = isinstance(num_rows, (list, np.ndarray))
        if isinstance(row_offset, np.ndarray) and row_offset.ndim == 0:
            row_offset_is_list = False
        if isinstance(num_rows, np.ndarray) and num_rows.ndim == 0:
            num_rows_is_list = False
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

                if row_offset_is_list and count >= len(row_offset):
                    raise ValueError(
                        f"row_offset has {len(row_offset)} items but at least {count + 1} are required"
                    )
                if num_rows_is_list and count >= len(num_rows):
                    raise ValueError(
                        f"num_rows has {len(num_rows)} items but at least {count + 1} are required"
                    )

                row_offset_i = row_offset[count] if row_offset_is_list else row_offset
                num_rows_i = num_rows[count] if num_rows_is_list else num_rows

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

     Examples:
        >>> from hyperion.io.ark_data_reader import SequentialArkScriptDataReader
        >>> reader = SequentialArkScriptDataReader("feats.scp")
        >>> keys, data = reader.read(num_records=4)
        >>> keys2, data2 = reader.read(num_records=2, row_offset=10, num_rows=50)
        >>> reader.reset()
        >>> keys3, shapes3 = reader.read_shapes(num_records=3)
        >>> reader.close()
    """

    def __init__(
        self, file_path: PathLike, path_prefix: Optional[PathLike] = None, **kwargs
    ) -> None:
        """Initialize a sequential reader over Ark files indexed by scp."""
        super().__init__(file_path, permissive=False, **kwargs)
        self.feature_set = FeatureSet.load(self.file_path)

        if self.num_parts > 1:
            self.feature_set = self.feature_set.split(self.part_idx, self.num_parts)

        if path_prefix is not None:
            self.feature_set.add_prefix_to_storage_path(path_prefix)

        self.cur_item = 0

    @property
    def keys(self) -> np.ndarray:
        """Array of recording keys in read order."""
        return self.feature_set["id"].values

    def reset(self) -> None:
        """Closes all the open Ark files and puts the read pointer pointing
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
            offset = feature_spec["storage_byte"]
            file_path = feature_spec["storage_path"]
            self._open_archive(file_path, offset)
            binary = init_kaldi_input_stream(self.f)
            shape_i = KaldiMatrix.read_shape(self.f, binary, sequential_mode=True)
            if "start" in feature_spec and "num_frames" in feature_spec:
                range_spec = [feature_spec["start"], feature_spec["num_frames"]]
                row_offset_i, num_rows_i = self._combine_ranges(range_spec, 0, 0)
                shape_i = self._apply_range_to_shape(shape_i, row_offset_i, num_rows_i)

            keys.append(key)
            shapes.append(shape_i)
            self.cur_item += 1

        if assert_same_dim and len(shapes) > 0:
            dims = np.array([s[-1] for s in shapes], dtype=int)
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
                offset = feature_spec["storage_byte"]
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

     Examples:
        >>> from hyperion.io.ark_data_reader import RandomAccessArkDataReader
        >>> with RandomAccessArkDataReader("feats.scp", permissive=True) as reader:
        ...     data = reader.read(["utt1", "utt2"])
        ...     dims = reader.read_dims(["utt1", "utt2"])
        ...     sliced = reader.read(["utt1", "utt2"], row_offset=[5, 0], num_rows=[100, 50])
    """

    def __init__(
        self,
        file_path: PathLike,
        path_prefix: Optional[PathLike] = None,
        transform: TransformArg = None,
        permissive: bool = False,
    ) -> None:
        """Initialize a random-access Ark reader backed by an scp index."""
        super().__init__(file_path, transform, permissive)

        self.feature_set = FeatureSet.load(self.file_path)
        if path_prefix is not None:
            self.feature_set.add_prefix_to_storage_path(path_prefix)

        archives, archive_idx = np.unique(
            self.feature_set["storage_path"], return_inverse=True
        )
        self.archives = archives
        self.archive_idx = archive_idx
        self.f = [None] * len(self.archives)
        self.locks = [threading.Lock() for i in range(len(self.archives))]

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
        self.locks = [threading.Lock() for _ in range(len(self.archives))]

    @property
    def keys(self) -> np.ndarray:
        """Array of keys that can be queried."""
        return self.feature_set["id"].values

    def close(self) -> None:
        """Closes all the open Ark files."""
        for f in self.f:
            if f is not None:
                f.close()
        self.f = [None] * len(self.f)

    def _open_archive(
        self, key_idx: int, offset: int = 0
    ) -> Tuple[object, ProcessLock]:
        """Opens the Ark file correspoding to a given feature/matrix
           if it is not already open and moves the file pointer to the
           point where we can read that feature matrix.

           If the file was already open, it only moves the file pointer.

        Args:
          key_idx: Integer position of the feature matrix in the scp file.
          offset: Byte where we can find the feature matrix in the Ark file.

        Returns:
          Python file object.
          multiprocessing lock object corresponding to the file.
        """
        archive_idx = self.archive_idx[key_idx]
        with self.locks[archive_idx]:
            if self.f[archive_idx] is None:
                self.f[archive_idx] = open(self.archives[archive_idx], "rb")

            f = self.f[archive_idx]
            f.seek(offset, 0)

        return f, self.locks[archive_idx]

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
        dims = np.array([s[-1] for s in shapes], dtype=int)
        if assert_same_dim and len(dims) > 0:
            if self.permissive:
                dims_to_assert = np.array(
                    [s[-1] for s in shapes if not (len(s) == 1 and s[0] == 0)],
                    dtype=int,
                )
            else:
                dims_to_assert = dims
            if len(dims_to_assert) > 0:
                assert np.all(dims_to_assert == dims_to_assert[0])
        return dims

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

            if not (key in self.feature_set.index):
                if self.permissive:
                    shapes.append((0,))
                    continue
                else:
                    raise Exception("Key %s not found" % key)

            index = self.feature_set.get_loc(key)
            feature_spec = self.feature_set.loc[key]
            offset = feature_spec["storage_byte"]
            f, lock = self._open_archive(index)
            with lock:
                f.seek(offset, 0)
                binary = init_kaldi_input_stream(f)
                shape_i = KaldiMatrix.read_shape(f, binary, sequential_mode=False)

            if "start" in feature_spec and "num_frames" in feature_spec:
                range_spec = [feature_spec["start"], feature_spec["num_frames"]]
                row_offset_i, num_rows_i = self._combine_ranges(range_spec, 0, 0)
                shape_i = self._apply_range_to_shape(shape_i, row_offset_i, num_rows_i)

            shapes.append(shape_i)

        if assert_same_dim and len(shapes) > 0:
            if self.permissive:
                dims = np.array(
                    [s[-1] for s in shapes if not (len(s) == 1 and s[0] == 0)],
                    dtype=int,
                )
            else:
                dims = np.array([s[-1] for s in shapes], dtype=int)
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
            offset = feature_spec["storage_byte"]
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


# class SequentialArkDataReader(SequentialDataReader):
#     """Abstract base class to read Ark feature files in
#     sequential order.

#      Attributes:
#         file_path: ark or scp file to read.
#         transform: TransformList object, applies a transformation to the
#                    features after reading them from disk.
#         part_idx: It splits the input into num_parts and writes only
#                   part part_idx, where part_idx=1,...,num_parts.
#         num_parts: Number of parts to split the input data.
#     """

#     def __init__(self, file_path: PathLike, **kwargs) -> None:
#         """Initialize a sequential reader for a single Ark stream."""
#         super().__init__(file_path, **kwargs)
#         self.f = None
#         self.lock = threading.Lock()
#         self.cur_file = None

#     def close(self) -> None:
#         """Closes input file."""
#         if self.f is not None:
#             self.f.close()
#             self.f = None

#     def _seek(self, offset: int) -> None:
#         """Moves the pointer of the input file.

#         Args:
#           offset: Byte where we want to put the pointer.
#         """
#         cur_pos = self.f.tell()
#         delta = offset - cur_pos
#         self.f.seek(delta, 1)

#     def _open_archive(self, file_path: PathLike, offset: int = 0) -> None:
#         """Opens the current file if it is not open and moves the
#            file pointer to a given position.
#            Closes previous open Ark files.

#         Args:
#           file_path: File from which we want to read the next feature matrix.
#           offset: Byte position where feature matrix is in the file.
#         """
#         if self.f is None or file_path != self.cur_file:
#             self.close()
#             self.cur_file = file_path
#             self.f = open(file_path, "rb")

#         if offset > 0:
#             self._seek(offset)

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
#         dims = np.array([s[-1] for s in shapes], dtype=int)
#         if assert_same_dim and len(dims) > 0:
#             assert np.all(dims == dims[0])
#         return keys, dims


# class SequentialArkFileDataReader(SequentialArkDataReader):
#     """Class to read feature matrices/vectors in
#     sequential order from a single Ark file.

#      Attributes:
#         file_path: Ark file to read.
#         transform: TransformList object, applies a transformation to the
#                    features after reading them from disk.
#         part_idx: It splits the input into num_parts and writes only
#                   part part_idx, where part_idx=1,...,num_parts.
#         num_parts: Number of parts to split the input data.
#         split_by_key: If True, all the elements with the same key go to the same part.
#     """

#     def __init__(self, file_path: PathLike, **kwargs) -> None:
#         """Initialize a sequential reader over one Ark file."""
#         super().__init__(file_path, permissive=False, **kwargs)
#         self._open_archive(self.file_path)
#         self._eof = False
#         self._keys = None
#         if self.num_parts > 1:
#             raise NotImplementedError(
#                 "Dataset splitting not available for %s" % self.__class__.__name__
#             )

#     def reset(self) -> None:
#         """Puts the file pointer back to the beginning of the file"""
#         if self.f is not None:
#             self.f.seek(0, 0)
#             self._eof = False

#     def eof(self) -> bool:
#         """Returns True when it reaches the end of the ark file."""
#         return self._eof or self.f is None

#     @property
#     def keys(self) -> List[str]:
#         """List recording keys available in the Ark file."""
#         if self._keys is None:
#             self.reset()
#             self._keys, _ = self.read_shapes()
#             self.reset()

#         return self._keys

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
#         keys = []
#         shapes = []
#         count = 0
#         binary = False
#         while num_records == 0 or count < num_records:

#             key_i = read_token(self.f, binary)
#             if key_i == "":
#                 self._eof = True
#                 break

#             binary = init_kaldi_input_stream(self.f)
#             shape_i = KaldiMatrix.read_shape(self.f, binary, sequential_mode=True)

#             keys.append(key_i)
#             shapes.append(shape_i)
#             count += 1

#         if assert_same_dim and len(shapes) > 0:
#             dims = np.array([s[-1] for s in shapes], dtype=int)
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
#         row_offset_is_list = isinstance(row_offset, (list, np.ndarray))
#         num_rows_is_list = isinstance(num_rows, (list, np.ndarray))
#         if isinstance(row_offset, np.ndarray) and row_offset.ndim == 0:
#             row_offset_is_list = False
#         if isinstance(num_rows, np.ndarray) and num_rows.ndim == 0:
#             num_rows_is_list = False
#         keys = []
#         data = []
#         count = 0
#         binary = False
#         with self.lock:
#             while num_records == 0 or count < num_records:

#                 key_i = read_token(self.f, binary)
#                 if key_i == "":
#                     self._eof = True
#                     break

#                 if row_offset_is_list and count >= len(row_offset):
#                     raise ValueError(
#                         f"row_offset has {len(row_offset)} items but at least {count + 1} are required"
#                     )
#                 if num_rows_is_list and count >= len(num_rows):
#                     raise ValueError(
#                         f"num_rows has {len(num_rows)} items but at least {count + 1} are required"
#                     )

#                 row_offset_i = row_offset[count] if row_offset_is_list else row_offset
#                 num_rows_i = num_rows[count] if num_rows_is_list else num_rows

#                 binary = init_kaldi_input_stream(self.f)
#                 data_i = KaldiMatrix.read(
#                     self.f, binary, row_offset_i, num_rows_i, sequential_mode=True
#                 ).to_ndarray()

#                 assert num_rows_i == 0 or data_i.shape[0] == num_rows_i

#                 if self.transform is not None:
#                     data_i = self.transform.predict(data_i)

#                 keys.append(key_i)
#                 data.append(data_i)
#                 count += 1

#         if squeeze:
#             data = self._squeeze(data)

#         return keys, data


# class SequentialArkScriptDataReader(SequentialArkDataReader):
#     """Class to read Ark feature files indexed by a scp file in
#     sequential order.

#      Attributes:
#         file_path: scp file to read.
#         path_prefix: If input_spec is a scp file, it pre-appends
#                      path_prefix string to the second column of
#                      the scp file. This is useful when data
#                      is read from a different directory of that
#                      it was created.
#         scp_sep: Separator for scp files (default ' ').
#         transform: TransformList object, applies a transformation to the
#                    features after reading them from disk.
#         part_idx: It splits the input into num_parts and writes only
#                   part part_idx, where part_idx=1,...,num_parts.
#         num_parts: Number of parts to split the input data.
#     """

#     def __init__(
#         self, file_path: PathLike, path_prefix: Optional[PathLike] = None, **kwargs
#     ) -> None:
#         """Initialize a sequential reader over Ark files indexed by scp."""
#         super().__init__(file_path, permissive=False, **kwargs)
#         self.feature_set = FeatureSet.load(self.file_path)

#         if self.num_parts > 1:
#             self.feature_set = self.feature_set.split(self.part_idx, self.num_parts)

#         if path_prefix is not None:
#             self.feature_set.add_prefix_to_storage_path(path_prefix)

#         self.cur_item = 0

#     @property
#     def keys(self) -> np.ndarray:
#         """Array of recording keys in read order."""
#         return self.feature_set["id"].values

#     def reset(self) -> None:
#         """Closes all the open Ark files and puts the read pointer pointing
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
#             offset = feature_spec["storage_byte"]
#             file_path = feature_spec["storage_path"]
#             self._open_archive(file_path, offset)
#             binary = init_kaldi_input_stream(self.f)
#             shape_i = KaldiMatrix.read_shape(self.f, binary, sequential_mode=True)
#             if "start" in feature_spec and "num_frames" in feature_spec:
#                 range_spec = [feature_spec["start"], feature_spec["num_frames"]]
#                 row_offset_i, num_rows_i = self._combine_ranges(range_spec, 0, 0)
#                 shape_i = self._apply_range_to_shape(shape_i, row_offset_i, num_rows_i)

#             keys.append(key)
#             shapes.append(shape_i)
#             self.cur_item += 1

#         if assert_same_dim and len(shapes) > 0:
#             dims = np.array([s[-1] for s in shapes], dtype=int)
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
#                 offset = feature_spec["storage_byte"]
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

#                 self._open_archive(file_path, offset)
#                 binary = init_kaldi_input_stream(self.f)
#                 data_i = KaldiMatrix.read(
#                     self.f, binary, row_offset_i, num_rows_i, sequential_mode=True
#                 ).to_ndarray()

#                 assert num_rows_i == 0 or data_i.shape[0] == num_rows_i

#                 if self.transform is not None:
#                     data_i = self.transform.predict(data_i)

#                 keys.append(key)
#                 data.append(data_i)
#                 self.cur_item += 1

#         if squeeze:
#             data = self._squeeze(data)

#         return keys, data


# class RandomAccessArkDataReader(RandomAccessDataReader):
#     """Class to read Ark files in random order, using scp file to
#     index the Ark files.

#      Attributes:
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
#         self,
#         file_path: PathLike,
#         path_prefix: Optional[PathLike] = None,
#         transform: TransformArg = None,
#         permissive: bool = False,
#     ) -> None:
#         """Initialize a random-access Ark reader backed by an scp index."""
#         super().__init__(file_path, transform, permissive)

#         self.feature_set = FeatureSet.load(self.file_path)
#         if path_prefix is not None:
#             self.feature_set.add_prefix_to_storage_path(path_prefix)

#         archives, archive_idx = np.unique(
#             self.feature_set["storage_path"], return_inverse=True
#         )
#         self.archives = archives
#         self.archive_idx = archive_idx
#         self.f = [None] * len(self.archives)
#         self.locks = [threading.Lock() for i in range(len(self.archives))]

#     @property
#     def keys(self) -> np.ndarray:
#         """Array of keys that can be queried."""
#         return self.feature_set["id"].values

#     def close(self) -> None:
#         """Closes all the open Ark files."""
#         for f in self.f:
#             if f is not None:
#                 f.close()
#         self.f = [None] * len(self.f)

#     def _open_archive(
#         self, key_idx: int, offset: int = 0
#     ) -> Tuple[object, ProcessLock]:
#         """Opens the Ark file correspoding to a given feature/matrix
#            if it is not already open and moves the file pointer to the
#            point where we can read that feature matrix.

#            If the file was already open, it only moves the file pointer.

#         Args:
#           key_idx: Integer position of the feature matrix in the scp file.
#           offset: Byte where we can find the feature matrix in the Ark file.

#         Returns:
#           Python file object.
#           multiprocessing lock object corresponding to the file.
#         """
#         archive_idx = self.archive_idx[key_idx]
#         with self.locks[archive_idx]:
#             if self.f[archive_idx] is None:
#                 self.f[archive_idx] = open(self.archives[archive_idx], "rb")

#             f = self.f[archive_idx]
#             f.seek(offset, 0)

#         return f, self.locks[archive_idx]

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
#         dims = np.array([s[-1] for s in shapes], dtype=int)
#         if assert_same_dim and len(dims) > 0:
#             if self.permissive:
#                 dims_to_assert = np.array(
#                     [s[-1] for s in shapes if not (len(s) == 1 and s[0] == 0)], dtype=int
#                 )
#             else:
#                 dims_to_assert = dims
#             if len(dims_to_assert) > 0:
#                 assert np.all(dims_to_assert == dims_to_assert[0])
#         return dims

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

#             if not (key in self.feature_set.index):
#                 if self.permissive:
#                     shapes.append((0,))
#                     continue
#                 else:
#                     raise Exception("Key %s not found" % key)

#             index = self.feature_set.get_loc(key)
#             feature_spec = self.feature_set.loc[key]
#             offset = feature_spec["storage_byte"]
#             f, lock = self._open_archive(index)
#             with lock:
#                 f.seek(offset, 0)
#                 binary = init_kaldi_input_stream(f)
#                 shape_i = KaldiMatrix.read_shape(f, binary, sequential_mode=False)

#             if "start" in feature_spec and "num_frames" in feature_spec:
#                 range_spec = [feature_spec["start"], feature_spec["num_frames"]]
#                 row_offset_i, num_rows_i = self._combine_ranges(range_spec, 0, 0)
#                 shape_i = self._apply_range_to_shape(shape_i, row_offset_i, num_rows_i)

#             shapes.append(shape_i)

#         if assert_same_dim and len(shapes) > 0:
#             if self.permissive:
#                 dims = np.array(
#                     [s[-1] for s in shapes if not (len(s) == 1 and s[0] == 0)], dtype=int
#                 )
#             else:
#                 dims = np.array([s[-1] for s in shapes], dtype=int)
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
#             offset = feature_spec["storage_byte"]
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
#                 f.seek(offset, 0)
#                 binary = init_kaldi_input_stream(f)
#                 data_i = KaldiMatrix.read(
#                     f, binary, row_offset_i, num_rows_i, sequential_mode=False
#                 ).to_ndarray()

#             assert num_rows_i == 0 or data_i.shape[0] == num_rows_i

#             if self.transform is not None:
#                 data_i = self.transform.predict(data_i)

#             data.append(data_i)

#         if squeeze:
#             data = self._squeeze(data, self.permissive)

#         return data
