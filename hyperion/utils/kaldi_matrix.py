"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

 Classes to to write and read kaldi matrices
"""

import struct
import numpy as np

from ..hyp_defs import float_cpu
from ..utils.kaldi_io_funcs import *


class KaldiMatrix(object):
    """Class to read/write uncompressed kaldi matrices/vectors.

    When compressed matrix is found in file, it calls
    KaldiCompressedMatrix class automatically to uncompress.

    Attributes:
      data: numpy array with the matrix/vector values.

    """

    def __init__(self, data):
        self.data = data

    def to_ndarray(self):
        """
        Returns:
          numpy array containing the matrix/vector
        """
        return self.data

    @property
    def num_rows(self):
        if self.data.ndim == 1:
            return 1
        return self.data.shape[0]

    @property
    def num_cols(self):
        if self.data.ndim == 1:
            return self.data.shape[0]
        return self.data.shape[1]

    @classmethod
    def read(cls, f, binary, row_offset=0, num_rows=0, sequential_mode=True):
        """Reads kaldi matrix/vector from file.

        Args:
          f: Python file object
          binary: True if we read from binary file and False if we read from text file.
          row_offset: Reads matrix starting from a given row instead of row 0.
          num_rows: Num. of rows to read, if 0 if read all the rows.
          sequential_mode: True if we are reading the ark file sequentially and False if
            we are using random access.

        Returns:
          KaldiMatrix object.
        """
        if binary:
            peekval = peek(f, binary)
            if peekval == b"C":
                mat = KaldiCompressedMatrix.read(
                    f, binary, row_offset, num_rows, sequential_mode
                ).to_ndarray()
                return cls(mat)
            token = read_token(f, binary)
            if token[0] == "F":
                dtype = "float32"
            elif token[0] == "D":
                dtype = "float64"
            else:
                ValueError("Wrong token %s " % token)
            if token[1] == "V":
                ndim = 1
            elif token[1] == "M":
                ndim = 2
            else:
                ValueError("Wrong token %s " % token)

            rows_left = 0
            if ndim == 2:
                total_rows = read_int32(f, binary)
                assert row_offset <= total_rows, "row_offset (%d) > num_rows (%d)" % (
                    row_offset,
                    total_rows,
                )
                total_rows -= row_offset
                if num_rows == 0:
                    num_rows = total_rows
                else:
                    assert (
                        num_rows <= total_rows
                    ), "requested rows (%d) > available rows (%d)" % (
                        num_rows,
                        total_rows,
                    )
                    rows_left = total_rows - num_rows

            else:
                num_rows = 1
            num_cols = read_int32(f, binary)

            if row_offset > 0:
                f.seek(row_offset * num_cols * np.dtype(dtype).itemsize, 1)
            data = f.read(num_rows * num_cols * np.dtype(dtype).itemsize)
            if rows_left > 0 and sequential_mode:
                f.seek(rows_left * num_cols * np.dtype(dtype).itemsize, 1)

            vec = np.frombuffer(data, dtype=dtype)

            if ndim == 2:
                return cls(np.reshape(vec, (num_rows, num_cols)))
            return cls(vec)

        else:
            if row_offset > 0 or num_rows > 0:
                raise NotImplementedError(
                    "Reading slices supported in text mode because it is inefficient"
                )

            first_line = True
            rows = []
            is_vector = False
            for line in f:
                if isinstance(line, bytes):
                    line = line.decode("ascii")

                if len(line) == 0:
                    raise BadInputFormat(
                        "EOF reading matrix"
                    )  # eof, should not happen!
                if len(line.strip()) == 0:
                    continue  # skip empty line

                arr = line.strip().split()
                if first_line:
                    if arr == "[]":
                        return np.array([], dtype="float32")
                    if arr[0] != "[":
                        raise ValueError("Wrong matrix format %s " % line)
                    first_line = False
                    if len(arr) > 1:
                        is_vector = True
                        arr = arr[1:]
                    else:
                        continue

                if arr[-1] != "]":
                    rows.append(np.array(arr, dtype="float32"))  # not last line
                else:
                    rows.append(np.array(arr[:-1], dtype="float32"))  # last line
                    mat = np.vstack(rows)
                    if mat.shape[0] == 1 and is_vector:
                        mat = mat.ravel()
                    return cls(mat)

            return cls(np.array([], dtype="float32"))

    def write(self, f, binary):
        """Writes matrix/vector to ark file.

        Args:
          f: Python file object.
          binary: True if we write in binary file and False if we write to text file.
        """
        if binary:
            t1 = "F" if self.data.dtype == np.float32 else "D"
            t2 = "M" if self.data.ndim == 2 else "V"
            token = t1 + t2
            write_token(f, binary, token)
            if self.data.ndim == 2:
                write_int32(f, binary, self.num_rows)
            write_int32(f, binary, self.num_cols)
            f.write(self.data.tobytes())
        else:
            if self.num_cols == 0:
                f.write(" [ ]\n")
            else:
                f.write(" [")
                if self.data.ndim == 1:
                    f.write(" ")
                    for j in range(self.num_cols):
                        f.write("%f " % self.data[j])
                else:
                    for i in range(self.num_rows):
                        f.write("\n ")
                        for j in range(self.num_cols):
                            f.write("%f " % self.data[i, j])
                f.write("]\n")

    @staticmethod
    def read_shape(f, binary, sequential_mode=True):
        """Reads the shape of the current matrix/vector in the ark file.

        Args:
          f: Python file object
          binary: True if we read from binary file and False if we read from text file.
          sequential_mode: True if we are reading the ark file sequentially and False if
            we are using random access. In sequential_mode=True it moves the file pointer
            to the next matrix.

        Returns:
          Tuple object with shape.
        """
        if binary:
            peekval = peek(f, binary)
            if peekval == b"C":
                return KaldiCompressedMatrix.read_shape(f, binary, sequential_mode)
            token = read_token(f, binary)
            if token[0] == "F":
                dtype = "float32"
            elif token[0] == "D":
                dtype = "float64"
            else:
                ValueError("Wrong token %s " % token)
            if token[1] == "V":
                ndim = 1
            elif token[1] == "M":
                ndim = 2
            else:
                ValueError("Wrong token %s " % token)

            if ndim == 2:
                num_rows = read_int32(f, binary)
            else:
                num_rows = 1
            num_cols = read_int32(f, binary)
            if sequential_mode:
                f.seek(num_rows * num_cols * np.dtype(dtype).itemsize, 1)

            if ndim == 1:
                return (num_cols,)
            else:
                return (num_rows, num_cols)
        else:
            matrix = KaldiMatrix.read(f, binary, sequential_mode=sequential_mode)
            return matrix.data.shape


compression_methods = [
    "auto",
    "speech-feat",
    "2byte-auto",
    "2byte-signed-integer",
    "1byte-auto",
    "1byte-unsigned-integer",
    "1byte-0-1",
    "speech-feat-t",
]

compression_method2format = {
    "speech-feat": 1,
    "2byte-auto": 2,
    "2byte-signed-integer": 2,
    "1byte-auto": 3,
    "1byte-unsigned-integer": 3,
    "1byte-0-1": 3,
    "speech-feat-t": 4,
}


class KaldiCompressedMatrix(object):
    """Class to read/write compressed kaldi matrices.

    When compressed matrix is found in file, it calls
    KaldiCompressedMatrix class automatically to uncompress.

    Attributes:
      data: numpy byte array with the compressed coded matrix.
      data_format: {1, 2, 3, 4}
      min_value: Minimum value in the matrix.
      data_range: max_value - min_value
      num_rows: Number of rows in the matrix
      num_columns: Number of columns in the matrix
    """

    def __init__(self, data=None):
        self.data = data
        self.data_format = 1
        self.min_value = 0
        self.data_range = 0
        self.num_rows = 0
        self.num_cols = 0

        if data is not None:
            self._unpack_header()
        # self.col_headers = col_headers

    def get_data_attrs(self):
        """
        Returns:
           Coded matrix values in 2D format.
           Dictionary object with data attributes: data_format, min_value, data_range, percentiles.
        """
        attrs = {
            "data_format": self.data_format,
            "min_value": self.min_value,
            "data_range": self.data_range,
        }

        header_offset = 20
        if self.data_format == 1 or self.data_format == 4:
            data_offset = header_offset + self.num_cols * 8
            p = np.frombuffer(self.data[header_offset:data_offset], dtype=np.uint16)
            attrs["perc"] = p
            data = (
                np.reshape(
                    np.frombuffer(self.data[data_offset:], dtype=np.uint8),
                    (self.num_cols, self.num_rows),
                )
                .transpose()
                .copy()
            )
        elif self.data_format == 2:
            data = np.reshape(
                np.frombuffer(self.data[header_offset:], dtype=np.uint16),
                (self.num_rows, self.num_cols),
            )
        else:
            data = np.reshape(
                np.frombuffer(self.data[header_offset:], dtype=np.uint8),
                (self.num_rows, self.num_cols),
            )

        return data, attrs

    @classmethod
    def build_from_data_attrs(cls, data, attrs):
        """Builds object from coded values and attributes

        Args:
          data: Coded matrix values in 2D format.
          attrs: Dictionary object with data attributes: data_format, min_value, data_range, percentiles.

        Returns:
          KaldiCompressedMatrix object.
        """
        num_rows = data.shape[0]
        num_cols = data.shape[1]
        h = struct.pack(
            "<iffii",
            attrs["data_format"],
            attrs["min_value"],
            attrs["data_range"],
            num_rows,
            num_cols,
        )
        if attrs["data_format"] == 1 or attrs["data_format"] == 4:
            col_header = attrs["perc"].tobytes()
            h = h + col_header
            data_bytes = data.transpose().flatten().tobytes()
        else:
            data_bytes = data.ravel().tobytes()

        return cls(h + data_bytes)

    def _unpack_header(self):
        """Unpacks attributes from header"""
        h = struct.unpack("<iffii", self.data[:20])
        self.data_format = h[0]
        self.min_value = h[1]
        self.data_range = h[2]
        self.num_rows = h[3]
        self.num_cols = h[4]

    def _pack_header(self):
        """Creates header from the object attributes"""
        return struct.pack(
            "<iffii",
            self.data_format,
            self.min_value,
            self.data_range,
            self.num_rows,
            self.num_cols,
        )

    def scale(self, alpha):
        """Multiplies matrix by alpha"""
        self.min_value *= alpha
        self.data_range *= alpha
        header = self._pack_header()
        self.data = header + self.data[20:]

    def _compute_global_header(self, mat, method):
        """Computes the header

        Args:
          mat: numpy array with the uncompressed matrix.
          method: Compression method.

        Returns:
          Byte array with header.
        """
        if method == "auto":
            if mat.shape[0] > 8:
                method = "speech-feat"
            else:
                method = "2byte-auto"
        self.data_format = compression_method2format[method]
        self.num_rows = mat.shape[0]
        self.num_cols = mat.shape[1]

        # now compute min_val and range
        if (
            method == "speech-feat"
            or method == "2byte-auto"
            or method == "1byte-auto"
            or method == "speech-feat-t"
        ):
            min_value = np.min(mat)
            max_value = np.max(mat)
            if max_value == min_value:
                max_value = min_value + 1 + np.abs(min_value)
            assert (
                min_value - min_value == 0 and max_value - max_value == 0
            ), "cannot compress matrix with infs or nans"
            self.min_value = min_value
            self.data_range = max_value - min_value
            assert self.data_range > 0
        elif method == "2byte-signed-integer":
            self.min_value = -32768.0
            self.data_range = 65535.0
        elif method == "1byte-unsigned-integer":
            self.min_value = 0.0
            self.data_range = 255.0
        elif method == "1byte-0-1":
            self.min_value = 0.0
            self.data_range = 1.0
        else:
            raise ValueError(method)

        header = self._pack_header()
        return header

    @staticmethod
    def _get_read_info(header, row_offset=0, num_rows=0):
        """Gets info needed to read the matrix from file"""
        data_format, min_value, data_range, total_rows, num_cols = struct.unpack(
            "<iffii", header
        )
        make_header = True if row_offset != 0 or num_rows != 0 else False

        rows_left = 0
        assert row_offset <= total_rows, "row_offset (%d) > num_rows (%d)" % (
            row_offset,
            total_rows,
        )
        total_rows -= row_offset
        if num_rows == 0:
            num_rows = total_rows
        else:
            assert (
                num_rows <= total_rows
            ), "requested rows (%d) > available rows (%d)" % (num_rows, total_rows)
            rows_left = total_rows - num_rows

        bytes_col_header = 0
        if data_format == 1 or data_format == 4:
            bytes_col_header = num_cols * 8
            bytes_offset = row_offset
            bytes_data = num_rows
            bytes_left = rows_left
        elif data_format == 2:
            bytes_offset = 2 * row_offset
            bytes_data = 2 * num_rows
            bytes_left = 2 * rows_left
        else:
            bytes_offset = row_offset
            bytes_data = num_rows
            bytes_left = rows_left

        if make_header:
            header = struct.pack(
                "<iffii", data_format, min_value, data_range, num_rows, num_cols
            )

        return header, num_cols, bytes_col_header, bytes_offset, bytes_data, bytes_left

    @staticmethod
    def _data_size(header):
        """
        Returns:
          Number of bytes of the coded matrix.
        """
        data_format, _, _, num_rows, num_cols = struct.unpack("<iffii", header)
        if data_format == 1 or data_format == 4:
            return len(header) + num_cols * (8 + num_rows)
        elif data_format == 2:
            return len(header) + 2 * num_rows * num_cols
        else:
            return len(header) + num_rows * num_cols

    @classmethod
    def compress(cls, mat, method="auto"):
        """Creates compressed matrix from uncompressed numpy matrix
        Args:
          mat: numpy array with the uncompressed matrix.
          method: Compression method.

        Returns:
          KaldiCompressedMatrix object.
        """
        if isinstance(mat, KaldiMatrix):
            mat = mat.data

        M = cls()
        header = M._compute_global_header(mat, method)
        cols_header = bytes()
        data = bytes()
        if M.data_format == 1 or M.data_format == 4:
            for col in range(M.num_cols):
                col_header, col_data = M._compress_column(mat[:, col])
                cols_header += col_header
                data += col_data

        elif M.data_format == 2:
            data = M._float_to_uint16(mat).tobytes()
        else:
            data = M._float_to_uint8(mat).tobytes()

        M.data = header + cols_header + data
        return M

    def _float_to_uint16(self, mat):
        f = (mat.ravel() - self.min_value) / self.data_range
        f[f > 1.0] = 1
        f[f < 0.0] = 0
        return (f * 65535 + 0.499).astype(np.uint16)

    def _float_to_uint8(self, mat):
        f = (mat.ravel() - self.min_value) / self.data_range
        f[f > 1.0] = 1
        f[f < 0.0] = 0
        return (f * 255 + 0.499).astype(np.uint8)

    def _uint16_to_float(self, byte_data):
        return self.min_value + self.data_range * 1.52590218966964e-05 * np.frombuffer(
            byte_data, dtype=np.uint16
        ).astype(float_cpu())

    def _uint8_to_float(self, byte_data):
        return self.min_value + self.data_range / 255.0 * np.frombuffer(
            byte_data, dtype=np.uint8
        ).astype(float_cpu())

    def _compute_column_header(self, v):
        """Creates the column headers for the speech-feat compression.

        Args:
          v: numpy array with the column to compress.

        Returns:
          Byte array with the header of the column containg the 0, 25, 75 and 100 percentile values.
        """
        one = np.uint16(1)
        if self.num_rows >= 5:
            quarter_nr = int(self.num_rows / 4)
            v_sort = np.partition(v, (0, quarter_nr, 3 * quarter_nr, -1))
            perc_0 = min(self._float_to_uint16(v_sort[0])[0], np.uint16(65532))
            perc_25 = min(
                max(self._float_to_uint16(v_sort[quarter_nr])[0], perc_0 + one),
                np.uint16(65533),
            )
            perc_75 = min(
                max(self._float_to_uint16(v_sort[3 * quarter_nr])[0], perc_25 + one),
                np.uint16(65534),
            )
            perc_100 = max(self._float_to_uint16(v_sort[-1])[0], perc_75 + one)
        else:
            v_sort = np.sort(v)
            perc_0 = min(self._float_to_uint16(v_sort[0])[0], np.uint16(65532))
            if self.num_rows > 1:
                perc_25 = min(
                    max(self._float_to_uint16(v_sort[1])[0], perc_0 + one),
                    np.uint16(65533),
                )
            else:
                perc_25 = perc_0 + one
            if self.num_rows > 2:
                perc_75 = min(
                    max(self._float_to_uint16(v_sort[2])[0], perc_25 + one),
                    np.uint16(65534),
                )
            else:
                perc_75 = perc_25 + one

            if self.num_rows > 3:
                perc_100 = max(self._float_to_uint16(v_sort[3])[0], perc_75 + one)
            else:
                perc_100 = perc_75 + one
        return struct.pack("<HHHH", perc_0, perc_25, perc_75, perc_100)

    def _compress_column(self, v):
        """Compress column for the speech-feat compression.

        Args:
          v: numpy array with the column to compress.

        Returns:
          Byte array with the header of the column containg the 0, 25, 75 and 100 percentile values.
          Byte array with the coded column.
        """

        col_header = self._compute_column_header(v)
        p0, p25, p75, p100 = self._uint16_to_float(col_header)
        return col_header, self._float_to_char(v, p0, p25, p75, p100).tobytes()

    def _uncompress_column(self, col_header, col_data):
        """Compress column for the speech-feat compression.

        Args:
          col_header: Byte array with the header of the column containg the 0, 25, 75 and 100 percentile values.
          col_data: Byte array with the coded column.

        Returns:
          numpy array with the uncompressed column
        """
        p0, p25, p75, p100 = self._uint16_to_float(col_header)
        return self._char_to_float(col_data, p0, p25, p75, p100)

    @staticmethod
    def _float_to_char(v, p0, p25, p75, p100):
        """Codes the column from float to bytes using the given percentiles"""
        v_out = np.zeros(v.shape, dtype=np.int32)
        idx = v < p25
        f = (v[idx] - p0) / (p25 - p0)
        c = (f * 64 + 0.5).astype(np.int32)
        c[c < 0] = 0
        c[c > 64] = 64
        v_out[idx] = c
        idx = np.logical_and(v >= p25, v < p75)
        f = (v[idx] - p25) / (p75 - p25)
        c = 64 + (f * 128 + 0.5).astype(np.int32)
        c[c < 64] = 64
        c[c > 192] = 192
        v_out[idx] = c
        idx = v >= p75
        f = (v[idx] - p75) / (p100 - p75)
        c = 192 + (f * 63 + 0.5).astype(np.int32)
        c[c < 192] = 192
        c[c > 255] = 255
        v_out[idx] = c
        return v_out.astype(np.uint8)

    @staticmethod
    def _char_to_float(v, p0, p25, p75, p100):
        """Decodes the column from bytes to float using the given percentiles"""
        v_in = np.frombuffer(v, dtype=np.uint8).astype(float_cpu())
        v_out = np.zeros(v_in.shape, dtype=float_cpu())
        idx = v_in <= 64
        v_out[idx] = p0 + (p25 - p0) * v_in[idx] / 64.0
        idx = np.logical_and(v_in > 64, v_in <= 192)
        v_out[idx] = p25 + (p75 - p25) * (v_in[idx] - 64) / 128.0
        idx = v_in > 192
        v_out[idx] = p75 + (p100 - p75) * (v_in[idx] - 192) / 63.0
        return v_out

    def to_ndarray(self):
        """Uncompresses matrix to numpy array.
        Returns:
          numpy array with uncompressed matrix.
        """
        if self.data_format == 1 or self.data_format == 4:
            mat = np.zeros((self.num_rows, self.num_cols), dtype=float_cpu())
            header_offset = 20
            data_offset = header_offset + self.num_cols * 8
            for i in range(self.num_cols):
                mat[:, i] = self._uncompress_column(
                    self.data[header_offset : header_offset + 8],
                    self.data[data_offset : data_offset + self.num_rows],
                )
                header_offset += 8
                data_offset += self.num_rows
        elif self.data_format == 2:
            mat = np.reshape(
                self._uint16_to_float(self.data[20:]), (self.num_rows, self.num_cols)
            ).astype(float_cpu(), copy=False)
        else:
            mat = np.reshape(
                self._uint8_to_float(self.data[20:]), (self.num_rows, self.num_cols)
            ).astype(float_cpu(), copy=False)

        return mat

    def to_matrix(self):
        """Uncompresses matrix to KaldiMatrix object.
        Returns:
          KaldiMatrix with uncompressed matrix.
        """

        mat = self.to_ndarray()
        return KaldiMatrix(mat)

    @classmethod
    def read(cls, f, binary, row_offset=0, num_rows=0, sequential_mode=True):
        """Reads kaldi compressed matrix/vector from file.

        Args:
          f: Python file object
          binary: True if we read from binary file and False if we read from text file.
          row_offset: Reads matrix starting from a given row instead of row 0.
          num_rows: Num. of rows to read, if 0 if read all the rows.
          sequential_mode: True if we are reading the ark file sequentially and False if
            we are using random access.

        Returns:
          KaldiCompressedMatrix object.
        """

        if binary:
            peekval = peek(f, binary)
            if peekval == b"C":
                token = read_token(f, binary)
                if token == "CM":
                    data_format = 1
                elif token == "CM2":
                    data_format = 2
                elif token == "CM3":
                    data_format = 3
                elif token == "CM4":
                    data_format = 4
                else:
                    raise ValueError("Unexpected token %s" % token)

                header = struct.pack("<i", data_format) + f.read(16)
                (
                    header,
                    num_cols,
                    bytes_col_header,
                    bytes_offset,
                    bytes_col,
                    bytes_left,
                ) = cls._get_read_info(header, row_offset, num_rows)

                if bytes_offset == 0 and bytes_left == 0:
                    if data_format == 4:
                        col_header = f.read(bytes_col_header)
                        data = f.read(num_cols * bytes_col)
                        data = (
                            np.frombuffer(data, dtype=np.uint8)
                            .reshape(-1, num_cols)
                            .transpose()
                            .tobytes()
                        )
                        data = header + col_header + data
                    else:
                        data = header + f.read(bytes_col_header + num_cols * bytes_col)
                else:
                    if data_format == 1:
                        col_header = f.read(bytes_col_header)
                        data = bytes()
                        for c in range(num_cols):
                            if bytes_offset > 0:
                                f.seek(bytes_offset, 1)
                            data += f.read(bytes_col)
                            if bytes_left > 0:
                                f.seek(bytes_left, 1)
                        data = header + col_header + data
                    elif data_format == 4:
                        col_header = f.read(bytes_col_header)
                        if bytes_offset > 0:
                            f.seek(bytes_offset * num_cols, 1)
                        data = f.read(bytes_col * num_cols)
                        data = (
                            np.frombuffer(data, dtype=np.uint8)
                            .reshape(-1, num_cols)
                            .transpose()
                            .tobytes()
                        )
                        if bytes_left > 0:
                            f.seek(bytes_left * num_cols, 1)
                        data = header + col_header + data
                    else:
                        if bytes_offset > 0:
                            f.seek(bytes_offset * num_cols, 1)
                        data = f.read(bytes_col * num_cols)
                        if bytes_left > 0:
                            f.seek(bytes_left * num_cols, 1)
                        data = header + data

                return cls(data)
            else:
                matrix = KaldiMatrix.read(f, binary, row_offset, num_rows)
                return cls.compress(matrix)

        if row_offset > 0 or num_rows > 0:
            raise NotImplementedError(
                "Reading slices supported in text mode because it is inefficient"
            )
        matrix = KaldiMatrix.read(f, binary)
        return cls.compress(matrix)

    def write(self, f, binary):
        """Writes matrix/vector to ark file.

        Args:
          f: Python file object.
          binary: True if we write in binary file and False if we write to text file.
        """

        if binary:
            if self.data is not None:
                if self.data_format == 1:
                    write_token(f, binary, "CM")
                elif self.data_format == 2:
                    write_token(f, binary, "CM2")
                elif self.data_format == 3:
                    write_token(f, binary, "CM3")
                elif self.data_format == 4:
                    write_token(f, binary, "CM4")

                if self.data_format == 4:
                    header_offset = 20
                    data_offset = header_offset + self.num_cols * 8
                    data = (
                        np.frombuffer(self.data[data_offset:], dtype=np.uint8)
                        .reshape(self.num_cols, self.num_rows)
                        .transpose()
                        .tobytes()
                    )
                    f.write(self.data[4:data_offset])
                    f.write(data)
                else:
                    f.write(self.data[4:])
            else:
                write_token(f, binary, "CM")
                header = struct.pack("<ffii", 0, 0, 0, 0)
                f.write(header)
        else:
            self.to_matrix().write(f, binary)

    @staticmethod
    def read_shape(f, binary, sequential_mode=True):
        """Reads the shape of the current matrix/vector in the ark file.

        Args:
          f: Python file object
          binary: True if we read from binary file and False if we read from text file.
          sequential_mode: True if we are reading the ark file sequentially and False if
            we are using random access. In sequential_mode=True it moves the file pointer
            to the next matrix.

        Returns:
          Tuple object with shape.
        """

        if binary:
            peekval = peek(f, binary)
            if peekval == b"C":
                token = read_token(f, binary)
                if token == "CM":
                    data_format = 1
                elif token == "CM2":
                    data_format = 2
                elif token == "CM3":
                    data_format = 3
                elif token == "CM4":
                    data_format = 4
                else:
                    raise ValueError("Unexpected token %s" % token)

                header = struct.pack("<i", data_format) + f.read(16)
                num_rows, num_cols = struct.unpack("<iffii", header)[-2:]
                if sequential_mode:
                    size = KaldiCompressedMatrix._data_size(header) - len(header)
                    f.seek(size, 1)

                return (num_rows, num_cols)
            else:
                matrix = KaldiMatrix.read(f, binary, row_offset, num_rows)
                return matrix.data.shape

        matrix = KaldiMatrix.read(f, binary)
        return matrix.data.shape
