"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from types import TracebackType
from typing import Optional, Type, Union

import numpy as np

from ..hyp_defs import float_save
from ..utils import PathLike
from ..utils.kaldi_io_funcs import init_kaldi_output_stream, is_token, write_token
from ..utils.kaldi_matrix import KaldiCompressedMatrix, KaldiMatrix
from .data_writer import DataWriter, MetadataArg, WriteData, WriteKeys

ArkWriteData = Union[WriteData, KaldiMatrix, KaldiCompressedMatrix]


class ArkDataWriter(DataWriter):
    """Class to write Ark feature files.

    Attributes:
      archive_path: output data file path.
      script_path: optional output scp file.
      binary: True if the the Ark file is binary, False if it is text file.
      flush: If True, it flushes the output after writing each feature file.
      compress: It True, it uses Kaldi compression.
      compression_method: Kaldi compression method:
                          {auto (default), speech_feat,
                           2byte-auto, 2byte-signed-integer,
                           1byte-auto, 1byte-unsigned-integer, 1byte-0-1}.

    """

    def __init__(
        self,
        archive_path: PathLike,
        script_path: Optional[PathLike] = None,
        binary: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(archive_path, script_path, **kwargs)
        self.binary = binary

        if binary:
            self.f = open(archive_path, "wb")
        else:
            self.f = open(archive_path, "w")

        if script_path is not None and not self.script_is_scp:
            columns = ["id", "storage_path", "storage_byte"]
            if self.metadata_columns is not None:
                columns += self.metadata_columns
            row = self.script_sep.join(columns)
            self.f_script.write(f"{row}\n")

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Function required when exiting from constructions of type

           with ArkDataWriter('file.h5') as f:
              f.write(key, data)

        It closes the output file.
        """
        self.close()

    def close(self) -> None:
        """Close output files."""
        self.f.close()
        if self.f_script is not None:
            self.f_script.close()

    def flush(self) -> None:
        """Flush buffered output data."""
        self.f.flush()
        if self.f_script is not None:
            self.f_script.flush()

    def _convert_data(
        self, data: Union[np.ndarray, KaldiMatrix, KaldiCompressedMatrix]
    ) -> Union[KaldiMatrix, KaldiCompressedMatrix]:
        """Converts the feature matrix from numpy array to KaldiMatrix
        or KaldiCompressedMatrix.

        Compression is only applied to 2D arrays/matrices.
        """
        if isinstance(data, np.ndarray):
            if data.ndim not in (1, 2):
                raise ValueError(
                    f"ArkDataWriter expects 1D or 2D arrays, got ndim={data.ndim}"
                )
            data = data.astype(float_save(), copy=False)
            if self.compress and data.ndim == 2:
                return KaldiCompressedMatrix.compress(data, self.compression_method)
            return KaldiMatrix(data)

        if isinstance(data, KaldiMatrix):
            if data.data.ndim not in (1, 2):
                raise ValueError(
                    "ArkDataWriter expects KaldiMatrix with 1D or 2D data, "
                    f"got ndim={data.data.ndim}"
                )
            if self.compress and data.data.ndim == 2:
                return KaldiCompressedMatrix.compress(data, self.compression_method)
            return data

        if isinstance(data, KaldiCompressedMatrix):
            if not self.compress:
                return data.to_matrix()
            return data

        raise ValueError("Data is not ndarray or KaldiMatrix")

    def write(
        self,
        keys: WriteKeys,
        data: ArkWriteData,
        metadata: MetadataArg = None,
    ) -> None:
        """Writes data to file.

        Args:
          keys: List of recording names.
          data: List of Feature matrices or vectors.
                If all the matrices have the same dimension
                it can be a 3D numpy array.
                If they are vectors, it can be a 2D numpy array.
                It also accepts KaldiMatrix/KaldiCompressedMatrix objects.
          metadata: Dictionary/DataFrame with metadata values.
        """
        if isinstance(data, (KaldiMatrix, KaldiCompressedMatrix)):
            data = [data]
        keys, data, metadata = self.standardize_write_args(keys, data, metadata)

        for i, key_i in enumerate(keys):
            if not is_token(key_i):
                raise ValueError(f"Token {key_i} not valid")
            write_token(self.f, self.binary, key_i)

            pos = self.f.tell()
            data_i = self._convert_data(data[i])

            init_kaldi_output_stream(self.f, self.binary)
            data_i.write(self.f, self.binary)

            if self.f_script is not None:
                if self.script_is_scp:
                    self.f_script.write(f"{key_i} {self.archive_path}:{pos}\n")
                else:
                    columns = [
                        self._escape_script_field(key_i, self.script_sep),
                        self._escape_script_field(self.archive_path, self.script_sep),
                        str(pos),
                    ]
                    if self.metadata_columns is not None:
                        if metadata is not None:
                            metadata_i = [
                                self._escape_script_field(m[i], self.script_sep)
                                for m in metadata
                            ]
                        else:
                            metadata_i = [""] * len(self.metadata_columns)
                        columns += metadata_i
                    row = self.script_sep.join(columns)
                    self.f_script.write(f"{row}\n")

            if self._flush:
                self.flush()
