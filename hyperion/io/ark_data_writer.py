"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import sys
import numpy as np

from ..hyp_defs import float_save
from ..utils.scp_list import SCPList
from ..utils.kaldi_io_funcs import is_token, write_token, init_kaldi_output_stream
from ..utils.kaldi_matrix import KaldiMatrix, KaldiCompressedMatrix
from .data_writer import DataWriter


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
      scp_sep: Separator for scp files (default ' ').
    """

    def __init__(self, archive_path, script_path=None, binary=True, **kwargs):
        super(ArkDataWriter, self).__init__(archive_path, script_path, **kwargs)
        self.binary = binary

        if binary:
            self.f = open(archive_path, "wb")
        else:
            self.f = open(archive_path, "w")

        if script_path is not None:
            self.f_script = open(script_path, "w")
        else:
            self.f_script = None

    def __exit__(self, exc_type, exc_value, traceback):
        """Function required when exiting from contructions of type

           with ArkDataWriter('file.h5') as f:
              f.write(key, data)

        It closes the output file.
        """
        self.close()

    def close(self):
        """Closes the output file"""
        self.f.close()
        if self.f_script is not None:
            self.f_script.close()

    def flush(self):
        """Flushes the file"""
        self.f.flush()
        if self.f_script is not None:
            self.f_script.flush()

    def _convert_data(self, data):
        """Converts the feature matrix from numpy array to KaldiMatrix
        or KaldiCompressedMatrix.
        """
        if isinstance(data, np.ndarray):
            data = data.astype(float_save(), copy=False)
            if self.compress:
                return KaldiCompressedMatrix.compress(data, self.compression_method)
            return KaldiMatrix(data)

        if isinstance(data, KaldiMatrix):
            if self.compress:
                return KaldiCompressedMatrix.compress(data, self.compression_method)
            return data

        if isinstance(data, KaldiCompressedMatrix):
            if not self.compress:
                return data.to_matrix()
            return data

        raise ValueError("Data is not ndarray or KaldiMatrix")

    def write(self, keys, data):
        """Writes data to file.

        Args:
          key: List of recodings names.
          data: List of Feature matrices or vectors.
                If all the matrices have the same dimension
                it can be a 3D numpy array.
                If they are vectors, it can be a 2D numpy array.
        """
        if isinstance(keys, str):
            keys = [keys]
            data = [data]

        for i, key_i in enumerate(keys):
            assert is_token(key_i), "Token %s not valid" % key_i
            write_token(self.f, self.binary, key_i)

            pos = self.f.tell()
            data_i = self._convert_data(data[i])

            init_kaldi_output_stream(self.f, self.binary)
            data_i.write(self.f, self.binary)

            if self.f_script is not None:
                self.f_script.write(
                    "%s%s%s:%d\n" % (key_i, self.scp_sep, self.archive_path, pos)
                )

            if self._flush:
                self.flush()
