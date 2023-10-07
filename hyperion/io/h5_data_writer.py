"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Union, Optional, List, Dict

import h5py
import numpy as np
import pandas as pd

from ..hyp_defs import float_save
from ..utils.kaldi_io_funcs import is_token
from ..utils.kaldi_matrix import KaldiCompressedMatrix, KaldiMatrix
from ..utils import PathLike
from .data_writer import DataWriter


class H5DataWriter(DataWriter):
    """Class to write hdf5 feature files.

    Attributes:
      archive_path: output data file path.
      script_path: optional output scp file.
      flush: If True, it flushes the output after writing each feature file.
      compress: It True, it uses Kaldi compression.
      compression_method: Kaldi compression method:
                          {auto (default), speech_feat,
                           2byte-auto, 2byte-signed-integer,
                           1byte-auto, 1byte-unsigned-integer, 1byte-0-1}.
    """

    def __init__(
        self, archive_path: PathLike, script_path: Optional[PathLike] = None, **kwargs
    ):

        super().__init__(archive_path, script_path, **kwargs)

        self.f = h5py.File(archive_path, "w")
        if script_path is not None and not self.script_is_scp:
            columns = ["id", "storage_path"]
            if self.metadata_columns is not None:
                columns += self.metadata_columns
            row = self.script_sep.join(columns)
            self.f_script.write(f"{row}\n")

    def __exit__(self, exc_type, exc_value, traceback):
        """Function required when exiting from contructions of type

           with H5DataWriter('file.h5') as f:
              f.write(key, data)

        It closes the output file.
        """
        self.close()

    def close(self):
        """Closes the output file"""
        if self.f is not None:
            self.f.close()
            self.f = None
        if self.f_script is not None:
            self.f_script.close()

    def flush(self):
        """Flushes the file"""
        self.f.flush()
        if self.f_script is not None:
            self.f_script.flush()

    def _convert_data(self, data: np.array):
        """Converts data to the format for saving.
        Compresses the data it needed.
        Args:
          Numpy array feature matrix/vector.

        Returns:
          Numpy array to save in h5 file.
          Atrributes for the hdf5 dataset with information about the
          compression.
        """
        if isinstance(data, np.ndarray):
            if self.compress:
                mat = KaldiCompressedMatrix.compress(data, self.compression_method)
                return mat.get_data_attrs()
            else:
                data = data.astype(float_save(), copy=False)
                return data, None
        else:
            raise ValueError("Data is not ndarray")

    def write(
        self,
        keys: Union[str, List[str], np.array],
        data: Union[np.array, List[np.array]],
        metadata: Optional[Union[pd.DataFrame, Dict]] = None,
    ):
        """Writes data to file.

        Args:
          key: List of recodings names.
          data: List of Feature matrices or vectors.
                If all the matrices have the same dimension
                it can be a 3D numpy array.
                If they are vectors, it can be a 2D numpy array.
        """
        keys, data, metadata = self.standardize_write_args(keys, data, metadata)

        for i, key_i in enumerate(keys):
            assert is_token(key_i), "Token %s not valid" % key_i
            data_i, attrs = self._convert_data(data[i])
            dset = self.f.create_dataset(key_i, data=data_i)
            if attrs is not None:
                for k, v in attrs.items():
                    dset.attrs[k] = v

            if self.f_script is not None:
                if self.script_is_scp:
                    self.f_script.write(f"{key_i} {self.archive_path}\n")
                else:
                    columns = [key_i, str(self.archive_path)]
                    if metadata is not None:
                        metadata_i = [str(m[i]) for m in metadata]
                        columns += metadata_i
                    row = self.script_sep.join(columns)
                    self.f_script.write(f"{row}\n")

            if self._flush:
                self.flush()
