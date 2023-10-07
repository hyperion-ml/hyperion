"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import os
from abc import ABCMeta, abstractmethod
from typing import Union, Optional, List, Dict
from pathlib import Path
import numpy as np
import pandas as pd
from ..utils import PathLike


class DataWriter:
    """Abstract base class to write Ark or hdf5 feature files.

    Attributes:
      archive_path: output data file path.
      script_path: optional output scp file.
      flush: If True, it flushes the output after writing each feature matrix.
      compress: It True, it uses Kaldi compression.
      compression_method: Kaldi compression method:
                          {auto (default), speech_feat,
                           2byte-auto, 2byte-signed-integer,
                           1byte-auto, 1byte-unsigned-integer, 1byte-0-1}.
    """

    __metaclass__ = ABCMeta

    def __init__(
        self,
        archive_path: PathLike,
        script_path: Optional[PathLike] = None,
        flush: bool = False,
        compress: bool = False,
        compression_method: str = "auto",
        metadata_columns: Optional[List[str]] = None,
    ):
        self.archive_path = Path(archive_path)
        self.script_path = Path(script_path) if script_path is not None else None
        self._flush = flush
        self.compress = compress
        self.compression_method = compression_method
        self.metadata_columns = metadata_columns

        archive_dir = self.archive_path.parent
        archive_dir.mkdir(exist_ok=True, parents=True)

        self.script_is_scp = False
        self.script_sep = None
        self.f_script = None
        if script_path is not None:
            self.script_path.parent.mkdir(exist_ok=True, parents=True)
            script_ext = self.script_path.suffix
            self.script_is_scp = script_ext == ".scp"

            if self.script_is_scp:
                self.f_script = open(self.script_path, "w")
            else:
                self.script_sep = "," if script_ext == ".csv" else "\t"
                self.f_script = open(self.script_path, "w", encoding="utf-8")

    def __enter__(self):
        """Function required when entering contructions of type

        with DataWriter('file.h5') as f:
           f.write(key, data)
        """
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        """Function required when exiting from contructions of type

        with DataWriter('file.h5') as f:
           f.write(key, data)
        """
        pass

    @abstractmethod
    def close(self):
        """Closes the output file"""
        pass

    @abstractmethod
    def flush(self):
        """Flushes the file"""
        pass

    def standardize_write_args(
        self,
        keys: Union[str, List[str], np.array],
        data: Union[np.array, List[np.array]],
        metadata: Optional[Union[pd.DataFrame, Dict]] = None,
    ):
        if isinstance(keys, str):
            keys = [keys]
            data = [data]

        if metadata is not None:
            if isinstance(metadata, pd.DataFrame):
                metadata = metadata.to_dict()

            metadata_list = []
            for c in self.metadata_columns:
                m_c = metadata[c]
                if not isinstance(m_c, (list, np.ndarray)):
                    m_c = [m_c]
                metadata_list.append(m_c)

            metadata = metadata_list

        return keys, data, metadata

    @abstractmethod
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
          metadata: dictionary/DataFrame with metadata
        """
        pass
