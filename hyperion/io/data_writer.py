"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import os
from abc import ABCMeta, abstractmethod


class DataWriter(object):
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
      scp_sep: Separator for scp files (default ' ').
    """

    __metaclass__ = ABCMeta

    def __init__(
        self,
        archive_path,
        script_path=None,
        flush=False,
        compress=False,
        compression_method="auto",
        scp_sep=" ",
    ):
        self.archive_path = archive_path
        self.script_path = script_path
        self._flush = flush
        self.compress = compress
        self.compression_method = compression_method
        self.scp_sep = scp_sep

        archive_dir = os.path.dirname(archive_path)
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)

        if script_path is not None:
            script_dir = os.path.dirname(script_path)
            if not os.path.exists(script_dir):
                os.makedirs(script_dir)

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

    @abstractmethod
    def write(self, key, data):
        """Writes data to file.

        Args:
          key: List of recodings names.
          data: List of Feature matrices or vectors.
                If all the matrices have the same dimension
                it can be a 3D numpy array.
                If they are vectors, it can be a 2D numpy array.
        """
        pass
