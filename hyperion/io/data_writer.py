"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from abc import ABCMeta, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from ..utils import PathLike

if TYPE_CHECKING:
    from ..utils.kaldi_matrix import KaldiCompressedMatrix, KaldiMatrix

WriteKeys = Union[str, List[str], np.ndarray]
WriteDataItem = Union[np.ndarray, "KaldiMatrix", "KaldiCompressedMatrix"]
WriteData = Union[np.ndarray, List[WriteDataItem]]
MetadataValue = Union[object, List[object], np.ndarray]
MetadataDict = Dict[str, MetadataValue]
MetadataArg = Optional[Union[pd.DataFrame, MetadataDict]]
StandardizedMetadata = Optional[List[List[object]]]


class DataWriter(metaclass=ABCMeta):
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
      metadata_columns: Optional metadata columns to export to non-scp script files.
    """

    def __init__(
        self,
        archive_path: PathLike,
        script_path: Optional[PathLike] = None,
        flush: bool = False,
        compress: bool = False,
        compression_method: str = "auto",
        metadata_columns: Optional[List[str]] = None,
    ) -> None:
        """Initialize writer configuration and output files."""
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

    def __enter__(self) -> "DataWriter":
        """Function required when entering constructions of type

        with DataWriter('file.h5') as f:
           f.write(key, data)
        """
        return self

    @abstractmethod
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Function required when exiting from constructions of type

        with DataWriter('file.h5') as f:
           f.write(key, data)
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the output files."""
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush buffered output data."""
        pass

    def standardize_write_args(
        self,
        keys: WriteKeys,
        data: WriteData,
        metadata: MetadataArg = None,
    ) -> Tuple[List[str], List[WriteDataItem], StandardizedMetadata]:
        """Normalize write arguments to list form and validate list lengths."""
        if isinstance(keys, np.ndarray):
            keys = [keys.item()] if keys.ndim == 0 else keys.tolist()
        elif isinstance(keys, str):
            keys = [keys]
        else:
            keys = list(keys)

        num_items = len(keys)

        if isinstance(data, np.ndarray):
            if num_items == 1:
                data = [data]
            else:
                if data.ndim == 0:
                    raise ValueError(
                        f"data is a scalar array but {num_items} keys were provided"
                    )
                if data.shape[0] != num_items:
                    raise ValueError(
                        f"data has {data.shape[0]} items but {num_items} keys were provided"
                    )
                if data.ndim < 2:
                    raise ValueError(
                        "for multiple keys, ndarray data must have at least 2 dimensions"
                    )
                data = [d for d in data]
        else:
            data = list(data)
            if len(data) != num_items:
                raise ValueError(
                    f"data has {len(data)} items but {num_items} keys were provided"
                )

        metadata_out = None
        if metadata is not None:
            if isinstance(metadata, pd.DataFrame):
                metadata = metadata.to_dict(orient="list")
            else:
                metadata = dict(metadata)

            if self.metadata_columns is None:
                raise ValueError(
                    "metadata_columns must be provided when metadata is passed"
                )

            metadata_out = []
            for c in self.metadata_columns:
                if c not in metadata:
                    raise KeyError(f"metadata column '{c}' is missing")

                m_c = metadata[c]
                if isinstance(m_c, np.ndarray):
                    values = [m_c.item()] if m_c.ndim == 0 else m_c.tolist()
                elif isinstance(m_c, list):
                    values = m_c
                else:
                    values = [m_c]

                if len(values) != num_items:
                    raise ValueError(
                        f"metadata column '{c}' has {len(values)} items but {num_items} keys were provided"
                    )

                metadata_out.append(values)

        return keys, data, metadata_out

    @staticmethod
    def _escape_script_field(value: object, sep: Optional[str]) -> str:
        """Escape a value written to a CSV/TSV-like script row."""
        text = str(value)
        if sep is None:
            return text

        needs_quotes = sep in text or '"' in text or "\n" in text or "\r" in text
        if '"' in text:
            text = text.replace('"', '""')
        if needs_quotes:
            text = f'"{text}"'
        return text

    @abstractmethod
    def write(
        self,
        keys: WriteKeys,
        data: WriteData,
        metadata: MetadataArg = None,
    ) -> None:
        """Writes data to file.

        Args:
          keys: List of recording names.
          data: List of Feature matrices or vectors.
                If all the matrices have the same dimension
                it can be a 3D numpy array.
                If they are vectors, it can be a 2D numpy array.
          metadata: Dictionary/DataFrame with metadata values.
        """
        pass
