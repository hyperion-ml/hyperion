"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import List, Optional, Union

from ..utils import PathLike
from .h5_data_reader import RandomAccessH5FileDataReader
from .h5_data_writer import H5DataWriter

InputFiles = Union[PathLike, List[PathLike]]


class H5Merger:
    """Merge multiple HDF5 feature files into a single HDF5 file.

    This class uses the current HDF5 IO stack (`RandomAccessH5FileDataReader`
    and `H5DataWriter`) and avoids deprecated HypData* readers/writers.

    Attributes:
      input_files: Ordered list of input HDF5 files to merge.
      output_file: Output HDF5 file path.
      chunk_size: Number of datasets copied per batch. If ``None``, each input
        file is copied in a single batch.

    Examples:
      >>> from hyperion.io.h5_merger import H5Merger
      >>> merger = H5Merger(
      ...     input_files=["exp/part1.h5", "exp/part2.h5"],
      ...     output_file="exp/merged.h5",
      ...     chunk_size=500,
      ... )
      >>> merger.merge()
    """

    def __init__(
        self,
        input_files: InputFiles,
        output_file: PathLike,
        chunk_size: Optional[int] = None,
    ) -> None:
        """Initialize merger configuration.

        Args:
          input_files: One input HDF5 file path or a list of input file paths.
          output_file: Destination HDF5 file path.
          chunk_size: Number of dataset keys to copy per batch. If ``None``,
            all keys in each input file are copied in a single batch.

        Raises:
          ValueError: If ``input_files`` is empty or ``chunk_size`` is not
            positive when provided.
        """
        if not isinstance(input_files, list):
            input_files = [input_files]
        self.input_files = [str(f) for f in input_files]
        if len(self.input_files) == 0:
            raise ValueError("input_files cannot be empty")

        self.output_file = output_file
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be > 0 when provided")
        self.chunk_size = chunk_size

    def merge(self) -> None:
        """Merge all configured input files into the output HDF5 file."""
        with H5DataWriter(self.output_file) as writer:
            for h5_file in self.input_files:
                self._merge_file(writer, h5_file)

    def _merge_file(self, writer: H5DataWriter, input_file: PathLike) -> None:
        """Merge one input HDF5 file into the output writer.

        Args:
          writer: Open output HDF5 writer.
          input_file: Input HDF5 file path.
        """
        with RandomAccessH5FileDataReader(input_file) as reader:
            datasets = reader.keys
            chunk = len(datasets) if self.chunk_size is None else self.chunk_size
            if chunk == 0:
                return

            for first in range(0, len(datasets), chunk):
                last = min(first + chunk, len(datasets))
                keys = datasets[first:last]
                data = reader.read(keys)
                writer.write(keys, data)
