"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Union

from jsonargparse import ActionParser, ArgumentParser

from ..utils import PathLike
from ..utils.kaldi_matrix import compression_methods
from .data_rw_factory import DataWriterFactory as DWF
from .data_rw_factory import SequentialDataReaderFactory as DRF

InputSpec = Union[PathLike, List[PathLike]]


class CopyFeats:
    """Copy feature matrices between Ark/HDF5 sources and destinations.

    The copy is executed during object construction.

    Attributes:
      input_spec: Input read specifier or list of read specifiers.
      output_spec: Output write specifier.
      path_prefix: Optional prefix prepended to paths read from script inputs.
      compress: Whether to enable feature compression on write.
      compression_method: Kaldi compression method used when ``compress=True``.
      write_num_frames: Optional output file for ``<key> <num_frames>`` lines.
      part_idx: Selected split index when processing partitioned input.
      num_parts: Number of partitions used to split input processing.
      chunk_size: Number of records to read/write per iteration.

    Usage example:
      >>> CopyFeats(
      ...     input_spec="scp:data/feats.scp",
      ...     output_spec="h5,csv:exp/feats.h5,exp/feats.csv",
      ...     compress=True,
      ...     compression_method="speech_feat",
      ...     chunk_size=100,
      ... )
      >>> CopyFeats(
      ...     input_spec=["scp:data/part1.scp", "scp:data/part2.scp"],
      ...     output_spec="ark,scp:exp/feats.ark,exp/feats.scp",
      ... )
    """

    def __init__(
        self,
        input_spec: InputSpec,
        output_spec: PathLike,
        path_prefix: Optional[PathLike] = None,
        compress: bool = False,
        compression_method: str = "auto",
        write_num_frames: Optional[PathLike] = None,
        part_idx: int = 1,
        num_parts: int = 1,
        chunk_size: int = 1,
    ) -> None:
        """Execute feature-copy operation.

        Args:
          input_spec: Kaldi style read specifier, e.g.:
                      file.h5
                      h5:file.h5
                      ark:file.ark
                      scp:file.scp

                      or list of specifiers, e.g.:
                      ['scp:file1.scp', 'scp:file2.scp']

                      if it is a list, it merges the input files.

           output_spec: Kaldi style write specifier, e.g.:
                        file.h5
                        h5:file.h5
                        ark:file.ark
                        h5,csv:file.h5,file.csv
                        h5,scp:file.h5,file.scp
                        ark,csv:file.ark,file.csv
                        ark,scp:file.ark,file.scp

           path_prefix: If input_spec is a scp file, it pre-appends
                        path_prefix string to the second column of
                        the scp file. This is useful when data
                        is read from a different directory of that
                        it was created.
           compress: if True, it  compress the features (default: False).
           compression_method: Kaldi compression method:
                               {auto (default), speech_feat,
                                2byte-auto, 2byte-signed-integer,
                                1byte-auto, 1byte-unsigned-integer, 1byte-0-1}.
           write_num_frames: Optional output file where ``<key> <num_frames>``
                             is written for each copied feature matrix.
           part_idx: It splits the input into num_parts and writes only
                     part part_idx, where part_idx=1,...,num_parts.
           num_parts: Number of parts to split the input data.
           chunk_size: When copying, it reads the input files in groups of
                       chunk_size (default:1).
        """
        if not isinstance(input_spec, list):
            input_spec = [input_spec]
        input_spec = [str(spec) for spec in input_spec]

        if num_parts > 1 and len(input_spec) > 1:
            raise ValueError("Merging and splitting at the same time is not supported")

        num_frames_ctx = (
            open(str(write_num_frames), "w", encoding="utf-8")
            if write_num_frames is not None
            else nullcontext()
        )
        with num_frames_ctx as f_nf:
            logging.info("opening output stream: %s" % (output_spec))
            with DWF.create(
                output_spec,
                compress=compress,
                compression_method=compression_method,
            ) as writer:

                for rspec in input_spec:
                    logging.info("opening input stream: %s" % (rspec))
                    with DRF.create(
                        rspec,
                        path_prefix=path_prefix,
                        part_idx=part_idx,
                        num_parts=num_parts,
                    ) as reader:
                        while not reader.eof():
                            key, data = reader.read(chunk_size)
                            if len(key) == 0:
                                break
                            logging.info("copying %d feat matrices" % (len(key)))
                            writer.write(key, data)
                            if f_nf is not None:
                                for k, v in zip(key, data):
                                    f_nf.write("%s %d\n" % (k, v.shape[0]))

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Extract relevant keyword arguments for :class:`CopyFeats`.

        Args:
          kwargs: Dictionary containing arguments for several classes.

        Returns:
          Dictionary with the relevant arguments to initialize the object.
        """
        valid_args = (
            "path_prefix",
            "part_idx",
            "num_parts",
            "compress",
            "compression_method",
            "write_num_frames",
            "chunk_size",
        )
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser: Any, prefix: Optional[str] = None) -> None:
        """Add ``CopyFeats`` arguments to an argparse-like parser.

        Args:
          parser: Python argparse object.
          prefix: Prefix for the argument names. The prefix is useful when you have
                  several objects of the same class in the program and you want to
                  initialize each of them with different arguments.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--path-prefix", default=None, help=("scp file_path prefix")
        )
        parser.add_argument(
            "--part-idx",
            type=int,
            default=1,
            help=("splits the list of files in num-parts and process part_idx"),
        )
        parser.add_argument(
            "--num-parts",
            type=int,
            default=1,
            help=("splits the list of files in num-parts and process part_idx"),
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=1,
            help=("number of feature matrices to read/write per iteration"),
        )
        parser.add_argument(
            "--write-num-frames",
            default=None,
            help=("optional output file to write number of frames per utterance"),
        )

        parser.add_argument("--compress", default=False, action="store_true")
        parser.add_argument(
            "--compression-method", default="auto", choices=compression_methods
        )
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    add_argparse_args = add_class_args
