"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging

from ..utils.kaldi_matrix import compression_methods
from .data_rw_factory import DataWriterFactory as DWF
from .data_rw_factory import SequentialDataReaderFactory as DRF


class CopyFeats(object):
    """Class to convet between Ark/hdf5 feature formats."""

    def __init__(
        self,
        input_spec,
        output_spec,
        path_prefix=None,
        compress=False,
        compression_method="auto",
        write_num_frames=None,
        scp_sep=" ",
        part_idx=1,
        num_parts=1,
        chunk_size=1,
    ):
        """CopyFeats constructor, it executes the conversion.

        Args:
          input_spec: Kaldi style read specifier, e.g.:
                      file.h5
                      h5:file.h5
                      ark:file.ark
                      scp:file.scp

                      or list of specifiers, e.g.:
                      ['scp:file1.scp', 'scp:file2.scp']

                      if it is a list, it merges the input files.

           output_spec: Kaldi stype write specifier, e.g.:
                        file.h5
                        h5:file.h5
                        ark:file.ark
                        h5,scp:file.h5,file.scp
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
           scp_sep: Separator for scp files (default ' ').
           part_idx: It splits the input into num_parts and writes only
                     part part_idx, where part_idx=1,...,num_parts.
           num_parts: Number of parts to split the input data.
           chunk_size: When copying, it reads the input files in groups of
                       chunk_size (default:1).
        """
        if isinstance(input_spec, str):
            input_spec = [input_spec]

        assert not (
            num_parts > 1 and len(input_spec) > 1
        ), "Merging and splitting at the same time is not supported"

        if write_num_frames is not None:
            f_nf = open(write_num_frames, "w")

        logging.info("opening output stream: %s" % (output_spec))
        with DWF.create(
            output_spec,
            compress=compress,
            compression_method=compression_method,
            scp_sep=scp_sep,
        ) as writer:

            for rspec in input_spec:
                logging.info("opening input stream: %s" % (rspec))
                with DRF.create(
                    rspec,
                    path_prefix=path_prefix,
                    scp_sep=scp_sep,
                    part_idx=part_idx,
                    num_parts=num_parts,
                ) as reader:
                    while not reader.eof():
                        key, data = reader.read(chunk_size)
                        if len(key) == 0:
                            break
                        logging.info("copying %d feat matrices" % (len(key)))
                        writer.write(key, data)
                        if write_num_frames is not None:
                            for k, v in zip(key, data):
                                f_nf.write("%s %d\n" % (k, v.shape[0]))

        if write_num_frames is not None:
            f_nf.close()

    @staticmethod
    def filter_args(**kwargs):
        """Extracts the relevant arguments for the CopyFeats object.

        Args:
          kwargs: Dictionary containing arguments for several classes.

        Returns:
          Dictionary with the relevant arguments to initialize the object.
        """
        valid_args = ("scp_sep", "path_prefix", "part_idx", "num_parts")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        """Adds arguments required to initialize the object to python
           argparse object.

        Args:
          parser: Python argparse object.
          prefix: Prefix for the argument names. The prefix is useful when you have
                  several objects of the same class in the program and you want to
                  initialize each of them with different arguments.
        """
        if prefix is None:
            p1 = "--"
        else:
            p1 = "--" + prefix + "."

        parser.add_argument(
            p1 + "scp-sep", default=" ", help=("scp file field separator")
        )
        parser.add_argument(
            p1 + "path-prefix", default=None, help=("scp file_path prefix")
        )
        parser.add_argument(
            p1 + "part-idx",
            type=int,
            default=1,
            help=("splits the list of files in num-parts and process part_idx"),
        )
        parser.add_argument(
            p1 + "num-parts",
            type=int,
            default=1,
            help=("splits the list of files in num-parts and process part_idx"),
        )

        parser.add_argument(p1 + "compress", default=False, action="store_true")
        parser.add_argument(
            p1 + "compression-method", default="auto", choices=compression_methods
        )

    add_argparse_args = add_class_args
