"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from jsonargparse import ArgumentParser, ActionParser

from ..utils.kaldi_matrix import compression_methods
from .rw_specifiers import ArchiveType, WSpecifier, RSpecifier, WSpecType, RSpecType
from .h5_data_writer import H5DataWriter as H5DW
from .ark_data_writer import ArkDataWriter as ADW
from .ark_data_reader import SequentialArkFileDataReader as SAFDR
from .ark_data_reader import SequentialArkScriptDataReader as SASDR
from .ark_data_reader import RandomAccessArkDataReader as RADR
from .h5_data_reader import SequentialH5FileDataReader as SH5FDR
from .h5_data_reader import SequentialH5ScriptDataReader as SH5SDR
from .h5_data_reader import RandomAccessH5FileDataReader as RH5FDR
from .h5_data_reader import RandomAccessH5ScriptDataReader as RH5SDR


class DataWriterFactory(object):
    """
    Class to create object that write data to hdf5/ark files.
    """

    @staticmethod
    def create(wspecifier, compress=False, compression_method="auto", scp_sep=" "):
        if isinstance(wspecifier, str):
            wspecifier = WSpecifier.create(wspecifier)

        if (
            wspecifier.spec_type == WSpecType.ARCHIVE
            or wspecifier.spec_type == WSpecType.BOTH
        ):

            if wspecifier.archive_type == ArchiveType.H5:
                return H5DW(
                    wspecifier.archive,
                    wspecifier.script,
                    flush=wspecifier.flush,
                    compress=compress,
                    compression_method=compression_method,
                    scp_sep=scp_sep,
                )
            else:
                return ADW(
                    wspecifier.archive,
                    wspecifier.script,
                    binary=wspecifier.binary,
                    flush=wspecifier.flush,
                    compress=compress,
                    compression_method=compression_method,
                    scp_sep=scp_sep,
                )

    @staticmethod
    def filter_args(**kwargs):
        valid_args = ("scp_sep", "compress", "compression_method")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument("--scp-sep", default=" ", help=("scp file field separator"))
        parser.add_argument("--compress", default=False, action="store_true")
        parser.add_argument(
            "--compression-method", default="auto", choices=compression_methods
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='data writer options')


class SequentialDataReaderFactory(object):
    @staticmethod
    def create(rspecifier, path_prefix=None, scp_sep=" ", **kwargs):

        if isinstance(rspecifier, str):
            rspecifier = RSpecifier.create(rspecifier)

        if rspecifier.spec_type == RSpecType.ARCHIVE:
            if rspecifier.archive_type == ArchiveType.H5:
                return SH5FDR(rspecifier.archive, **kwargs)
            else:
                return SAFDR(rspecifier.archive, **kwargs)
        else:
            if rspecifier.archive_type == ArchiveType.H5:
                return SH5SDR(rspecifier.script, path_prefix, scp_sep=scp_sep, **kwargs)
            else:
                return SASDR(rspecifier.script, path_prefix, scp_sep=scp_sep, **kwargs)

    @staticmethod
    def filter_args(**kwargs):
        valid_args = ("scp_sep", "path_prefix", "part_idx", "num_parts")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        try:
            parser.add_argument(
                "--scp-sep", default=" ", help=("scp file field separator")
            )
        except:
            pass
        parser.add_argument(
            "--path-prefix", default=None, help=("scp file_path prefix")
        )
        try:
            parser.add_argument(
                "--part-idx",
                type=int,
                default=1,
                help=("splits the list of files in num-parts " "and process part_idx"),
            )
            parser.add_argument(
                "--num-parts",
                type=int,
                default=1,
                help=("splits the list of files in num-parts " "and process part_idx"),
            )
        except:
            pass

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='data reader options')


class RandomAccessDataReaderFactory(object):
    @staticmethod
    def create(rspecifier, path_prefix=None, transform=None, scp_sep=" "):
        if isinstance(rspecifier, str):
            rspecifier = RSpecifier.create(rspecifier)
        logging.debug(rspecifier.__dict__)
        if rspecifier.spec_type == RSpecType.ARCHIVE:
            if rspecifier.archive_type == ArchiveType.H5:
                return RH5FDR(
                    rspecifier.archive,
                    transform=transform,
                    permissive=rspecifier.permissive,
                )
            else:
                raise ValueError(
                    "Random access to Ark file %s needs a script file"
                    % rspecifier.archive
                )
        else:
            if rspecifier.archive_type == ArchiveType.H5:
                return RH5SDR(
                    rspecifier.archive,
                    path_prefix,
                    transform=transform,
                    permissive=rspecifier.permissive,
                    scp_sep=scp_sep,
                )
            else:
                return RADR(
                    rspecifier.script,
                    path_prefix,
                    transform=transform,
                    permissive=rspecifier.permissive,
                    scp_sep=scp_sep,
                )

    @staticmethod
    def filter_args(**kwargs):
        valid_args = ("scp_sep", "path_prefix")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        try:
            parser.add_argument(
                "--scp-sep", default=" ", help=("scp file field separator")
            )
        except:
            pass
        parser.add_argument(
            "--path-prefix", default=None, help=("scp file_path prefix")
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='data reader options')

    add_argparse_args = add_class_args
