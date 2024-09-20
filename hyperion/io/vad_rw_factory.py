"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging

from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from .bin_vad_reader import BinVADReader as BVR
from .rw_specifiers import ArchiveType, RSpecifier, RSpecType, WSpecifier, WSpecType

# from .segment_vad_reader import SegmentVADReader as SVR
from .table_vad_reader import TableVADReader as TVR


class VADReaderFactory:
    @staticmethod
    def create(
        rspecifier,
        path_prefix=None,
        frame_length=25,
        frame_shift=10,
        snip_edges=False,
    ):

        if isinstance(rspecifier, str):
            rspecifier = RSpecifier.create(rspecifier)
        logging.debug(rspecifier.__dict__)
        if rspecifier.spec_type == RSpecType.ARCHIVE:
            if (
                rspecifier.archive_type == ArchiveType.H5
                or rspecifier.archive_type == ArchiveType.ARK
            ):
                return BVR(
                    rspecifier,
                    path_prefix,
                    frame_length=frame_length,
                    frame_shift=frame_shift,
                    snip_edges=snip_edges,
                )

        else:
            if (
                rspecifier.archive_type == ArchiveType.H5
                or rspecifier.archive_type == ArchiveType.ARK
            ):
                return BVR(
                    rspecifier,
                    path_prefix,
                    frame_length=frame_length,
                    frame_shift=frame_shift,
                    snip_edges=snip_edges,
                )
            if rspecifier.archive_type == ArchiveType.TABLE:
                return TVR(rspecifier.archive, path_prefix=path_prefix)

    @staticmethod
    def filter_args(**kwargs):
        valid_args = (
            "path_prefix",
            "frame_shift",
            "frame_length",
            "snip_edges",
        )
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--path-prefix", default=None, help=("scp file_path prefix")
        )
        parser.add_argument(
            "--frame-shift",
            default=10.0,
            type=float,
            help=("frame-shift used to compute binary VAD"),
        )
        parser.add_argument(
            "--frame-length",
            default=25.0,
            type=float,
            help=("frame-length used to compute binary VAD"),
        )
        parser.add_argument(
            "--snip-edges",
            default=False,
            action=ActionYesNo,
            help=("snip-edges was true when computing VAD"),
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    add_argparse_args = add_class_args
