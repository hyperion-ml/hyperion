"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging

from .rw_specifiers import ArchiveType, WSpecifier, RSpecifier, WSpecType, RSpecType
from .bin_vad_reader import BinVADReader as BVR
from .segment_vad_reader import SegmentVADReader as SVR


class VADReaderFactory(object):
    @staticmethod
    def create(
        rspecifier,
        path_prefix=None,
        scp_sep=" ",
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
                    scp_sep,
                    frame_length=frame_length,
                    frame_shift=frame_shift,
                    snip_edges=snip_edges,
                )
            if rspecifier.archive_type == ArchiveType.SEGMENT_LIST:
                return SVR(rspecifier.archive, permissive=rspecifier.permissive)
        else:
            if (
                rspecifier.archive_type == ArchiveType.H5
                or rspecifier.archive_type == ArchiveType.ARK
            ):
                return BVR(
                    rspecifier,
                    path_prefix,
                    scp_sep,
                    frame_length=frame_length,
                    frame_shift=frame_shift,
                    snip_edges=snip_edges,
                )

    @staticmethod
    def filter_args(**kwargs):
        valid_args = (
            "scp_sep",
            "path_prefix",
            "frame_shift",
            "frame_length",
            "snip_edges",
        )
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
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
            p1 + "frame-shift",
            default=10,
            help=("frame-shift used to compute binary VAD"),
        )
        parser.add_argument(
            p1 + "frame-length",
            default=25,
            help=("frame-length used to compute binary VAD"),
        )
        parser.add_argument(
            p1 + "snip-edges",
            default=False,
            action="store_true",
            help=("snip-edges was true when computing VAD"),
        )

    add_argparse_args = add_class_args
