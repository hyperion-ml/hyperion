"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Dict, Optional, Union

from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ..utils import PathLike
from .bin_vad_reader import BinVADReader as BVR
from .rw_specifiers import ArchiveType, RSpecifier, RSpecType

# from .segment_vad_reader import SegmentVADReader as SVR
from .table_vad_reader import TableVADReader as TVR

VADReaderType = Union[BVR, TVR]
ReadSpecifierArg = Union[PathLike, RSpecifier]


class VADReaderFactory:
    """Factory that builds VAD readers from Kaldi-style read specifiers.

    Examples:
      Create a table-based VAD reader from a CSV index:
      >>> r = VADReaderFactory.create("csv:data/vad.csv")
      >>> marks = r.read_time_marks(["utt1", "utt2"])
      >>> r.close()

      Create a table-based VAD reader:
      >>> r = VADReaderFactory.create("csv:data/vad_index.csv")
      >>> marks = r.read_time_marks(["utt1"])
      >>> r.close()

      Parse only factory-relevant arguments from a larger config:
      >>> vad_kwargs = VADReaderFactory.filter_args(
      ...     path_prefix="/mnt/vad", frame_shift=10.0, foo=1
      ... )
      >>> r = VADReaderFactory.create(
      ...     "csv:data/vad.csv", path_prefix=vad_kwargs["path_prefix"]
      ... )
      >>> r.close()
    """

    @staticmethod
    def create(
        rspecifier: ReadSpecifierArg,
        path_prefix: Optional[PathLike] = None,
        frame_length: float = 25,
        frame_shift: float = 10,
        snip_edges: bool = False,
    ) -> VADReaderType:
        """Create a VAD reader.

        Args:
          rspecifier: Read specifier string/path or pre-parsed ``RSpecifier``.
          path_prefix: Optional prefix added to script paths.
          frame_length: Frame length in milliseconds used for binary VAD conversion.
          frame_shift: Frame shift in milliseconds used for binary VAD conversion.
          snip_edges: Snip-edges setting used for binary VAD conversion.

        Returns:
          A ``BinVADReader`` for H5/ARK inputs, or a ``TableVADReader`` for table
          inputs.

        Raises:
          ValueError: If specifier type or archive type is unsupported.
        """

        if not isinstance(rspecifier, RSpecifier):
            rspecifier = RSpecifier.create(str(rspecifier))
        logging.debug(rspecifier.__dict__)

        if rspecifier.spec_type == RSpecType.ARCHIVE:
            if rspecifier.archive_type in (ArchiveType.H5, ArchiveType.ARK):
                return BVR(
                    rspecifier,
                    path_prefix,
                    frame_length=frame_length,
                    frame_shift=frame_shift,
                    snip_edges=snip_edges,
                )
            raise ValueError(
                "VADReaderFactory only supports H5/ARK archive inputs for "
                f"ARCHIVE spec_type, got archive_type={rspecifier.archive_type}"
            )

        if rspecifier.archive_type in (ArchiveType.H5, ArchiveType.ARK):
            return BVR(
                rspecifier,
                path_prefix,
                frame_length=frame_length,
                frame_shift=frame_shift,
                snip_edges=snip_edges,
            )
        if rspecifier.archive_type == ArchiveType.TABLE:
            return TVR(rspecifier.archive, path_prefix=path_prefix)

        raise ValueError(
            "VADReaderFactory only supports H5/ARK/TABLE inputs, "
            f"got archive_type={rspecifier.archive_type}"
        )

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter kwargs to those accepted by :meth:`create`."""
        valid_args = (
            "path_prefix",
            "frame_shift",
            "frame_length",
            "snip_edges",
        )
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Register VAD reader arguments in a ``jsonargparse`` parser.

        Args:
          parser: Target parser to augment.
          prefix: Optional nested argument prefix.
        """
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
