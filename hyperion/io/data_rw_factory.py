"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from jsonargparse import ActionParser, ArgumentParser

from ..utils import PathLike
from ..utils.kaldi_matrix import compression_methods
from .ark_data_reader import RandomAccessArkDataReader as RADR
from .ark_data_reader import SequentialArkFileDataReader as SAFDR
from .ark_data_reader import SequentialArkScriptDataReader as SASDR
from .ark_data_writer import ArkDataWriter as ADW
from .h5_data_reader import RandomAccessH5FileDataReader as RH5FDR
from .h5_data_reader import RandomAccessH5ScriptDataReader as RH5SDR
from .h5_data_reader import SequentialH5FileDataReader as SH5FDR
from .h5_data_reader import SequentialH5ScriptDataReader as SH5SDR
from .h5_data_writer import H5DataWriter as H5DW
from .rw_specifiers import ArchiveType, RSpecifier, RSpecType, WSpecifier, WSpecType

WriteSpecifierArg = Union[PathLike, WSpecifier]
ReadSpecifierArg = Union[PathLike, RSpecifier]
ReaderTransform = Callable[[np.ndarray], np.ndarray]
WriterType = Union[H5DW, ADW]
SequentialReaderType = Union[SH5FDR, SAFDR, SH5SDR, SASDR]
RandomReaderType = Union[RH5FDR, RH5SDR, RADR]


def _looks_like_wspecifier(value: str) -> bool:
    """Return True when ``value`` looks like an explicit Kaldi write specifier."""
    if ":" not in value:
        return False

    # Do not treat Windows-style drive paths (e.g. C:\foo\bar.ark) as specifiers.
    if re.match(r"^[A-Za-z]:[\\/]", value):
        return False

    options = value.split(":", 1)[0].split(",")
    valid_options = {
        "h5",
        "ark",
        "audio",
        "segments",
        "rttm",
        "scp",
        "csv",
        "tsv",
        "t",
        "b",
        "f",
        "nf",
        "p",
    }
    return all(opt in valid_options for opt in options)


class DataWriterFactory:
    """Factory that builds feature writers for H5 and Ark outputs.

    Usage examples:
      Create an H5 writer with CSV index output:
      >>> w = DataWriterFactory.create("h5,csv:out/feat.h5,out/feat.csv")
      >>> w.close()

      Write one record using an H5+CSV writer:
      >>> import numpy as np
      >>> w = DataWriterFactory.create(
      ...     "h5,csv:out/feat.h5,out/feat.csv",
      ...     metadata_columns=["speaker"],
      ... )
      >>> x = np.random.randn(100, 80).astype("float32")
      >>> w.write("utt1", x, metadata={"speaker": "spk1"})
      >>> w.close()

      Create an Ark writer with CSV index, compression and metadata columns:
      >>> w = DataWriterFactory.create(
      ...     "ark,csv:out/feat.ark,out/feat.csv",
      ...     compress=True,
      ...     compression_method="auto",
      ...     metadata_columns=["speaker", "session"],
      ... )
      >>> w.close()

      Parse external kwargs before creating a writer:
      >>> writer_kwargs = DataWriterFactory.filter_args(
      ...     compress=True, compression_method="speech_feat"
      ... )
      >>> w = DataWriterFactory.create(
      ...     "h5:out/feat.h5", compress=writer_kwargs["compress"]
      ... )
      >>> w.close()

      Create an Ark writer with a CSV sidecar:
      >>> w = DataWriterFactory.create("ark,csv:out/feat.ark,out/feat.csv")
      >>> w.close()
    """

    @staticmethod
    def create(
        wspecifier: WriteSpecifierArg,
        compress: bool = False,
        compression_method: str = "auto",
        metadata_columns: Optional[List[str]] = None,
    ) -> WriterType:
        """Create a writer instance from a write specifier.

        Args:
          wspecifier: Write specifier as string/path or pre-parsed ``WSpecifier``.
          compress: If True, enable Kaldi compression when supported by the writer.
          compression_method: Kaldi compression method name.
          metadata_columns: Optional metadata column names for CSV/TSV script outputs.

        Returns:
          ``H5DataWriter`` or ``ArkDataWriter`` depending on the parsed archive type.

        Raises:
          ValueError: If the specifier is not an archive/both specifier or if the
            archive type is not supported by this factory.
        """
        if not isinstance(wspecifier, WSpecifier):
            wspecifier_str = str(wspecifier)
            if not _looks_like_wspecifier(wspecifier_str):
                suffix = Path(wspecifier_str).suffix.lower()
                if suffix == ".ark":
                    wspecifier_str = f"ark:{wspecifier_str}"
                elif suffix in [".h5", ".hdf5"]:
                    wspecifier_str = f"h5:{wspecifier_str}"
                elif suffix in [".scp", ".csv"]:
                    wspecifier_str = f"{suffix[1:]}:{wspecifier_str}"
                elif suffix == ".tsv":
                    wspecifier_str = f"tsv:{wspecifier_str}"
            wspecifier = WSpecifier.create(wspecifier_str)

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
                    metadata_columns=metadata_columns,
                )
            if wspecifier.archive_type == ArchiveType.ARK:
                return ADW(
                    wspecifier.archive,
                    wspecifier.script,
                    binary=wspecifier.binary,
                    flush=wspecifier.flush,
                    compress=compress,
                    compression_method=compression_method,
                    metadata_columns=metadata_columns,
                )
            raise ValueError(
                "DataWriterFactory only supports H5/ARK archive types, "
                f"got archive_type={wspecifier.archive_type}"
            )

        raise ValueError(
            "DataWriterFactory requires an archive write specifier "
            f"(got spec_type={wspecifier.spec_type})"
        )

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter a kwargs dictionary to only writer-factory arguments.

        Args:
          kwargs: Arbitrary keyword arguments from CLI/config.

        Returns:
          Dictionary with keys accepted by :meth:`create`.
        """
        valid_args = ("compress", "compression_method", "metadata_columns")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Register writer-factory arguments in a ``jsonargparse`` parser.

        Args:
          parser: Target parser to augment.
          prefix: Optional group prefix. When provided, arguments are nested under
            ``--<prefix>`` via ``ActionParser``.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument("--compress", default=False, action="store_true")
        parser.add_argument(
            "--compression-method", default="auto", choices=compression_methods
        )
        # parser.add_argument(
        #     "--metadata-columns",
        #     type=List[str],
        #     default=None,
        #     help=("metadata column names to add to CSV/TSV script outputs"),
        # )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))


class SequentialDataReaderFactory:
    """Factory that builds sequential readers for H5/Ark sources.

    Usage examples:
      Create a sequential H5 file reader:
      >>> r = SequentialDataReaderFactory.create("h5:data/feat.h5")
      >>> keys, data = r.read(10)
      >>> r.close()

      Create a sequential reader from CSV script with path prefix:
      >>> r = SequentialDataReaderFactory.create(
      ...     "csv:data/feat.csv",
      ...     path_prefix="/mnt/storage",
      ...     part_idx=1,
      ...     num_parts=4,
      ... )
      >>> r.close()

      Parse kwargs from a larger config dictionary:
      >>> reader_kwargs = SequentialDataReaderFactory.filter_args(
      ...     path_prefix="/mnt/storage", part_idx=2, num_parts=8
      ... )
      >>> r = SequentialDataReaderFactory.create(
      ...     "csv:data/feat.csv", path_prefix=reader_kwargs["path_prefix"]
      ... )
      >>> r.close()
    """

    @staticmethod
    def create(
        rspecifier: ReadSpecifierArg,
        path_prefix: Optional[PathLike] = None,
        **kwargs: Any,
    ) -> SequentialReaderType:
        """Create a sequential reader from a read specifier.

        Args:
          rspecifier: Read specifier as string/path or pre-parsed ``RSpecifier``.
          path_prefix: Optional path prefix prepended to script entries.
          kwargs: Extra reader options (for example ``part_idx`` and ``num_parts``).

        Returns:
          One of sequential reader implementations for H5 or Ark.
        """

        if not isinstance(rspecifier, RSpecifier):
            rspecifier = RSpecifier.create(str(rspecifier))

        if rspecifier.spec_type == RSpecType.ARCHIVE:
            if rspecifier.archive_type == ArchiveType.H5:
                return SH5FDR(rspecifier.archive, **kwargs)
            if rspecifier.archive_type == ArchiveType.ARK:
                return SAFDR(rspecifier.archive, **kwargs)
            raise ValueError(
                "SequentialDataReaderFactory only supports H5/ARK archive types, "
                f"got archive_type={rspecifier.archive_type}"
            )
        else:
            if rspecifier.archive_type == ArchiveType.H5:
                return SH5SDR(rspecifier.script, path_prefix, **kwargs)
            if rspecifier.archive_type == ArchiveType.ARK:
                return SASDR(rspecifier.script, path_prefix, **kwargs)
            raise ValueError(
                "SequentialDataReaderFactory only supports H5/ARK script types, "
                f"got archive_type={rspecifier.archive_type}"
            )

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter kwargs to arguments accepted by sequential readers."""
        valid_args = ("path_prefix", "part_idx", "num_parts")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Register sequential-reader arguments in a parser.

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
        existing_options = {
            opt for action in parser._actions for opt in action.option_strings
        }
        if "--part-idx" not in existing_options:
            parser.add_argument(
                "--part-idx",
                type=int,
                default=1,
                help=("splits the list of files in num-parts and process part_idx"),
            )
        if "--num-parts" not in existing_options:
            parser.add_argument(
                "--num-parts",
                type=int,
                default=1,
                help=("splits the list of files in num-parts and process part_idx"),
            )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='data reader options')


class RandomAccessDataReaderFactory:
    """Factory that builds random-access readers for H5/Ark scripts.

    Usage examples:
      Create random-access reader from an H5 file:
      >>> r = RandomAccessDataReaderFactory.create("h5:data/feat.h5")
      >>> x = r.read(["utt1", "utt2"])
      >>> r.close()

      Create random-access reader from a CSV script with transform:
      >>> transform = lambda x: x.astype(np.float32)
      >>> r = RandomAccessDataReaderFactory.create(
      ...     "csv:data/feat_h5.csv",
      ...     path_prefix="/mnt/storage",
      ...     transform=transform,
      ... )
      >>> r.close()

      Create random-access Ark reader from an scp file:
      >>> r = RandomAccessDataReaderFactory.create("csv:data/feat_ark.csv")
      >>> r.close()
    """

    @staticmethod
    def create(
        rspecifier: ReadSpecifierArg,
        path_prefix: Optional[PathLike] = None,
        transform: Optional[ReaderTransform] = None,
    ) -> RandomReaderType:
        """Create a random-access reader from a read specifier.

        Args:
          rspecifier: Read specifier as string/path or pre-parsed ``RSpecifier``.
          path_prefix: Optional path prefix prepended to script entries.
          transform: Optional callable applied to each loaded matrix.

        Returns:
          One of random-access reader implementations for H5 or Ark-script.

        Raises:
          ValueError: If random access is requested directly on an Ark archive file
            without an accompanying script.
        """
        if not isinstance(rspecifier, RSpecifier):
            rspecifier = RSpecifier.create(str(rspecifier))
        logging.debug(rspecifier.__dict__)
        if rspecifier.spec_type == RSpecType.ARCHIVE:
            if rspecifier.archive_type == ArchiveType.H5:
                return RH5FDR(
                    rspecifier.archive,
                    transform=transform,
                    permissive=rspecifier.permissive,
                )
            if rspecifier.archive_type == ArchiveType.ARK:
                raise ValueError(
                    "Random access to Ark file %s needs a script file"
                    % rspecifier.archive
                )
            raise ValueError(
                "RandomAccessDataReaderFactory only supports H5 archives for direct "
                f"random access, got archive_type={rspecifier.archive_type}"
            )
        else:
            if rspecifier.archive_type == ArchiveType.H5:
                return RH5SDR(
                    rspecifier.archive,
                    path_prefix,
                    transform=transform,
                    permissive=rspecifier.permissive,
                )
            if rspecifier.archive_type == ArchiveType.ARK:
                return RADR(
                    rspecifier.script,
                    path_prefix,
                    transform=transform,
                    permissive=rspecifier.permissive,
                )
            raise ValueError(
                "RandomAccessDataReaderFactory only supports H5/ARK script types, "
                f"got archive_type={rspecifier.archive_type}"
            )

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter kwargs to arguments accepted by random-access readers."""
        valid_args = ("path_prefix",)
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Register random-access-reader arguments in a parser.

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

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    add_argparse_args = add_class_args
