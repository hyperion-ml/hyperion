"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from types import TracebackType
from typing import Optional, Type

from ..utils import PathLike
from .data_writer import DataWriter, MetadataArg, WriteData, WriteKeys


class Int32Writer(DataWriter):
    """Placeholder writer for int32 output.

    This class is currently not implemented.
    """

    def __init__(self, wspecifier: PathLike) -> None:
        """Initialize placeholder int32 writer."""
        super().__init__(wspecifier)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()

    def close(self) -> None:
        raise NotImplementedError("Int32Writer is not implemented")

    def flush(self) -> None:
        raise NotImplementedError("Int32Writer is not implemented")

    def write(
        self, keys: WriteKeys, data: WriteData, metadata: MetadataArg = None
    ) -> None:
        raise NotImplementedError("Int32Writer is not implemented")
