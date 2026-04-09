"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from types import TracebackType
from typing import List, Optional, Tuple, Type, Union

import numpy as np

from ..utils import PathLike

ReadKeys = Union[str, List[str], np.ndarray]
FrameIndexArg = Union[int, List[int], np.ndarray]
FrameCountArg = Union[int, List[int], np.ndarray, None]
TimeArg = Union[float, List[float], np.ndarray]


class VADReader:
    """Base class for readers that return voice-activity detection (VAD) data.

    Attributes:
       file_path: Source file/path used by the reader.
       permissive: If True, if the data that we want to read is not in the file
                   it returns an empty matrix, if False it raises an exception.

    """

    def __init__(self, file_path: PathLike, permissive: bool = False) -> None:
        """Initialize common VAD reader state."""
        self.file_path = file_path
        self.permissive = permissive

    def __enter__(self) -> "VADReader":
        """Enter context manager.

        Returns:
          Self, so the reader can be used in ``with`` blocks.
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Exit context manager and close underlying resources."""
        self.close()

    def close(self) -> None:
        """Close any underlying resources."""
        pass

    @staticmethod
    def _assert_offsets_num_frames(
        keys: Union[List[str], np.ndarray],
        offset: Union[FrameIndexArg, TimeArg, None],
        num_frames: Union[FrameCountArg, TimeArg, None],
    ) -> Tuple[bool, bool]:
        """Check whether offset/length arguments are per-key sequences.

        Args:
          keys: Key list used in the read operation.
          offset: Scalar or per-key offsets.
          num_frames: Scalar or per-key lengths.

        Returns:
          Tuple ``(offset_is_list, num_frames_is_list)``.
        """
        n = len(keys)
        offset_is_list = isinstance(offset, list) or isinstance(offset, np.ndarray)
        num_frames_is_list = isinstance(num_frames, list) or isinstance(
            num_frames, np.ndarray
        )
        if isinstance(offset, np.ndarray) and offset.ndim == 0:
            offset_is_list = False
        if isinstance(num_frames, np.ndarray) and num_frames.ndim == 0:
            num_frames_is_list = False

        if offset_is_list:
            if len(offset) != n:
                raise ValueError(
                    f"offset has {len(offset)} items but {n} keys were provided"
                )
        if num_frames_is_list:
            if len(num_frames) != n:
                raise ValueError(
                    f"num_frames has {len(num_frames)} items but {n} keys were provided"
                )

        return offset_is_list, num_frames_is_list

    @staticmethod
    def _get_bin_vad_slice(
        vad: np.ndarray,
        offset: int,
        num_frames: int,
    ) -> np.ndarray:
        """Apply frame offset and frame count cropping/padding to a binary VAD."""
        if offset > 0:
            vad = vad[offset:]

        if num_frames > 0:
            n = len(vad)
            if n > num_frames:
                vad = vad[:num_frames]
            elif n < num_frames:
                new_vad = np.zeros((num_frames,), dtype=bool)
                new_vad[:n] = vad
                vad = new_vad

        return vad

    @staticmethod
    def _duration_to_num_frames(
        duration: TimeArg,
        frame_length: float,
        frame_shift: float,
        snip_edges: bool,
    ) -> Union[int, np.ndarray]:
        """Convert duration(s) in seconds into frame counts.

        Args:
          duration: Duration value(s) in seconds.
          frame_length: Frame length in milliseconds.
          frame_shift: Frame shift in milliseconds.
          snip_edges: Whether frame extraction uses snip-edges behavior.

        Returns:
          Integer frame count for scalar duration input, or a numpy array of
          integer frame counts for vectorized duration input.
        """
        frame_length = frame_length / 1000
        frame_shift = frame_shift / 1000
        duration = np.asarray(duration, dtype=float)
        if snip_edges:
            num_frames = np.floor(
                (duration - frame_length + frame_shift) / frame_shift
            )
        else:
            num_frames = np.round(duration / frame_shift)
        num_frames = np.maximum(num_frames, 0)

        if np.ndim(num_frames) == 0:
            return int(num_frames)

        return np.asarray(num_frames, dtype=int)
