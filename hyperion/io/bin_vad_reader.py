"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import List, Optional, Tuple, Union

import numpy as np

from ..utils import PathLike
from ..utils.vad_utils import bin_vad_to_timestamps
from .data_reader import DataReader
from .data_rw_factory import RandomAccessDataReaderFactory as DRF
from .rw_specifiers import RSpecifier
from .vad_reader import FrameCountArg, FrameIndexArg, ReadKeys, TimeArg, VADReader


class BinVADReader(VADReader):
    """Read binary VAD vectors from Ark/HDF5 inputs.

    Attributes:
      r: Underlying random-access feature reader that returns binary VAD vectors.
      frame_shift: Frame shift in milliseconds used for timestamp conversion.
      frame_length: Frame length in milliseconds used for timestamp conversion.
      snip_edges: Whether the source VAD was computed with snip-edges.

    Examples:
      >>> from hyperion.io.bin_vad_reader import BinVADReader
      >>> with BinVADReader("csv:data/vad.csv", frame_length=25, frame_shift=10) as r:
      ...     vad = r.read_binary(["utt1", "utt2"])
      ...     t_start, t_end = r.read_time_marks(["utt1", "utt2"])
    """

    def __init__(
        self,
        rspecifier: Union[PathLike, RSpecifier],
        path_prefix: Optional[PathLike] = None,
        frame_length: float = 25,
        frame_shift: float = 10,
        snip_edges: bool = False,
    ) -> None:
        """Initialize binary VAD reader.

        Args:
          rspecifier: Kaldi-style read specifier or parsed ``RSpecifier``.
          path_prefix: Optional path prefix for script-based readers.
          frame_length: Frame length in milliseconds.
          frame_shift: Frame shift in milliseconds.
          snip_edges: Whether frame extraction used snip-edges.
        """
        r = DRF.create(rspecifier, path_prefix)
        super().__init__(r.file_path, r.permissive)
        self.r = r
        self.frame_shift = frame_shift
        self.frame_length = frame_length
        self.snip_edges = snip_edges

    def close(self) -> None:
        """Close underlying random-access reader resources."""
        self.r.close()

    def read_num_frames(self, keys: ReadKeys) -> np.ndarray:
        """Read VAD vector lengths (in frames) for requested keys."""
        return self.r.read_dims(keys, assert_same_dim=False)

    @property
    def keys(self) -> np.ndarray:
        """Available recording keys."""
        return self.r.keys

    @property
    def ids(self) -> np.ndarray:
        """Alias for :attr:`keys`."""
        return self.r.keys

    def read(
        self,
        keys: ReadKeys,
        squeeze: bool = False,
        offset: FrameIndexArg = 0,
        num_frames: FrameCountArg = 0,
        frame_length: float = 25.0,
        frame_shift: float = 10.0,
        snip_edges: bool = False,
        duration: Optional[TimeArg] = None,
    ) -> Union[List[np.ndarray], np.ndarray]:
        """Read binary VAD vectors.

        Args:
          keys: Recording key or list/array of keys.
          squeeze: If True, stack outputs when shapes are compatible.
          offset: Starting frame offset(s).
          num_frames: Number of frames to return (0 means full length).
          frame_length: Frame length in milliseconds (must match reader config).
          frame_shift: Frame shift in milliseconds (must match reader config).
          snip_edges: Snip-edges flag (must match reader config).
          duration: Optional duration(s) in seconds used to derive ``num_frames``.

        Returns:
          List of binary VAD vectors, or a stacked numpy array when ``squeeze=True``.
        """

        if isinstance(keys, str):
            keys = [keys]

        if not np.isclose(frame_length, self.frame_length, rtol=0.0, atol=1e-8):
            raise ValueError(
                f"frame_length={frame_length} does not match configured value "
                f"{self.frame_length}"
            )
        if not np.isclose(frame_shift, self.frame_shift, rtol=0.0, atol=1e-8):
            raise ValueError(
                f"frame_shift={frame_shift} does not match configured value "
                f"{self.frame_shift}"
            )
        if snip_edges != self.snip_edges:
            raise ValueError(
                f"snip_edges={snip_edges} does not match configured value "
                f"{self.snip_edges}"
            )

        if duration is not None:
            num_frames = self._duration_to_num_frames(
                duration,
                frame_length=frame_length,
                frame_shift=frame_shift,
                snip_edges=snip_edges,
            )

        offset_is_list, num_frames_is_list = self._assert_offsets_num_frames(
            keys, offset, num_frames
        )

        vad = self.r.read(keys)
        output_vad = []
        for i in range(len(keys)):
            vad_i = vad[i].astype(bool, copy=False)
            offset_i = offset[i] if offset_is_list else offset
            num_frames_i = num_frames[i] if num_frames_is_list else num_frames
            vad_i = self._get_bin_vad_slice(vad_i, offset_i, num_frames_i)
            output_vad.append(vad_i)

        if squeeze:
            output_vad = DataReader._squeeze(output_vad, self.permissive)

        return output_vad

    def read_binary(
        self,
        keys: ReadKeys,
        squeeze: bool = False,
        offset: FrameIndexArg = 0,
        num_frames: FrameCountArg = 0,
        frame_length: float = 25.0,
        frame_shift: float = 10.0,
        snip_edges: bool = False,
        duration: Optional[TimeArg] = None,
    ) -> Union[List[np.ndarray], np.ndarray]:
        """Alias for :meth:`read` with identical arguments."""
        return self.read(
            keys,
            squeeze=squeeze,
            offset=offset,
            num_frames=num_frames,
            frame_length=frame_length,
            frame_shift=frame_shift,
            snip_edges=snip_edges,
            duration=duration,
        )

    def read_time_marks(
        self,
        keys: ReadKeys,
        merge_tol: float = 0.001,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Convert binary VAD into start/end timestamp arrays.

        Args:
          keys: Recording key or list/array of keys.
          merge_tol: Timestamp merge tolerance in seconds.

        Returns:
          Tuple ``(t_start, t_end)`` where each element is a list of numpy arrays.
        """
        if isinstance(keys, str):
            keys = [keys]

        vad = self.r.read(keys)
        t_start = []
        t_end = []
        for i in range(len(keys)):
            vad_i = vad[i].astype(bool, copy=False)
            ts_i = bin_vad_to_timestamps(
                vad_i,
                self.frame_length,
                self.frame_shift,
                self.snip_edges,
                merge_tol,
            )
            t_start.append(ts_i[0])
            t_end.append(ts_i[1])

        return t_start, t_end
