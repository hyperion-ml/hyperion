"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from typing import List, Optional, Union

import numpy as np

from ..utils import SegmentList
from ..utils.vad_utils import vad_timestamps_to_bin
from .data_reader import DataReader
from .vad_reader import FrameCountArg, FrameIndexArg, ReadKeys, VADReader


class SegmentVADReader(VADReader):
    """Read VAD from Kaldi-style segment files.

    Examples:
      >>> from hyperion.io.segment_vad_reader import SegmentVADReader
      >>> with SegmentVADReader("data/vad.segments") as r:
      ...     vad = r.read(["utt1", "utt2"], frame_shift=10, frame_length=25)
      ...     ts = r.read_timestamps(["utt1"])
    """

    def __init__(self, segments_file: str, permissive: bool = False) -> None:
        """Initialize segment-based VAD reader.

        Args:
          segments_file: Path to segments file.
          permissive: If True, tolerate missing keys.
        """
        super().__init__(segments_file, permissive)
        self.segments = SegmentList.load(segments_file)

    def close(self) -> None:
        """No-op close method (reader keeps no open file handles)."""
        return None

    def read(
        self,
        keys: ReadKeys,
        squeeze: bool = False,
        offset: FrameIndexArg = 0,
        num_frames: FrameCountArg = 0,
        frame_length: float = 25,
        frame_shift: float = 10,
        snip_edges: bool = False,
        signal_lengths: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Union[List[np.ndarray], np.ndarray]:
        """Read binary VAD vectors for keys.

        Args:
          keys: Recording key or list/array of keys.
          squeeze: If True, stack outputs when shapes are compatible.
          offset: Frame offset(s) applied after conversion.
          num_frames: Number of frames to keep (0 means full length).
          frame_length: Frame length in milliseconds.
          frame_shift: Frame shift in milliseconds.
          snip_edges: Snip-edges flag used for time-to-frame conversion.
          signal_lengths: Optional signal lengths in seconds used as max duration.

        Returns:
          List of binary VAD vectors, or stacked numpy array when ``squeeze=True``.
        """

        if isinstance(keys, str):
            keys = [keys]

        offset_is_list, num_frames_is_list = self._assert_offsets_num_frames(
            keys, offset, num_frames
        )

        vad = []
        for i in range(len(keys)):
            key_i = keys[i]
            try:
                df = self.segments[key_i]
                ts = np.stack((np.asarray(df.tbeg), np.asarray(df.tend)), axis=1)
            except KeyError:
                if self.permissive:
                    ts = np.zeros((0, 2), dtype=float)
                else:
                    raise KeyError(f"Key {key_i} not found")

            signal_length = None if signal_lengths is None else signal_lengths[i]
            vad_i = vad_timestamps_to_bin(
                ts, frame_length / 1000, frame_shift / 1000, snip_edges, signal_length
            )
            offset_i = offset[i] if offset_is_list else offset
            num_frames_i = num_frames[i] if num_frames_is_list else num_frames
            vad_i = self._get_bin_vad_slice(vad_i, offset_i, num_frames_i)
            vad.append(vad_i)

        if squeeze:
            vad = DataReader._squeeze(vad, self.permissive)

        return vad

    def read_timestamps(
        self, keys: ReadKeys, merge_tol: float = 0
    ) -> List[np.ndarray]:
        """Read timestamp intervals for keys.

        Args:
          keys: Recording key or list/array of keys.
          merge_tol: Reserved for API compatibility.

        Returns:
          List of ``[start, end]`` timestamp arrays.
        """
        del merge_tol  # kept for API compatibility

        if isinstance(keys, str):
            keys = [keys]

        ts = []
        for i in range(len(keys)):
            key_i = keys[i]
            try:
                df = self.segments[key_i]
                ts_i = np.stack((np.asarray(df.tbeg), np.asarray(df.tend)), axis=1)
            except KeyError:
                if self.permissive:
                    ts_i = np.zeros((0, 2), dtype=float)
                else:
                    raise KeyError(f"Key {key_i} not found")
            ts.append(ts_i)

        return ts
