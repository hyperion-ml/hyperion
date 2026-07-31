"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from ..utils import PathLike
from ..utils.vad_set import VADSet
from ..utils.vad_utils import vad_timestamps_to_bin
from .data_reader import DataReader
from .vad_reader import FrameCountArg, FrameIndexArg, ReadKeys, TimeArg, VADReader


class TableVADReader(VADReader):
    """Read VAD timestamps from table/script files.

    The input ``file_path`` points to a VAD table (for example CSV/TSV or an
    equivalent script managed by :class:`~hyperion.utils.vad_set.VADSet`).
    Each key maps to a timestamps file containing ``start``/``end`` columns.

    Examples:
      >>> from hyperion.io.table_vad_reader import TableVADReader
      >>> with TableVADReader("data/vad_index.csv") as r:
      ...     marks = r.read_time_marks(["utt1"])
      ...     vad = r.read_binary(["utt1", "utt2"], frame_length=25, frame_shift=10)
    """

    def __init__(
        self,
        file_path: PathLike,
        path_prefix: Optional[PathLike] = None,
    ) -> None:
        """Initialize table-based VAD reader.

        Args:
          file_path: VADSet table path.
          path_prefix: Optional prefix prepended to each ``storage_path`` entry.
        """
        file_path_str = str(file_path)
        if file_path_str.startswith(("csv:", "tsv:")):
            file_path = file_path_str[4:]

        super().__init__(file_path)
        self.vad_set = VADSet.load(file_path)
        if path_prefix is not None:
            path_prefix = Path(path_prefix)
            self.vad_set["storage_path"] = self.vad_set["storage_path"].apply(
                lambda x: path_prefix / x
            )

    @property
    def keys(self) -> np.ndarray:
        """Array of available recording keys."""
        return self.vad_set["id"].values

    @property
    def ids(self) -> np.ndarray:
        """Alias for :attr:`keys`."""
        return self.vad_set["id"].values

    def close(self) -> None:
        """No-op close method (reader keeps no open file handles)."""
        return None

    def read_num_frames(self, keys: ReadKeys) -> np.ndarray:
        """Read VAD lengths in frames for each key."""
        bin_vad = self.read_binary(keys, squeeze=False)
        return np.asarray([len(vad_i) for vad_i in bin_vad], dtype=int)

    def read_binary(
        self,
        keys: ReadKeys,
        squeeze: bool = False,
        t_start: TimeArg = 0,
        duration: Optional[TimeArg] = None,
        offset_frames: FrameIndexArg = 0,
        num_frames: FrameCountArg = None,
        frame_length: float = 25,
        frame_shift: float = 10,
        snip_edges: bool = False,
    ) -> Union[List[np.ndarray], np.ndarray]:
        """Read binary VAD vectors from timestamp marks.

        Args:
          keys: Recording key or list/array of keys.
          squeeze: If True, stack outputs when shapes are compatible.
          t_start: Time offset(s) in seconds used to crop marks.
          duration: Duration(s) in seconds used to limit marks.
          offset_frames: Frame offset(s). Only ``0`` is currently supported.
          num_frames: Optional maximum number of output frames.
          frame_length: Frame length in milliseconds.
          frame_shift: Frame shift in milliseconds.
          snip_edges: Snip-edges flag used for time-to-frame conversion.

        Returns:
          List of binary VAD vectors, or stacked numpy array when ``squeeze=True``.
        """

        if isinstance(keys, str):
            keys = [keys]

        offset_arr = np.asarray(offset_frames)
        if np.any(offset_arr != 0):
            raise ValueError("offset_frames is not supported and must be 0")

        if duration is not None:
            num_frames = self._duration_to_num_frames(
                duration,
                frame_length=frame_length,
                frame_shift=frame_shift,
                snip_edges=snip_edges,
            )

        offset_is_list, num_frames_is_list = self._assert_offsets_num_frames(
            keys, offset_frames, num_frames
        )
        time_marks = self.read_time_marks(keys, t_start, duration)
        output = []
        for i, time_marks_i in enumerate(time_marks):
            num_frames_i = num_frames[i] if num_frames_is_list else num_frames
            vad_i = vad_timestamps_to_bin(
                time_marks_i.start.values,
                time_marks_i.end.values,
                frame_length=frame_length,
                frame_shift=frame_shift,
                snip_edges=snip_edges,
                max_frames=num_frames_i,
            )

            output.append(vad_i)

        if squeeze:
            output = DataReader._squeeze(output, self.permissive)

        return output

    def read_time_marks(
        self,
        keys: ReadKeys,
        t_start: TimeArg = 0,
        duration: Optional[TimeArg] = None,
        merge_tol: float = 0.001,
    ) -> List[pd.DataFrame]:
        """Read timestamp marks for keys and optionally crop by time window.

        Args:
          keys: Recording key or list/array of keys.
          t_start: Start time(s) in seconds.
          duration: Duration(s) in seconds. If provided and > 0, marks are
            cropped to ``[t_start, t_start + duration)``.
          merge_tol: Reserved for API compatibility.

        Returns:
          List of data frames with ``start`` and ``end`` columns.
        """
        del merge_tol  # kept for API compatibility

        if isinstance(keys, str):
            keys = [keys]

        t_start_is_list, duration_is_list = self._assert_offsets_num_frames(
            keys,
            t_start,
            duration,
        )

        output = []
        for i, key in enumerate(keys):
            if key not in self.vad_set.index:
                if self.permissive:
                    output.append(pd.DataFrame(columns=["start", "end"]))
                    continue
                raise KeyError(f"Key {key} not found")

            vad_file = Path(self.vad_set.loc[key, "storage_path"])
            if vad_file.suffix == ".tsv":
                sep = "\t"
            else:
                sep = ","

            time_marks = pd.read_csv(vad_file, sep=sep)
            t_start_i = t_start[i] if t_start_is_list else t_start
            duration_i = duration[i] if duration_is_list else duration
            if t_start_i > 0.0:
                # Keep segments that overlap [t_start_i, +inf) and clip starts.
                time_marks = time_marks.loc[time_marks.end > t_start_i]
                idx = time_marks.start < t_start_i
                time_marks.loc[idx, "start"] = t_start_i

            if duration_i is not None and duration_i > 0.0:
                max_t_end_i = t_start_i + duration_i
                time_marks = time_marks.loc[time_marks.start < max_t_end_i]
                idx = time_marks.end > max_t_end_i
                time_marks.loc[idx, "end"] = max_t_end_i

            output.append(time_marks)

        return output
