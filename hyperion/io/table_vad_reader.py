"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from ..hyp_defs import float_cpu
from ..utils.vad_set import VADSet
from ..utils.vad_utils import vad_timestamps_to_bin
from .vad_reader import VADReader


class TableVADReader(VADReader):
    def __init__(
        self,
        file_path,
        path_prefix=None,
    ):

        if str(file_path)[:4] == "csv:":
            file_path = str(file_path)[:4]

        super().__init__(file_path)
        self.vad_set = VADSet.load(file_path)
        if path_prefix is not None:
            path_prefix = Path(path_prefix)
            self.vad_set["storage_path"] = self.vad_set["storage_path"].apply(
                lambda x: path_prefix / x
            )

    @property
    def keys(self):
        return self.vad_set["id"]

    @property
    def ids(self):
        return self.vad_set["id"]

    def read_num_frames(self, keys):
        return self.r.read_dims(keys, assert_same_dim=False)

    def read_binary(
        self,
        keys,
        squeeze=False,
        t_start: Union[float, List[float], np.array] = 0,
        duration: Union[float, List[float], np.array] = None,
        offset_frames: Union[int, List[int], np.array] = 0,
        num_frames: Union[int, List[int], np.array, None] = None,
        frame_length: float = 25,
        frame_shift: float = 10,
        snip_edges: bool = False,
    ):

        if isinstance(keys, str):
            keys = [keys]

        assert offset_frames == 0, f"not supported"

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
            output = self.r._squeeeze(output, self.permissive)

        return output

    def read_time_marks(
        self,
        keys,
        t_start: Union[float, List[float], np.array] = 0,
        duration: Union[float, List[float], np.array, None] = None,
        merge_tol: float = 0.001,
    ):
        if isinstance(keys, str):
            keys = [keys]

        t_start_is_list, duration_is_list = self._assert_offsets_num_frames(
            keys,
            t_start,
            duration,
        )

        output = []
        for i, key in enumerate(keys):
            vad_file = Path(self.vad_set.loc[key, "storage_path"])
            if vad_file.suffix == ".tsv":
                sep = "\t"
            else:
                sep = ","

            time_marks = pd.read_csv(vad_file, sep=sep)
            t_start_i = t_start[i] if t_start_is_list else t_start
            duration_i = duration[i] if duration_is_list else duration
            if t_start_i > 0.0:
                time_marks = time_marks.loc[time_marks.start >= t_start_i]

            if duration_i is None or duration_i == 0.0:
                max_t_end_i = t_start_i + duration_i
                time_marks = time_marks.loc[time_marks.start < max_t_end_i]
                idx = time_marks.end > max_t_end_i
                time_marks.loc[idx, "end"] = max_t_end_i

            output.append(time_marks)

        return output
