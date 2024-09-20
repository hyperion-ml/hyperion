"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging

import numpy as np

from ..hyp_defs import float_cpu
from ..utils.vad_utils import bin_vad_to_timestamps
from .data_rw_factory import RandomAccessDataReaderFactory as DRF
from .vad_reader import VADReader


class BinVADReader(VADReader):
    def __init__(
        self,
        rspecifier,
        path_prefix=None,
        frame_length=25,
        frame_shift=10,
        snip_edges=False,
    ):

        r = DRF.create(rspecifier, path_prefix)
        super().__init__(r.file_path, r.permissive)
        self.r = r
        self.frame_shift = frame_shift
        self.frame_length = frame_length
        self.snip_edges = snip_edges

    def read_num_frames(self, keys):
        return self.r.read_dims(keys, assert_same_dim=False)

    @property
    def keys(self):
        return self.r.keys

    @property
    def ids(self):
        return self.r.keys

    def read(
        self,
        keys,
        squeeze=False,
        offset=0,
        num_frames=0,
        frame_length=25.0,
        frame_shift=10.0,
        snip_edges=False,
        duration=None,
    ):

        if isinstance(keys, str):
            keys = [keys]

        assert frame_length == self.frame_length
        assert frame_shift == self.frame_shift
        assert snip_edges == self.snip_edges

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
            output_vad = self.r._squeeeze(output_vad, self.permissive)

        return output_vad

    def read_binary(
        self,
        keys,
        squeeze=False,
        offset=0,
        num_frames=0,
        frame_length=25.0,
        frame_shift=10.0,
        snip_edges=False,
        duration=None,
    ):
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

    def read_time_marks(self, keys, merge_tol=0.001):
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
