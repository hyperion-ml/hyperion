"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
import numpy as np

from ..hyp_defs import float_cpu
from ..utils import SegmentList
from ..utils.vad_utils import vad_timestamps_to_bin
from .vad_reader import VADReader
from .data_reader import DataReader


class SegmentVADReader(VADReader):
    def __init__(self, segments_file, permissive=False):
        super().__init__(segments_file, permissive)
        self.segments = SegmentList.load(segments_file)

    def read(
        self,
        keys,
        squeeze=False,
        offset=0,
        num_frames=0,
        frame_length=25,
        frame_shift=10,
        snip_edges=False,
        signal_lengths=None,
    ):

        if isinstance(keys, str):
            keys = [keys]

        offset_is_list, num_frames_is_list = self._assert_offsets_num_frames(
            keys, offset, num_frames
        )

        vad = []
        for i in range(len(keys)):
            df = self.segments[keys[i]]
            ts = np.concatenate((df.tbeg[:, None], df.tend[:, None]), axis=1)
            signal_length = None if signal_lengths is None else signal_lengths[i]
            vad_i = vad_timestamps_to_bin(
                ts, frame_length / 1000, frame_shift / 1000, snip_edges, signal_length
            )
            offset_i = offset[i] if offset_is_list else offset
            num_frames_i = num_frames[i] if num_frames_is_list else num_frames
            vad_i = self._get_bin_vad_slice(vad_i, offset_i, num_frames_i)
            vad.append(vad_i)

        if squeeze:
            DataReader._squeeze(vad, self.permissive)

        return vad

    def read_timestamps(self, keys, merge_tol=0):

        if isinstance(keys, str):
            keys = [keys]

        ts = []
        for i in range(len(keys)):
            df = self.segments[keys[i]]
            ts_i = np.concatenate((df.tbeg[:, None], df.tend[:, None]), axis=1)
            ts.append(ts_i)

        return ts
