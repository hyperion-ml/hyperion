"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import os.path as path
import logging
from copy import deepcopy

import numpy as np
import pandas as pd

from .list_utils import *


class SegmentList(object):
    """Class to manipulate segment files

    Attributes:
      segments: Pandas dataframe.
      _index_by_file: if True the df is index by file name, if False by segment id.
      iter_idx: index of the current element for the iterator.
      uniq_file_id: unique file names.
    """

    def __init__(self, segments, index_by_file=True):
        self.segments = segments
        self._index_by_file = index_by_file
        if index_by_file:
            self.segments.index = self.segments.file_id
        else:
            self.segments.index = self.segments.segment_id
        self.validate()
        self.uniq_file_id = np.unique(self.segments.file_id)
        self.iter_idx = 0

    @classmethod
    def create(cls, segment_id, file_id, tbeg, tend, index_by_file=True):
        segments = pd.Dataframe(
            {"segment_id": segment_id, "file_id": file_id, "tbeg": tbeg, "tend": tend}
        )
        return cls(segments, index_by_file)

    def validate(self):
        """Validates the attributes of the SegmentList object."""
        # logging.debug(len(self.segments['tend']-self.segments['tbeg']>=0))
        # logging.debug(len(self.segments['tbeg'][1:]))
        # logging.debug(len(self.segments['tbeg'][:-1]))
        # logging.debug(self.segments['tbeg'][1:]-self.segments['tbeg'][:-1])
        # logging.debug(len(self.segments['tbeg'][1:]-self.segments['tbeg'][:-1]>=0))
        # logging.debug(len(self.file_id[1:] != self.file_id[:-1]))
        assert np.all(self.segments["tend"] - self.segments["tbeg"] >= 0)
        # assert np.all(np.logical_or(self.tbeg[1:]-self.tbeg[:-1]>=0,
        #                            self.file_id[1:] != self.file_id[:-1]))

    @property
    def index_by_file(self):
        return self._index_by_file

    @index_by_file.setter
    def index_by_file(self, value):
        self._index_by_file = value
        if self._index_by_file:
            self.segments.index = self.segments.file_id
        else:
            self.segments.index = self.segments.segment_id

    @property
    def file_id(self):
        return np.asarray(self.segments["file_id"])

    @property
    def segment_id(self):
        return np.asarray(self.segments["segment_id"])

    @property
    def tbeg(self):
        return np.asarray(self.segments["tbeg"])

    @property
    def tend(self):
        return np.asarray(self.segments["tend"])

    def copy(self):
        """Makes a copy of the object."""
        return deepcopy(self)

    def segments_ids_from_file(self, file_id):
        """Returns segments_ids corresponding to a given file_id"""
        if self.index_by_file:
            return np.asarray(self.segments.loc[file_id]["segment_id"])
        index = self.segments["file_id"] == file_id
        return np.asarray(self.segments.loc[index]["segment_id"])

    def __iter__(self):
        self.iter_idx = 0
        return self

    def __next__(self):
        if self.index_by_file:
            if self.iter_idx < len(self.uniq_file_id):
                r = self.getitem_by_key(self.uniq_file_id[self.iter_idx])
            else:
                raise StopIteration()
        else:
            if self.iter_idx < len(self.segments):
                # r = self.__getitem__(self.segments['segment_id'].iloc[self.iter_idx])
                r = self.segments.iloc[self.iter_idx]
            else:
                raise StopIteration()

        self.iter_idx += 1
        return r

    def __len__(self):
        """Returns the number of segments in the list."""
        return len(self.segments)

    def __contains__(self, key):
        """Returns True if the segments contains the key"""
        return key in self.segments.segment_id

    def getitem_by_key(self, key):
        """It acceses the segments by file_id or segment_id
           like in a ditionary, e.g.:
           If input is a string key:
               segmetns = SegmentList(...)
               segment, tbeg, tend = segments.getiem_by_key('file')
        Args:
          key: Segment or file key
        Returns:
          if index_by_file is True if returns segments of a given file_id
          in SegmentsList format, else it returns DataFrame
        """
        if self.index_by_file:
            df = self.segments.loc[key]
            return SegmentList(df, index_by_file=False)
        else:
            return self.segments.loc[key]

    def getitem_by_index(self, index):
        """It accesses the segments by index
           like in a ditionary, e.g.:
           If input is a string key:
               segmetns = SegmentList(...)
               segment, tbeg, tend = segments.getitem_by_index(0)
        Args:
          key: Segment or file key
        Returns:
          if index_by_file is True if returns segments of a given file_id
          in SegmentsList format, else it returns DataFrame
        """
        if self.index_by_file:
            if index < len(self.uniq_file_id):
                return self.getitem_by_key(self.uniq_file_id[self.iter_idx])
            else:
                raise Exception(
                    "SegmentList error index>=num_files (%d,%d)"
                    % (index, len(self.uniq_file_id))
                )
        else:
            if index < len(self.segments):
                return self.segments.iloc[index]
            else:
                raise Exception(
                    "SegmentList error index>=num_segments (%d,%d)" % (index, len(self))
                )

    def __getitem__(self, key):
        """It accesses the de segments by file_id or segment_id
           like in a ditionary, e.g.:
           If input is a string key:
               segmetns = SegmentList(...)
               segment, tbeg, tend = segments['file']
        Args:
          key: Segment or file key
        Returns:
          if index_by_file is True if returns segments of a given file_id
          in SegmentsList format, else it returns DataFrame
        """
        if isinstance(key, str):
            return self.getitem_by_key(key)
        else:
            return self.getitem_by_index(key)

    def save(self, file_path, sep=" "):
        """Saves segments to text file.

        Args:
          file_path: File to write the list.
          sep: Separator between the fields
        """
        self.segments[["segment_id", "file_id", "tbeg", "tend"]].to_csv(
            file_path, sep=sep, float_format="%.3f", index=False, header=False
        )

    @classmethod
    def load(cls, file_path, sep=" ", index_by_file=True):
        """Loads script list from text file.

        Args:
          file_path: File to read the list.
          sep: Separator between the key and file_path in the text file.

        Returns:
          SegmentList object.
        """
        df = pd.read_csv(
            file_path,
            sep=sep,
            header=None,
            names=["segment_id", "file_id", "tbeg", "tend"],
        )
        return cls(df, index_by_file=index_by_file)

    def filter(self, filter_key, keep=True):
        if not keep:
            filter_key = np.setdiff1d(np.asarray(self.segments.index), filter_key)
        df = self.segments.loc[filter_key]
        return SegmentList(df, index_by_file=self.index_by_file)

    def split(self, idx, num_parts):
        if self.index_by_file:
            key, _ = split_list(self.uniq_file_id, idx, num_parts)
        else:
            key, _ = split_list(self.segment_id, idx, num_parts)
        df = self.segments.loc[key]
        return SegmentList(df, index_by_file=self.index_by_file)

    @classmethod
    def merge(cls, segment_lists, index_by_file=True):
        dfs = []
        for sl in segment_lists:
            dfs.append(sl.segments)
        df = pd.concat(dfs)
        return cls(df, index_by_file=index_by_file)

    def to_bin_vad(self, key, frame_shift=10, num_frames=None):
        """Converts segments to binary VAD

        Args:
          key: Segment or file key
          frame_shift: frame_shift in milliseconds
          num_frames: number of frames of file corresponding to key,
                      if None it takes the maximum tend for file
        Returns:
          if index_by_file is True if returns VAD joining all segments of one file
          else if returns VAD for one given segment

        """
        tbeg = np.round(
            np.array(self.segments.loc[key]["tbeg"], dtype=float, ndmin=1)
            * 1000
            / frame_shift
        ).astype(dtype=int)
        tend = np.round(
            np.array(self.segments.loc[key]["tend"], dtype=float, ndmin=1)
            * 1000
            / frame_shift
        ).astype(dtype=int)

        if num_frames is None:
            if self.index_by_file:
                num_frames = tend[-1]
            else:
                file_id = self.segments.loc[key]["file_id"]
                sel_idx = self.segments["file_id"] == file_id
                num_frames = int(
                    np.round(
                        self.segments[sel_idx]["tend"].max() * 1000 / self.frame_shift
                    )
                )

        tend = np.minimum(num_frames - 1, tend)
        vad = np.zeros((num_frames,), dtype=int)
        for j in range(len(tbeg)):
            vad[tbeg[j] : tend[j] + 1] = 1

        return vad

    def __eq__(self, other):
        """Equal operator"""
        eq = self.segments.equals(other.segments)
        eq = eq and self.index_by_file == other.index_by_file

        return eq

    def __ne__(self, other):
        """Non-equal operator"""
        return not self.__eq__(other)

    def __cmp__(self, other):
        """Comparison operator"""
        if self.__eq__(other):
            return 0
        return 1
