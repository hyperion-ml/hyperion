"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from copy import deepcopy
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .list_utils import *
from .misc import PathLike


class SegmentList:
    """Class to manipulate segment files

    Attributes:
      segments: Pandas dataframe.
      _index_by_file: if True the df is index by file name, if False by segment id.
      iter_idx: index of the current element for the iterator.
      uniq_file_id: unique file names.
    """

    def __init__(self, segments: pd.DataFrame, index_by_file: bool = True) -> None:
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
    def create(
        cls,
        segment_id: Union[Sequence[str], np.ndarray],
        file_id: Union[Sequence[str], np.ndarray],
        tbeg: Union[Sequence[float], np.ndarray],
        tend: Union[Sequence[float], np.ndarray],
        index_by_file: bool = True,
    ) -> "SegmentList":
        segments = pd.Dataframe(
            {"segment_id": segment_id, "file_id": file_id, "tbeg": tbeg, "tend": tend}
        )
        return cls(segments, index_by_file)

    def validate(self) -> None:
        """Validates the attributes of the SegmentList object."""
        assert np.all(self.segments["tend"] - self.segments["tbeg"] >= 0)

    @property
    def index_by_file(self) -> bool:
        return self._index_by_file

    @index_by_file.setter
    def index_by_file(self, value: bool) -> None:
        self._index_by_file = value
        if self._index_by_file:
            self.segments.index = self.segments.file_id
        else:
            self.segments.index = self.segments.segment_id

    @property
    def file_id(self) -> np.ndarray:
        return np.asarray(self.segments["file_id"])

    @property
    def segment_id(self) -> np.ndarray:
        return np.asarray(self.segments["segment_id"])

    @property
    def tbeg(self) -> np.ndarray:
        return np.asarray(self.segments["tbeg"])

    @property
    def tend(self) -> np.ndarray:
        return np.asarray(self.segments["tend"])

    def copy(self) -> "SegmentList":
        """Makes a copy of the object."""
        return deepcopy(self)

    def segments_ids_from_file(self, file_id: Any) -> np.ndarray:
        """Returns segments_ids corresponding to a given file_id"""
        if self.index_by_file:
            return np.asarray(self.segments.loc[file_id]["segment_id"])
        index = self.segments["file_id"] == file_id
        return np.asarray(self.segments.loc[index]["segment_id"])

    def __iter__(self) -> "SegmentList":
        self.iter_idx = 0
        return self

    def __next__(self) -> Union["SegmentList", pd.Series]:
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

    def __len__(self) -> int:
        """Returns the number of segments in the list."""
        return len(self.segments)

    def __contains__(self, key: Any) -> bool:
        """Returns True if the segments contains the key"""
        return key in self.segments.segment_id

    def getitem_by_key(self, key: str) -> Union["SegmentList", pd.Series]:
        """Access segments by file or segment identifier.

        Args:
          key: Segment or file key.

        Returns:
          A ``SegmentList`` for a file or a row from the segment table.
        """
        if self.index_by_file:
            df = self.segments.loc[key]
            return SegmentList(df, index_by_file=False)
        else:
            return self.segments.loc[key]

    def getitem_by_index(self, index: int) -> Union["SegmentList", pd.Series]:
        """Access segments by integer position.

        Args:
          index: Segment or file position.

        Returns:
          A ``SegmentList`` for a file or a row from the segment table.
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

    def __getitem__(
        self, key: Union[str, int, np.integer]
    ) -> Union["SegmentList", pd.Series]:
        """Access segments by file/segment key or integer position.

        Args:
          key: Segment or file key, or integer position.

        Returns:
          A ``SegmentList`` for a file or a row from the segment table.
        """
        if isinstance(key, str):
            return self.getitem_by_key(key)
        else:
            return self.getitem_by_index(key)

    def save(self, file_path: PathLike, sep: str = " ") -> None:
        """Saves segments to text file.

        Args:
          file_path: File to write the list.
          sep: Separator between the fields
        """
        self.segments[["segment_id", "file_id", "tbeg", "tend"]].to_csv(
            file_path, sep=sep, float_format="%.3f", index=False, header=False
        )

    @classmethod
    def load(
        cls, file_path: PathLike, sep: str = " ", index_by_file: bool = True
    ) -> "SegmentList":
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

    def filter(
        self, filter_key: Union[Sequence[Any], np.ndarray], keep: bool = True
    ) -> "SegmentList":
        if not keep:
            filter_key = np.setdiff1d(np.asarray(self.segments.index), filter_key)
        df = self.segments.loc[filter_key]
        return SegmentList(df, index_by_file=self.index_by_file)

    def split(self, idx: int, num_parts: int) -> "SegmentList":
        if self.index_by_file:
            key, _ = split_list(self.uniq_file_id, idx, num_parts)
        else:
            key, _ = split_list(self.segment_id, idx, num_parts)
        df = self.segments.loc[key]
        return SegmentList(df, index_by_file=self.index_by_file)

    @classmethod
    def merge(
        cls, segment_lists: Sequence["SegmentList"], index_by_file: bool = True
    ) -> "SegmentList":
        dfs = []
        for sl in segment_lists:
            dfs.append(sl.segments)
        df = pd.concat(dfs)
        return cls(df, index_by_file=index_by_file)

    def to_bin_vad(
        self, key: Any, frame_shift: float = 10, num_frames: Optional[int] = None
    ) -> np.ndarray:
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

    def __eq__(self, other: object) -> bool:
        """Equal operator"""
        if not isinstance(other, SegmentList):
            return False
        eq = self.segments.equals(other.segments)
        eq = eq and self.index_by_file == other.index_by_file

        return eq

    def __ne__(self, other: object) -> bool:
        """Non-equal operator"""
        return not self.__eq__(other)

    def __cmp__(self, other: object) -> int:
        """Comparison operator"""
        if self.__eq__(other):
            return 0
        return 1
