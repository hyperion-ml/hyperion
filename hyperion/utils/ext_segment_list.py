"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import os.path as path
import logging
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import pandas as pd

from .list_utils import *


class ExtSegmentList(object):
    """Class to manipulate extended segment files

    Attributes:
      segments: segments dataframe.
      files: file info dataframe.
      ext_segments: extended segments dataframe.
      _index_column: Column used as index in the dataframes (file_id, ext_segment_id, segment_id, series_id)
      iter_idx: index of the current element for the iterator.
      _uniq_file_id: unique file names.
      _uniq_series_id: unique series id.
    """

    def __init__(self, segments, ext_segments=None, files=None, index_column="file_id"):
        self.segments = segments
        if files is None:
            file_id = self.segments["file_id"].unique()
            files = pd.DataFrame({"file_id": file_id, "series_id": file_id})

        if ext_segments is None:
            if not "ext_segment_id" in self.segments:
                self.segments = self.segments.assign(
                    ext_segment_id=self.segments["segment_id"].values
                )
            ext_segment_id = self.segments.ext_segment_id.unique()
            ext_segments = pd.DataFrame(
                {"ext_segment_id": ext_segment_id, "name": np.nan, "score": np.nan}
            )

        self.files = files
        self.ext_segments = ext_segments
        self.index_column = index_column
        self.validate()
        self._uniq_series_id = None
        self.iter_idx = 0

    @classmethod
    def create(
        cls,
        segment_id,
        file_id,
        tbeg,
        tend,
        ext_segment_id=None,
        series_id=None,
        name=np.nan,
        score=np.nan,
        index_column="file_id",
    ):

        if ext_segment_id is None:
            ext_segment_id = segment_id

        segments = pd.DataFrame(
            {
                "segment_id": segment_id,
                "file_id": file_id,
                "tbeg": tbeg,
                "tend": tend,
                "ext_segment_id": ext_segment_id,
            }
        )

        if series_id is None:
            u_file_id = self.segments["file_id"].unique()
            files = pd.DataFrame({"file_id": u_file_id, "series_id": u_file_id})
        else:
            file_id = [f for f in v for k, v in series_id.items()]
            series_id = [k for f in v for k, v in series_id.items()]
            files = pd.DataFrame({"file_id": file_id, "series_id": series_id})

        if isinstance(name, str):
            ext_segment_id = segments["ext_segment_id"].unique()
        elif isinstance(name, dict):
            ext_segment_id = [k for k, v in name.items()]
            name = [v for k, v in name.items()]

        if isinstance(score, dict):
            score = [score[k] for k in ext_segment_id]

        ext_segments = pd.DataFrame(
            {"ext_segment_id": ext_segment_id, "name": name, "score": score}
        )

        return cls(segments, ext_segments, files, index_column)

    @classmethod
    def create_from_segment_list(
        cls,
        segment_list,
        series_id=None,
        name=np.nan,
        score=np.nan,
        index_column="file_id",
    ):

        segments = deepcopy(segment_list.segments)
        segments = segments.assign(ext_segment_id=segments["segment_id"])
        ext_segment_id = segments.ext_segment_id.unique()

        if not np.isnan(name):
            name = [name[k] for k in segments["ext_segment_id"].values]
        ext_segments = pd.DataFrame(
            {
                "ext_segment_id": segments["ext_segment_id"].values,
                "name": name,
                "score": score,
            }
        )

        if series_id is None:
            u_file_id = segments["file_id"].unique()
            files = pd.DataFrame({"file_id": u_file_id, "series_id": u_file_id})
        else:
            file_id = [f for f in v for k, v in series_id.items()]
            series_id = [k for f in v for k, v in series_id.items()]
            files = pd.DataFrame({"file_id": file_id, "series_id": series_id})

        return cls(segments, ext_segments, files, index_column)

    def validate(self):
        """Validates the attributes of the SegmentList object."""

        assert np.all(self.segments["tend"] - self.segments["tbeg"] >= 0)
        ok_tbeg = np.logical_or(
            self.tbeg[1:] - self.tbeg[:-1] >= 0, self.file_id[1:] != self.file_id[:-1]
        )
        if not np.all(ok_tbeg):
            bad_tbeg = np.logical_not(ok_tbeg)
            logging.critical(
                {"file_id": self.file_id[1:][bad_tbeg], "tbeg": self.tbeg[1:][bad_tbeg]}
            )
            raise Exception("tbeg is not in the right order")

    @property
    def index_column(self):
        return self._index_column

    @index_column.setter
    def index_column(self, value):

        self._index_column = value
        self.ext_segments.index = self.ext_segments.ext_segment_id
        if value == "file_id":
            self.segments.index = self.segments.file_id
            self.files.index = self.files.file_id
        elif value == "segment_id":
            self.segments.index = self.segments.segment_id
            self.files.index = self.files.file_id
        elif value == "ext_segment_id":
            self.segments.index = self.segments.ext_segment_id
            self.files.index = self.files.file_id
        elif value == "series_id":
            self.segments.index = self.segments.file_id
            self.files.index = self.files.series_id

    @property
    def file_id(self):
        return np.asarray(self.segments["file_id"])

    @property
    def segment_id(self):
        return np.asarray(self.segments["segment_id"])

    @property
    def ext_segment_id(self):
        return np.asarray(self.segments["ext_segment_id"])

    @property
    def segment_names(self):
        return np.asarray(
            pd.merge(
                self.segments, self.ext_segments, on="ext_segment_id", how="inner"
            )["name"]
        )

    @property
    def segment_names_index(self):
        _, index = np.unique(self.segment_names, return_inverse=True)
        return index

    @property
    def segment_score(self):
        return np.asarray(
            pd.merge(
                self.segments, self.ext_segments, on="ext_segment_id", how="inner"
            )["score"]
        )

    @property
    def uniq_segment_id(self):
        return np.asarray(self.ext_segments["ext_segment_id"])

    @property
    def series_id(self):
        return np.asarray(self.files["series_id"])

    @property
    def uniq_file_id(self):
        return np.asarry(self.files["file_id"])
        # if self._uniq_file_id is None:
        #     self._uniq_file_id = np.asarray(self.segments['file_id'].unique())

        # return self._uniq_file_id

    @property
    def uniq_series_id(self):
        if self._uniq_series_id is None:
            self._uniq_series_id = np.asarray(self.ext_segments["series_id"].unique())

        return self._uniq_series_id

    @property
    def num_ext_segments(self):
        return len(self.ext_segments)

    @property
    def tbeg(self):
        return np.asarray(self.segments["tbeg"])

    @property
    def tend(self):
        return np.asarray(self.segments["tend"])

    def copy(self):
        """Makes a copy of the object."""
        return deepcopy(self)

    def segment_ids_from_file(self, file_id):
        """Returns segments_ids corresponding to a given file_id"""
        if self.index_column == "file_id":
            return np.asarray(self.segments.loc[file_id]["segment_id"])
        index = self.segments["file_id"] == file_id
        return np.asarray(self.segments.loc[index]["segment_id"])

    def ext_segment_ids_from_file(self, file_id):
        """Returns ext_segments_ids corresponding to a given file_id"""
        if self.index_column == "file_id":
            return np.unique(np.asarray(self.segments.loc[file_id]["ext_segment_id"]))
        index = self.segments["file_id"] == file_id
        return np.unique(np.asarray(self.segments.loc[index]["ext_segment_id"]))

    def __iter__(self):
        self.iter_idx = 0
        return self

    def __next__(self):
        if self.index_column == "file_id":
            if self.iter_idx < len(self.uniq_file_id):
                r = self.__getitem__(self.uniq_file_id[self.iter_idx])
            else:
                raise StopIteration()
        elif self.index_column == "series_id":
            if self.iter_idx < len(self.uniq_series_id):
                r = self.__getitem__(self.uniq_series_id[self.iter_idx])
            else:
                raise StopIteration()
        elif self.index_column == "ext_segment_id":
            if self.iter_idx < len(self.ext_segments):
                r = self.__getitem__(
                    self.ext_segment["ext_segment_id"].iloc[self.iter_idx]
                )
            else:
                raise StopIteration()
        else:
            if self.iter_idx < len(self.segments):
                r = self.__getitem__(self.segments["segment_id"].iloc(self.iter_idx))
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

    def __getitem__(self, key):
        """It allows to acces the de segments by file_id or segment
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
        if self.index_column == "segment_id":
            return pd.merge(
                self.segments.loc[key], self.ext_segments, sort=False, how="inner"
            )
        else:
            return self.filter([key])

    def save(self, file_path, sep=" "):
        """Saves segments to text file.

        Args:
          file_path: File to write the list.
          sep: Separator between the fields
        """
        self.segments[
            ["segment_id", "file_id", "tbeg", "tend", "ext_segment_id"]
        ].to_csv(
            file_path + ".segments",
            sep=sep,
            float_format="%.3f",
            index=False,
            header=False,
        )
        self.ext_segments[["ext_segment_id", "name", "score"]].to_csv(
            file_path + ".ext_segments",
            sep=sep,
            float_format="%.3f",
            index=False,
            header=False,
            na_rep="NA",
        )
        self.files[["file_id", "series_id"]].to_csv(
            file_path + ".files",
            sep=sep,
            float_format="%.3f",
            index=False,
            header=False,
        )

    @classmethod
    def load(cls, file_path, sep=" ", index_column="file_id"):
        """Loads script list from text file.

        Args:
          file_path: File to read the list.
          sep: Separator between the key and file_path in the text file.

        Returns:
          SegmentList object.
        """
        segments = pd.read_csv(
            file_path + ".segments",
            sep=sep,
            header=None,
            names=["segment_id", "file_id", "tbeg", "tend", "ext_segment_id"],
        )
        if path.isfile(file_path + ".ext_segments"):
            ext_segments = pd.read_csv(
                file_path + ".ext_segments",
                sep=sep,
                header=None,
                names=["ext_segment_id", "name", "score"],
                na_values="NA",
            )
        else:
            ext_segments = None

        if path.isfile(file_path + ".files"):
            files = pd.read_csv(
                file_path + ".files",
                sep=sep,
                header=None,
                names=["file_id", "series_id"],
            )
        else:
            files = None

        return cls(segments, ext_segments, files, index_column)

    def filter(self, filter_key, keep=True):
        if self.index_column == "series_id":
            if not keep:
                filter_key = np.setdiff1d(np.asarray(self.files.index), filter_key)
            files = self.files.loc[filter_key]
            segments = pd.merge(self.segments, files, on="file_id", how="inner")[
                ["segment_id", "file_id", "tbeg", "tend", "ext_segment_id"]
            ]
        else:
            if not keep:
                filter_key = np.setdiff1d(np.asarray(self.segments.index), filter_key)
            segments = self.segments.loc[filter_key]
            files = pd.merge(self.files, segments, on="file_id", how="inner")[
                ["file_id", "series_id"]
            ]

        ext_segments = pd.merge(
            self.ext_segments, segments, on="ext_segment_id", how="inner"
        )[["ext_segment_id", "name"]]

        return ExtSegmentList(segments, ext_segments, files, self.index_column)

    def split(self, idx, num_parts):
        if self.index_column == "file_id":
            key, _ = split_list(self.uniq_file_id, idx, num_parts)
        elif self.index_column == "series_id":
            key, _ = split_list(self.uniq_series_id, idx, num_parts)
        elif self.index_column == "segment_id":
            key, _ = split_list(self.segment_id, idx, num_parts)
        elif self.index_column == "ext_segment_id":
            key, _ = split_list(self.uniq_ext_segment_id, idx, num_parts)

        return self.filter(key)

    @classmethod
    def merge(cls, segment_lists, index_column="file_id"):
        segments = []
        files = []
        ext_segments = []
        for sl in segment_lists:
            segments.append(sl.segments)
            files.append(sl.files)
            ext_segments.append(ext_segments)

        segments = pd.concat(segments).drop_duplicates()
        files = pd.concat(files).drop_duplicates()
        ext_segments = pd.concat(ext_segments).drop_duplicates()

        return cls(segments, ext_segments, files, index_column)

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

    def merge_adjacent_segments_old(self, max_segments=0):
        if max_segments == 0:
            max_segments = len(self.segments)

        segm = pd.merge(
            self.segments, self.ext_segments, on="ext_segment_id", how="inner"
        )
        segm_1 = segm.shift(1)

        index = (segm.file_id == segm_1.file_id) & (segm.name == segm_1.name)

        merging = False
        count = 1
        d = {}
        for i in range(len(self.segments)):
            if index.iloc[i]:
                merging = True
                if count == 1:
                    first_idx = i - 1
                last_idx = i
                count += 1

            if (
                not index.iloc[i]
                or i == len(self.segments) - 1
                or count == max_segments
            ) and merging:
                if count == max_segments and i < len(self.segments) - 1:
                    # logging.debug(index)
                    index.iloc[i + 1] = False
                    # logging.debug(index)
                r = self.copy()
                count = 1
                merging = False
                first_segment = self.segments.iloc[first_idx].segment_id
                last_segment = self.segments.iloc[last_idx].segment_id
                new_ext_segment_id = self.segments[
                    first_idx : last_idx + 1
                ].segment_id.str.cat(sep="@")
                old_ext_segment_ids = np.array(
                    self.segments[first_idx : last_idx + 1].ext_segment_id.unique()
                )

                if (
                    len(old_ext_segment_ids) == 1
                    and old_ext_segment_ids[0] == new_ext_segment_id
                ):
                    continue

                kkk = self.ext_segments.ext_segment_id == new_ext_segment_id
                if np.sum(kkk) > 0:
                    logging.debug(first_segment)
                    logging.debug(last_segment)
                    logging.debug(new_ext_segment_id)
                    r.save("rrrr")
                    self.save("pppp")

                self.segments.iloc[
                    first_idx : last_idx + 1,
                    self.segments.columns.get_loc("ext_segment_id"),
                ] = new_ext_segment_id
                # logging.debug(old_ext_segment_ids)
                # logging.debug('A',self.ext_segments.ext_segment_id)
                # logging.debug(old_ext_segment_ids.iloc[0])
                # logging.debug(new_ext_segment_id)
                d[old_ext_segment_ids[0]] = new_ext_segment_id
                # logging.debug(old_ext_segment_ids[1:])
                # logging.debug(old_ext_segment_ids[1:])
                # logging.debug(self.ext_segments)
                # self.ext_segments.drop(old_ext_segment_ids[1:], inplace=True)
                # for osid in old_ext_segment_ids[1:]:
                #     kk = self.segments.ext_segment_id == osid
                #     if np.sum(kk) > 0:
                #         logging.debug(first_segment)
                #         logging.debug(last_segment)
                #         logging.debug(self.segments[kk])
                #         logging.debug(new_ext_segment_id)
                #         raise Exception()
                # logging.debug('C',self.ext_segments)
                if len(self.ext_segments.ext_segment_id.unique()) != len(
                    self.ext_segments.ext_segment_id
                ):
                    logging.debug(first_segment)
                    logging.debug(last_segment)
                    logging.debug(new_ext_segment_id)
                    r.save("rrrr")
                    self.save("pppp")

        for k, v in d.items():
            self.ext_segments.loc[k, "ext_segment_id"] = v
        self.ext_segments.reset_index(drop=True, inplace=True)
        drop_index = ~self.ext_segments.ext_segment_id.isin(
            self.segments.ext_segment_id
        )
        drop_index = self.ext_segments.index[drop_index]
        self.ext_segments.drop(drop_index, inplace=True)
        self.ext_segments = self.ext_segments.set_index(
            self.ext_segments.ext_segment_id, drop=False
        )
        assert len(self.ext_segments.ext_segment_id.unique()) == len(
            self.ext_segments.ext_segment_id
        )
        # logging.debug('E',self.ext_segments)

    def merge_adjacent_segments(self, max_segments=0):
        if max_segments == 0:
            max_segments = len(self.segments)

        segm = pd.merge(
            self.segments, self.ext_segments, on="ext_segment_id", how="inner"
        )
        segm_1 = segm.shift(1)

        index = (segm.file_id == segm_1.file_id) & (segm.name == segm_1.name)

        count = 1
        first_idx = 0
        last_idx = 0
        d = OrderedDict()
        # logging.debug('MERGE')
        # logging.debug(self.ext_segments)
        for i in range(1, len(self.segments) + 1):
            if i == len(self.segments) or not index.iloc[i] or count == max_segments:
                # logging.debug(i,first_idx, last_idx)
                new_ext_segment_id = self.segments[
                    first_idx : last_idx + 1
                ].segment_id.str.cat(sep="@")
                old_ext_segment_ids = np.array(
                    self.segments[first_idx : last_idx + 1].ext_segment_id.unique()
                )

                self.segments.iloc[
                    first_idx : last_idx + 1,
                    self.segments.columns.get_loc("ext_segment_id"),
                ] = new_ext_segment_id
                # logging.debug(old_ext_segment_ids)
                # logging.debug('A',self.ext_segments.ext_segment_id)
                # logging.debug('OLD SEGMENTS')
                # logging.debug(old_ext_segment_ids)
                # logging.debug('NEW SEGMENTS')
                # logging.debug(new_ext_segment_id)
                # logging.debug('NEW SEGMENTS FULL')
                # logging.debug(self.segments[:last_idx+1])
                d[new_ext_segment_id] = self.ext_segments.loc[
                    old_ext_segment_ids[0], "name"
                ]

                # logging.debug(old_ext_segment_ids[1:])
                # logging.debug(old_ext_segment_ids[1:])
                # logging.debug(self.ext_segments)
                # self.ext_segments.drop(old_ext_segment_ids[1:], inplace=True)
                # for osid in old_ext_segment_ids[1:]:
                #     kk = self.segments.ext_segment_id == osid
                #     if np.sum(kk) > 0:
                #         logging.debug(first_segment)
                #         logging.debug(last_segment)
                #         logging.debug(self.segments[kk])
                #         logging.debug(new_ext_segment_id)
                #         raise Exception()
                # logging.debug('C',self.ext_segments)
                # if len(self.ext_segments.ext_segment_id.unique()) != len(self.ext_segments.ext_segment_id):
                #     logging.debug(first_segment)
                #     logging.debug(last_segment)
                #     logging.debug(new_ext_segment_id)
                #     r.save('rrrr')
                #     self.save('pppp')

                count = 1
                first_idx = last_idx + 1
                last_idx = first_idx
            else:
                count += 1
                last_idx = i

        ext_segment_id = [k for k, v in d.items()]
        name = [v for k, v in d.items()]
        self.ext_segments = pd.DataFrame(
            {"ext_segment_id": ext_segment_id, "name": name}
        )
        # logging.debug('DICT', d)
        # for k,v in d.items():
        #     logging.debug(k)
        #     logging.debug(v)
        #     logging.debug(self.ext_segments[k])
        #     self.ext_segments.loc[k,'ext_segment_id'] = v
        # self.ext_segments.reset_index(drop=True, inplace=True)
        # drop_index = (~self.ext_segments.ext_segment_id.isin(self.segments.ext_segment_id))
        # drop_index = self.ext_segments.index[drop_index]
        # self.ext_segments.drop(drop_index, inplace=True)
        self.ext_segments = self.ext_segments.set_index(
            self.ext_segments.ext_segment_id, drop=False
        )
        # logging.debug(self.ext_segments)
        # assert len(self.ext_segments.ext_segment_id.unique()) == len(self.ext_segments.ext_segment_id)
        # #logging.debug('E',self.ext_segments)

    def assign_names(self, ext_segments_ids, names, scores=None):
        assert len(names) == len(ext_segments_ids)
        if scores is not None:
            assert len(scores) == len(ext_segments_ids)
        self.ext_segments.loc[ext_segments_ids, "name"] = names
        self.ext_segments.loc[ext_segments_ids, "score"] = scores

    def get_ext_segment_index(self):
        d = {s: i for i, s in enumerate(self.ext_segments.ext_segment_id)}
        index = np.array([d[s] for s in self.segments.ext_segment_id], dtype=int)
        return index
