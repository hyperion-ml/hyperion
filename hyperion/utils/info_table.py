"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from pathlib import Path
import logging
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd

from .list_utils import split_list, split_list_group_by_key


class InfoTable(object):
    """This is a base class to store information about recordings, segments,
    features, etc.

    Attributes:
      df: pandas dataframe.
    """

    def __init__(self, df):
        self.df = df
        assert "id" in df, f"info_table={df}"
        self.df.set_index("id", drop=False, inplace=True)

    def copy(self):
        """Makes a copy of the object."""
        return deepcopy(self)

    def clone(self):
        """Makes a copy of the object."""
        return deepcopy(self)

    @property
    def __len__(self):
        return self.df.__len__

    @property
    def __str__(self):
        return self.df.__str__

    @property
    def __repr__(self):
        return self.df.__repr__

    @property
    def iat(self):
        return self.df.iat

    @property
    def at(self):
        return self.df.at

    @property
    def iloc(self):
        return self.df.iloc

    @property
    def loc(self):
        return self.df.loc

    @property
    def __getitem__(self):
        return self.df.__getitem__

    @property
    def __setitem__(self):
        return self.df.__setitem__

    @property
    def __contains__(self):
        return self.df.__contains__

    @property
    def index(self):
        return self.df.index

    def save(self, file_path, sep=None):
        """Saves info table to file

        Args:
          file_path: File to write the list.
          sep: Separator between the key and file_path in the text file.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        ext = file_path.suffix
        if ext in ["", ".scp"]:
            # if no extension we save as kaldi utt2spk file
            self.df.to_csv(file_path, sep=" ", header=False, index=False)
            return

        if sep is None:
            sep = "\t" if ".tsv" in ext else ","

        self.df.to_csv(file_path, sep=sep, index=False)

    @classmethod
    def load(cls, file_path, sep=None):
        """Loads utt2info list from text file.

        Args:
          file_path: File to read the list.
          sep: Separator between the key and file_path in the text file.
          dtype: Dictionary with the dtypes of each column.
        Returns:
          Utt2Info object
        """
        file_path = Path(file_path)
        ext = file_path.suffix
        if ext in ["", ".scp"]:
            # if no extension we load as kaldi utt2spk file
            df = pd.read_csv(
                file_path,
                sep=" ",
                header=None,
                names=["id", "class_id"],
                dtype={"id": np.str, "class_id": np.str},
            )
        else:
            if sep is None:
                sep = "\t" if ".tsv" in ext else ","

            df = pd.read_csv(file_path, sep=sep)

        return cls(df)

    def sort(self, column="id", ascending=True):
        """Sorts the table by column"""
        self.df.sort_values(by=column, inplace=True, ascending=ascending)

    def split(self, idx, num_parts, group_by=None):
        """Splits SCPList into num_parts and return part idx.

        Args:
          idx: Part to return from 1 to num_parts.
          num_parts: Number of parts to split the list.
          group_by_field: All the lines with the same value in column
                          groub_by_field go to the same part

        Returns:
          Sub Utt2Info object
        """
        if group_by is None:
            _, idx1 = split_list(self.df["id"], idx, num_parts)
        else:
            _, idx1 = split_list_group_by_key(self.df[group_by], idx, num_parts)

        df = self.df.iloc[idx1]
        return self.__class__(df)

    @classmethod
    def merge(cls, tables):
        """Merges several Utt2Info tables.

        Args:
          info_lists: List of Utt2Info

        Returns:
          Utt2Info object concatenation the info_lists.
        """
        df_list = [table.df for table in tables]
        df = pd.concat(df_list)
        return cls(df)

    def filter(self, items=None, iindex=None, columns=None, by="id", keep=True):
        assert (
            items is None or iindex is None
        ), "items and iindex cannot be not None at the same time"
        df = self.df

        if not keep:
            if items is not None:
                items = np.setdiff1d(df[by], items)
            elif iindex is not None:
                iindex = np.setdiff1d(np.arange(len(df)), iindex)

            if columns is not None:
                columns = np.setdiff1d(df.columns, columns)

        if items is not None:
            if by != "id":
                missing = [False if v in df[by] else True for v in items]
                if any(missing):
                    raise Exception(f"{items[missing]} not found in table")
                items = [True if v in items else False for v in df[by]]

            if columns is None:
                df = df.loc[items]
            else:
                df = df.loc[items, columns]
        else:
            if iindex is not None:
                df = self.df.iloc[iindex]

            if columns is not None:
                df = df[columns]

        return self.__class__(df)

    def __eq__(self, other):
        """Equal operator"""
        if self.df.shape[0] == 0 and other.df.shape[0] == 0:
            return True
        eq = self.df.equals(other.df)
        return eq

    def __ne__(self, other):
        """Non-equal operator"""
        return not self.__eq__(other)

    def __cmp__(self, other):
        """Comparison operator"""
        if self.__eq__(other):
            return 0
        return 1

    # def __len__(self):
    #     """Returns the number of elements in the list."""
    #     return len(self.df)

    # def _create_dict(self):
    #     """Creates dictionary that returns the position of
    #     a segment in the list.
    #     """
    #     self.key_to_index = OrderedDict(
    #         (k, i) for i, k in enumerate(self.utt_info.index)
    #     )

    # def get_index(self, key):
    #     """Returns the position of key in the list."""
    #     if self.key_to_index is None:
    #         self._create_dict()
    #     return self.key_to_index[key]

    # def __contains__(self, id):
    #     """Returns True if the list contains the key"""
    #     return id in self.df.index

    # def __getitem__(self, id):
    #     """It allows to acces the data in the list by key or index like in
    #        a ditionary, e.g.:
    #        If input is a string key:
    #            utt2spk = Utt2Info(info)
    #            spk_id = utt2spk['data1']
    #        If input is an index:
    #            key, spk_id  = utt2spk[0]

    #     Args:
    #       key: String key or integer index.
    #     Returns:
    #       If key is a string:
    #           info corresponding to key
    #       If key is the index in the key list:
    #           key, info given index
    #     """
    #     if isinstance(id, str):
    #         row = np.array(self.utt_info.loc[key])[1:]
    #         if len(row) == 1:
    #             return row[0]
    #         else:
    #             return row
    #     else:
    #         row = np.array(self.utt_info.iloc[key])
    #         if len(row) == 2:
    #             return row[0], row[1]
    #         else:
    #             return row[0], row[1:]

    # def sort(self, field=0):
    #     """Sorts the list by key"""
    #     if field == 0:
    #         self.utt_info.sort_index(ascending=True, inplace=True)
    #     else:
    #         idx = np.argsort(self.utt_info[field])
    #         self.utt_info = self.utt_info.iloc[idx]
    #     self.key_to_index = None

    # @classmethod
    # def load(cls, file_path, sep=" ", dtype={0: np.str, 1: np.str}):
    #     """Loads utt2info list from text file.

    #     Args:
    #       file_path: File to read the list.
    #       sep: Separator between the key and file_path in the text file.
    #       dtype: Dictionary with the dtypes of each column.
    #     Returns:
    #       Utt2Info object
    #     """
    #     df = pd.read_csv(file_path, sep=sep, header=None, dtype=dtype)
    #     df = df.rename(index=str, columns={0: "key"})
    #     return cls(df)

    # def split(self, idx, num_parts, group_by_field=0):
    #     """Splits SCPList into num_parts and return part idx.

    #     Args:
    #       idx: Part to return from 1 to num_parts.
    #       num_parts: Number of parts to split the list.
    #       group_by_field: All the lines with the same value in column
    #                       groub_by_field go to the same part

    #     Returns:
    #       Sub Utt2Info object
    #     """
    #     if group_by_field == 0:
    #         key, idx1 = split_list(self.utt_info["key"], idx, num_parts)
    #     else:
    #         key, idx1 = split_list_group_by_key(
    #             self.utt_info[group_by_field], idx, num_parts
    #         )

    #     utt_info = self.utt_info.iloc[idx1]
    #     return Utt2Info(utt_info)

    # def filter(self, filter_key, keep=True):
    #     """Removes elements from Utt2Info object by key

    #     Args:
    #       filter_key: List with the keys of the elements to keep or remove.
    #       keep: If True, we keep the elements in filter_key;
    #             if False, we remove the elements in filter_key;

    #     Returns:
    #       Utt2Info object.
    #     """
    #     if not keep:
    #         filter_key = np.setdiff1d(self.utt_info["key"], filter_key)
    #     utt_info = self.utt_info.loc[filter_key]
    #     return Utt2Info(utt_info)

    # def filter_info(self, filter_key, field=1, keep=True):
    #     """Removes elements of Utt2Info by info value

    #     Args:
    #       filter_key: List with the file_path of the elements to keep or remove.
    #       field: Field number corresponding to the info to filter
    #       keep: If True, we keep the elements in filter_key;
    #             if False, we remove the elements in filter_key;

    #     Returns:
    #       Utt2Info object.
    #     """
    #     if not keep:
    #         filter_key = np.setdiff1d(self.utt_info[field], filter_key)
    #     f, _ = ismember(filter_key, self.utt_info[field])
    #     if not np.all(f):
    #         for k in filter_key[f == False]:
    #             logging.error("info %s not found in field %d" % (k, field))
    #         raise Exception("not all keys were found in field %d" % (field))

    #     f, _ = ismember(self.utt_info[field], filter_key)
    #     utt_info = self.utt_info.iloc[f]
    #     return Utt2Info(utt_info)

    # def filter_index(self, index, keep=True):
    #     """Removes elements of Utt2Info by index

    #     Args:
    #       filter_key: List with the index of the elements to keep or remove.
    #       keep: If True, we keep the elements in filter_key;
    #             if False, we remove the elements in filter_key;

    #     Returns:
    #       Utt2Info object.
    #     """

    #     if not keep:
    #         index = np.setdiff1d(np.arange(len(self.key), dtype=np.int64), index)

    #     utt_info = self.utt_info.iloc[index]
    #     return Utt2Info(utt_info)

    def shuffle(self, seed=1024, rng=None):
        """Shuffles the elements of the list.

        Args:
          seed: Seed for random number generator.
          rng: numpy random number generator object.

        Returns:
          Index used to shuffle the list.
        """
        if rng is None:
            rng = np.random.RandomState(seed=seed)
        index = np.arange(len(self.df))
        rng.shuffle(index)
        self.df = self.df.iloc[index]
        return index

    def set_index(self, keys, inplace=True):
        if inplace:
            self.df.set_index(keys, drop=False, inplace=True)
            return

        df = self.df.set_index(keys, drop=False, inplace=False)
        return type(self)(df)

    def reset_index(self):
        self.df.set_index("id", drop=False, inplace=True)

    def get_loc(self, keys):
        if isinstance(keys, (list, np.ndarray)):
            return self.df.index.get_indexer(keys)

        loc = self.df.index.get_loc(keys)
        if isinstance(loc, int):
            return loc
        elif isinstance(loc, np.ndarray) and loc.dtype == np.bool:
            return np.nonzero(loc)[0]
        else:
            return list(range(loc.start, loc.stop, loc.step))

    def get_col_idx(self, keys):
        return self.df.columns.get_loc(keys)
