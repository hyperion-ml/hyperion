"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import os.path as path
import logging
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd

from .list_utils import *


class Utt2Info(object):
    """Class to manipulate utt2spk, utt2lang, etc. files.

    Attributes:
      key: segment key name.
      info:
      key_to_index: Dictionary that returns the position of a key in the list.
    """

    def __init__(self, utt_info):
        self.utt_info = utt_info
        self.validate()
        self.utt_info.index = self.utt_info.key
        self.key_to_index = None

    def validate(self):
        """Validates the attributes of the Utt2Info object."""
        assert "key" in self.utt_info.columns
        assert self.utt_info.shape[1] >= 2
        # assert self.utt_info['key'].nunique() == self.utt_info.shape[0]

    @classmethod
    def create(cls, key, info):
        key = np.asarray(key)
        info = np.asarray(info)
        if info.ndim == 2:
            data = np.hstack((key[:, None], info))
        else:
            data = np.vstack((key, info)).T
        num_columns = data.shape[1]
        columns = ["key"] + [i for i in range(1, num_columns)]
        utt_info = pd.DataFrame(data, columns=columns)
        return cls(utt_info)

    @property
    def num_info_fields(self):
        return self.utt_info.shape[1] - 1

    @property
    def key(self):
        return np.asarray(self.utt_info["key"])

    @property
    def info(self):
        if self.utt_info.shape[1] > 2:
            return np.asarray(self.utt_info.iloc[:, 1:])
        else:
            return np.asarray(self.utt_info[1])

    def copy(self):
        """Makes a copy of the object."""
        return deepcopy(self)

    def __len__(self):
        """Returns the number of elements in the list."""
        return len(self.utt_info)

    def len(self):
        """Returns the number of elements in the list."""
        return len(self.utt_info)

    def _create_dict(self):
        """Creates dictionary that returns the position of
        a segment in the list.
        """
        self.key_to_index = OrderedDict(
            (k, i) for i, k in enumerate(self.utt_info.index)
        )

    def get_index(self, key):
        """Returns the position of key in the list."""
        if self.key_to_index is None:
            self._create_dict()
        return self.key_to_index[key]

    def __contains__(self, key):
        """Returns True if the list contains the key"""
        return key in self.utt_info.index

    def __getitem__(self, key):
        """It allows to acces the data in the list by key or index like in
           a ditionary, e.g.:
           If input is a string key:
               utt2spk = Utt2Info(info)
               spk_id = utt2spk['data1']
           If input is an index:
               key, spk_id  = utt2spk[0]

        Args:
          key: String key or integer index.
        Returns:
          If key is a string:
              info corresponding to key
          If key is the index in the key list:
              key, info given index
        """
        if isinstance(key, str):
            row = np.array(self.utt_info.loc[key])[1:]
            if len(row) == 1:
                return row[0]
            else:
                return row
        else:
            row = np.array(self.utt_info.iloc[key])
            if len(row) == 2:
                return row[0], row[1]
            else:
                return row[0], row[1:]

    def sort(self, field=0):
        """Sorts the list by key"""
        if field == 0:
            self.utt_info.sort_index(ascending=True, inplace=True)
        else:
            idx = np.argsort(self.utt_info[field])
            self.utt_info = self.utt_info.iloc[idx]
        self.key_to_index = None

    def save(self, file_path, sep=" "):
        """Saves uttinfo to text file.

        Args:
          file_path: File to write the list.
          sep: Separator between the key and file_path in the text file.
        """
        self.utt_info.to_csv(file_path, sep=sep, header=False, index=False)

    @classmethod
    def load(cls, file_path, sep=" ", dtype={0: np.str, 1: np.str}):
        """Loads utt2info list from text file.

        Args:
          file_path: File to read the list.
          sep: Separator between the key and file_path in the text file.
          dtype: Dictionary with the dtypes of each column.
        Returns:
          Utt2Info object
        """
        df = pd.read_csv(file_path, sep=sep, header=None, dtype=dtype)
        df = df.rename(index=str, columns={0: "key"})
        return cls(df)

    def split(self, idx, num_parts, group_by_field=0):
        """Splits SCPList into num_parts and return part idx.

        Args:
          idx: Part to return from 1 to num_parts.
          num_parts: Number of parts to split the list.
          group_by_field: All the lines with the same value in column
                          groub_by_field go to the same part

        Returns:
          Sub Utt2Info object
        """
        if group_by_field == 0:
            key, idx1 = split_list(self.utt_info["key"], idx, num_parts)
        else:
            key, idx1 = split_list_group_by_key(
                self.utt_info[group_by_field], idx, num_parts
            )

        utt_info = self.utt_info.iloc[idx1]
        return Utt2Info(utt_info)

    @classmethod
    def merge(cls, info_lists):
        """Merges several Utt2Info tables.

        Args:
          info_lists: List of Utt2Info

        Returns:
          Utt2Info object concatenation the info_lists.
        """
        df_list = [u2i.utt_info for u2i in info_lists]
        utt_info = pd.concat(df_list)
        return cls(utt_info)

    def filter(self, filter_key, keep=True):
        """Removes elements from Utt2Info object by key

        Args:
          filter_key: List with the keys of the elements to keep or remove.
          keep: If True, we keep the elements in filter_key;
                if False, we remove the elements in filter_key;

        Returns:
          Utt2Info object.
        """
        if not keep:
            filter_key = np.setdiff1d(self.utt_info["key"], filter_key)
        utt_info = self.utt_info.loc[filter_key]
        return Utt2Info(utt_info)

    def filter_info(self, filter_key, field=1, keep=True):
        """Removes elements of Utt2Info by info value

        Args:
          filter_key: List with the file_path of the elements to keep or remove.
          field: Field number corresponding to the info to filter
          keep: If True, we keep the elements in filter_key;
                if False, we remove the elements in filter_key;

        Returns:
          Utt2Info object.
        """
        if not keep:
            filter_key = np.setdiff1d(self.utt_info[field], filter_key)
        f, _ = ismember(filter_key, self.utt_info[field])
        if not np.all(f):
            for k in filter_key[f == False]:
                logging.error("info %s not found in field %d" % (k, field))
            raise Exception("not all keys were found in field %d" % (field))

        f, _ = ismember(self.utt_info[field], filter_key)
        utt_info = self.utt_info.iloc[f]
        return Utt2Info(utt_info)

    def filter_index(self, index, keep=True):
        """Removes elements of Utt2Info by index

        Args:
          filter_key: List with the index of the elements to keep or remove.
          keep: If True, we keep the elements in filter_key;
                if False, we remove the elements in filter_key;

        Returns:
          Utt2Info object.
        """

        if not keep:
            index = np.setdiff1d(np.arange(len(self.key), dtype=np.int64), index)

        utt_info = self.utt_info.iloc[index]
        return Utt2Info(utt_info)

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
        index = np.arange(len(self.key))
        rng.shuffle(index)
        self.utt_info = self.utt_info.iloc[index]
        self.key_to_index = None
        return index

    def __eq__(self, other):
        """Equal operator"""
        if self.utt_info.shape[0] == 0 and other.utt_info.shape[0] == 0:
            return True
        eq = self.utt_info.equals(other.utt_info)
        return eq

    def __ne__(self, other):
        """Non-equal operator"""
        return not self.__eq__(other)

    def __cmp__(self, other):
        """Comparison operator"""
        if self.__eq__(other):
            return 0
        return 1
