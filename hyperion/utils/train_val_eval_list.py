"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import os.path as path
import logging
from collections import OrderedDict
from copy import deepcopy

import numpy as np

from .list_utils import *


class TrainValEvalList(object):
    """Class to split dataset into train, validation and test.

    Attributes:
      key: String List with the names of the dataset/recording/i-vector
      folds: Int numpy array with the number of the fold of each key.
      mask: Boolean numpy array to mask elements in the key
    """

    def __init__(self, key, part, part_names=None, mask=None):
        self.part = part
        self.key = key
        self.part_names = part_names
        self.mask = mask
        self._part2num = None
        self.validate()

    def validate(self):
        """Validates the class attributes attributes"""
        self.key = list2ndarray(self.key)
        self.part = list2ndarray(self.part)
        if self.part.dtype != int:
            self.part = self.part.astype(int)
        assert len(self.key) == len(self.part)
        assert len(np.unique(self.part[self.part >= 0])) == np.max(self.part) + 1
        if self.mask is not None:
            assert len(self.mask) == len(self.part)

    def _make_part2num(self):
        if self._part2num is not None:
            return
        assert self.part_names is not None
        self._part2num = {p: k for k, p in enumerate(self.part_names)}

    def copy(self):
        """Returns a copy of the object."""
        return deepcopy(self)

    def __len__(self):
        """Returns number of parts."""
        return self.num_parts()

    def num_parts(self):
        """Returns number of parts."""
        return np.max(self.part) + 1

    def align_with_key(self, key, raise_missing=True):
        """Aligns the part list with a given key

        Args:
          key: Key to align the part and key variables of the object.
          raise_missing: if True, raises exception when an element of key is
                          not found in the object.
        """
        f, idx = ismember(key, self.key)
        if np.all(f):
            self.key = self.key[idx]
            self.part = self.part[idx]
            if self.mask is not None:
                self.mask = self.mask[idx]
        else:
            for i in (f == 0).nonzero()[0]:
                logging.warning("segment %s not found" % key[i])
            if raise_missing:
                raise Exception("some scores were not computed")

    def get_part_idx(self, part):
        """Returns a part boolean indices

        Args:
          part: Part number or name to return

        Returns:
          train_idx: Indices of the elements used for training
          test_idx: Indices of the elements used for test
        """
        if isinstance(part, str):
            self._make_part2num()
            part = self._part2num[part]

        idx = self.part == part
        if self.mask is not None:
            idx = np.logical_and(idx, self.mask)
        return idx

    def get_part(self, part):
        """Returns a part keys

        Args:
          part: Part number to return

        Returns:
          train_key: Keys of the elements used for training
          test_key: Keys of the elements used for test
        """

        train_idx, test_idx = self.get_part_idx(part)
        return self.key[train_idx], self.key[test_idx]

    def __getitem__(self, part):
        """Returns a part keys

        Args:
          part: Part number to return

        Returns:
          train_key: Keys of the elements used for training
          test_key: Keys of the elements used for test
        """

        return self.get_part(part)

    def save(self, file_path, sep=" "):
        """Saves object to txt file

        Args:
          file_path: File path
          sep: Separator between part field and key field
        """
        with open(file_path, "w") as f:
            for p, k in zip(self.part, self.key):
                if self.part_names is None:
                    f.write("%s%s%d\n" % (k, sep, p))
                else:
                    f.write("%s%s%d%s\n" % (k, sep, p, self.part_names[p]))

    @classmethod
    def load(cls, file_path, sep=" "):
        """Loads object from txt file

        Args:
          file_path: File path
          sep: Separator between part field and key field

        Returns:
          PartList object
        """

        with open(file_path, "r") as f:
            fields = [line.rstrip().split(sep=sep, maxsplit=2) for line in f]
        key = np.asarray([f[0] for f in fields])
        part = np.asarray([int(f[1]) for f in fields], dtype=int)
        if len(fields[0]) == 2:
            part_names = None
        else:
            part_names = np.asarray([f[2] for f in fields], dtype=int)
            _, part_idx = np.unique(part, return_index=True)
            part_names = part_names[part_idx]

        return cls(key, part, part_names=part_names)

    @classmethod
    def create(
        cls,
        segment_key,
        part_proportions,
        part_names=None,
        balance_by_key=None,
        group_by_key=None,
        mask=None,
        shuffle=True,
        seed=1024,
    ):
        """Creates a PartList object.

        Args:
          segment_key: String List of recordings/speech segments
          part_proportions: % of data assigned to each part.
                            We can do as many parts as we want, not only 3.
                            Vector of dimension num_parts - 1, the last part is assumed to be the rest of the data.
          part_names: Names of the parts, by default ['train', 'val', 'eval'].
          balance_by_key: String List of keys indicating a property of the segment to make all parts to
             have the same number of elements of each class. E.g. for language ID this would be the language
             of the recording.
          group_by_key: String List of keys indicating a property of the segment to make all the elements
             of the same class to be in the same part. E.g. for language ID this would be the speaker ID
             of the recording.
          mask: Boolean numpy array to mask elements of segment_key out.
          shuffle: Shuffles the segment list so that parts are not grouped in alphabetical order.
          seed : Seed for shuffling
        Returns:
          PartList object.
        """

        num_parts = len(part_proportions) + 1
        cum_prop = np.hstack(([0], np.cumsum(part_proportions), [1]))

        if part_names is None:
            if num_parts == 3:
                part_names = ["train", "val", "eval"]
            elif num_parts == 2:
                part_names = ["train", "eval"]

        if shuffle:
            rng = np.random.RandomState(seed=seed)

        if group_by_key is None:
            group_by_key = segment_key

        if balance_by_key is None:
            balance_by_key = np.zeros((len(segment_key),), dtype=int)
        else:
            _, balance_by_key = np.unique(balance_by_key, return_inverse=True)

        if mask is not None:
            balance_by_key[mask == False] = -1

        parts = -np.ones((len(segment_key),), dtype=int)

        num_classes = np.max(balance_by_key) + 1
        for i in range(num_classes):

            idx_i = (balance_by_key == i).nonzero()[0]
            group_key_i = group_by_key[idx_i]
            _, group_key_i = np.unique(group_key_i, return_inverse=True)
            num_groups_i = np.max(group_key_i) + 1

            if shuffle:
                shuffle_idx = np.arange(num_groups_i)
                rng.shuffle(shuffle_idx)
                group_key_tmp = np.zeros_like(group_key_i)
                for j in range(num_groups_i):
                    group_key_tmp[group_key_i == j] = shuffle_idx[j]
                group_key_i = group_key_tmp

            for j in range(num_parts):
                k1 = int(np.round(cum_prop[j] * num_groups_i))
                k2 = int(np.round(cum_prop[j + 1] * num_groups_i))
                idx_ij = np.logical_and(group_key_i >= k1, group_key_i < k2)
                idx_part = idx_i[idx_ij]
                parts[idx_part] = j

        if mask is None:
            assert np.all(parts >= 0)
        else:
            assert np.all(parts[mask] >= 0)
        return cls(segment_key, parts, part_names, mask)
