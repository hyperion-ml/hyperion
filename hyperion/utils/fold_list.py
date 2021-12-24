"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

 Class to make/read/write k-fold x-validation lists
"""

import os.path as path
import logging
from collections import OrderedDict
from copy import deepcopy

import numpy as np

from .list_utils import *


class FoldList(object):
    """Class to contain folds for cross-validation.

    Attributes:
      key: String List with the names of the dataset/recording/i-vector
      folds: Int numpy array with the number of the fold of each key.
      mask: Boolean numpy array to mask elements in the key
    """

    def __init__(self, fold, key, mask=None):
        self.fold = fold
        self.key = key
        self.mask = mask
        self.validate()

    def validate(self):
        """Validates the class attributes attributes"""
        self.key = list2ndarray(self.key)
        self.fold = list2ndarray(self.fold)
        if self.fold.dtype != int:
            self.fold = self.fold.astype(int)
        assert len(self.key) == len(self.fold)
        assert len(np.unique(self.fold[self.fold >= 0])) == np.max(self.fold) + 1
        if self.mask is not None:
            assert len(self.mask) == len(self.fold)

    def copy(self):
        """Returns a copy of the object."""
        return deepcopy(self)

    def __len__(self):
        """Returns number of folds."""
        return self.num_folds()

    @property
    def num_folds(self):
        """Returns number of folds."""
        return np.max(self.fold) + 1

    def align_with_key(self, key, raise_missing=True):
        """Aligns the fold list with a given key

        Args:
          key: Key to align the fold and key variables of the object.
          raise_missing: if True, raises exception when an element of key is
                          not found in the object.
        """
        f, idx = ismember(key, self.key)
        if np.all(f):
            self.key = self.key[idx]
            self.fold = self.fold[idx]
            if self.mask is not None:
                self.mask = self.mask[idx]
        else:
            for i in (f == 0).nonzero()[0]:
                logging.warning("segment %s not found" % key[i])
            if raise_missing:
                raise Exception("some scores were not computed")

    def get_fold_idx(self, fold):
        """Returns a fold boolean indices

        Args:
          fold: Fold number to return

        Returns:
          train_idx: Indices of the elements used for training
          test_idx: Indices of the elements used for test
        """
        test_idx = self.fold == fold
        train_idx = np.logical_not(test_idx)
        if self.mask is not None:
            train_idx = np.logical_and(train_idx, self.mask)
            test_idx = np.logical_and(test_idx, self.mask)
        return train_idx, test_idx

    def get_fold(self, fold):
        """Returns a fold keys

        Args:
          fold: Fold number to return

        Returns:
          train_key: Keys of the elements used for training
          test_key: Keys of the elements used for test
        """

        train_idx, test_idx = self.get_fold_idx(fold)
        return self.key[train_idx], self.key[test_idx]

    def __getitem__(self, fold):
        """Returns a fold keys

        Args:
          fold: Fold number to return

        Returns:
          train_key: Keys of the elements used for training
          test_key: Keys of the elements used for test
        """

        return self.get_fold(fold)

    def save(self, file_path, sep=" "):
        """Saves object to txt file

        Args:
          file_path: File path
          sep: Separator between fold field and key field
        """
        with open(file_path, "w") as f:
            for f, k in zip(self.fold, self.key):
                f.write("%s%s%s\n" % (f, sep, k))

    @classmethod
    def load(cls, file_path, sep=" "):
        """Loads object from txt file

        Args:
          file_path: File path
          sep: Separator between fold field and key field

        Returns:
          FoldList object
        """

        with open(file_path, "r") as f:
            fields = [line.rstrip().split(sep=sep, maxsplit=1) for line in f]
        fold = np.asarray([int(f[0]) for f in fields], dtype=int)
        key = np.asarray([f[1] for f in fields])
        return cls(fold, key)

    @classmethod
    def create(
        cls,
        segment_key,
        num_folds,
        balance_by_key=None,
        group_by_key=None,
        mask=None,
        shuffle=False,
        seed=1024,
    ):
        """Creates a FoldList object.

        Args:
          segment_key: String List of recordings/speech segments
          num_folds: Number of folds that we want to obtain.
          balance_by_key: String List of keys indicating a property of the segment to make all folds to
             have the same number of elements of each class. E.g. for language ID this would be the language
             of the recording.
          group_by_key: String List of keys indicating a property of the segment to make all the elements
             of the same class to be in the same fold. E.g. for language ID this would be the speaker ID
             of the recording.
          mask: Boolean numpy array to mask elements of segment_key out.
          shuffle: Shuffles the segment list so that folds are not grouped in alphabetical order.
          seed : Seed for shuffling
        Returns:
          FoldList object.
        """
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

        folds = -np.ones((len(segment_key),), dtype=int)

        num_classes = np.max(balance_by_key) + 1
        for i in range(num_classes):

            idx_i = (balance_by_key == i).nonzero()[0]
            group_key_i = group_by_key[idx_i]
            _, group_key_i = np.unique(group_key_i, return_inverse=True)
            num_groups_i = np.max(group_key_i) + 1
            delta = float(num_groups_i) / num_folds

            if shuffle:
                shuffle_idx = np.arange(num_groups_i)
                rng.shuffle(shuffle_idx)
                group_key_tmp = np.zeros_like(group_key_i)
                for j in range(num_groups_i):
                    group_key_tmp[group_key_i == j] = shuffle_idx[j]
                group_key_i = group_key_tmp

            for j in range(num_folds):
                k1 = int(np.round(j * delta))
                k2 = int(np.round((j + 1) * delta))
                idx_ij = np.logical_and(group_key_i >= k1, group_key_i < k2)
                idx_fold = idx_i[idx_ij]
                folds[idx_fold] = j

        if mask is None:
            assert np.all(folds >= 0)
        else:
            assert np.all(folds[mask] >= 0)
        return cls(folds, segment_key, mask)
