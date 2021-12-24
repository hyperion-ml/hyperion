"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import os
import os.path as path
from collections import OrderedDict
from copy import deepcopy
import logging

import numpy as np

from .list_utils import *


class SCPList(object):
    """Class to manipulate script lists.

    Attributes:
      key: segment key name.
      file_path: path to the file on hard drive, wav, ark or hdf5 file.
      offset: Byte in Ark file where the data is located.
      range_spec: range of frames (rows) to read.
      key_to_index: Dictionary that returns the position of a key in the list.
    """

    def __init__(self, key, file_path, offset=None, range_spec=None):
        self.key = key
        self.file_path = file_path
        self.offset = offset
        self.range_spec = range_spec
        self.key_to_index = None
        self.validate()

    def validate(self):
        """Validates the attributes of the SCPList object."""
        self.key = list2ndarray(self.key)
        self.file_path = list2ndarray(self.file_path, dtype=np.object)
        assert len(self.key) == len(self.file_path)
        if self.offset is not None:
            if isinstance(self.offset, list):
                self.offset = np.array(self.offset, dtype=np.int64)
            assert len(self.key) == len(self.offset)
        if self.range_spec is not None:
            if len(self.range_spec) == 0:
                self.range_spec = None
            else:
                if isinstance(self.range_spec, list):
                    self.range_spec = np.array(self.offset, dtype=np.int64)
                assert len(self.key) == self.range_spec.shape[0]
                assert self.range_spec.shape[1] == 2

    def copy(self):
        """Makes a copy of the object."""
        return deepcopy(self)

    def __len__(self):
        """Returns the number of elements in the list."""
        return len(self.key)

    def len(self):
        """Returns the number of elements in the list."""
        return len(self.key)

    def _create_dict(self):
        """Creates dictionary that returns the position of
        a segment in the list.
        """
        self.key_to_index = OrderedDict((k, i) for i, k in enumerate(self.key))

    def get_index(self, key):
        """Returns the position of key in the list."""
        if self.key_to_index is None:
            self._create_dict()
        return self.key_to_index[key]

    def __contains__(self, key):
        """Returns True if the list contains the key"""
        if self.key_to_index is None:
            self._create_dict()
        return key in self.key_to_index

    def __getitem__(self, key):
        """It allows to acces the data in the list by key or index like in
           a ditionary, e.g.:
           If input is a string key:
               scp = SCPList(keys, file_paths, offsets, ranges)
               file_path, offset, range = scp['data1']
           If input is an index:
               key, file_path, offset, range = scp[0]

        Args:
          key: String key or integer index.
        Returns:
          If key is a string:
              file_path, offset and range_spec given the key.
          If key is the index in the key list:
              key, file_path, offset and range_spec given the index.
        """
        return_key = True
        if isinstance(key, str):
            return_key = False
            index = self.get_index(key)
        else:
            index = key
        offset = None if self.offset is None else self.offset[index]
        range_spec = None if self.range_spec is None else self.range_spec[index]
        if return_key:
            return self.key[index], self.file_path[index], offset, range_spec
        else:
            return self.file_path[index], offset, range_spec

    def add_prefix_to_filepath(self, prefix):
        """Adds a prefix to the file path"""
        self.file_path = np.array([prefix + p for p in self.file_path])

    def sort(self):
        """Sorts the list by key"""
        self.key, idx = sort(self.key, return_index=True)
        self.file_path = self.file_path[idx]
        if self.offset is not None:
            self.offset = self.offset[idx]
        if self.range_spec is not None:
            self.range_spec = self.range_spec[idx]
        self.key_to_index = None

    def save(self, file_path, sep=" ", offset_sep=":"):
        """Saves script list to text file.

        Args:
          file_path: File to write the list.
          sep: Separator between the key and file_path in the text file.
          offset_sep: Separator between file_path and offset.
        """
        if self.range_spec is None:
            range_spec = ["" for k in self.key]
        else:
            range_spec = []
            for r in self.range_spec:
                if r[0] == 0 and r[1] == 0:
                    range_spec.append("")
                elif r[1] == 0:
                    range_spec.append("[%d:]" % r[0])
                else:
                    range_spec.append("[%d:%d]" % (r[0], r[0] + r[1] - 1))

        with open(file_path, "w") as f:
            if self.offset is None:
                for k, p, r in zip(self.key, self.file_path, range_spec):
                    f.write("%s%s%s%s\n" % (k, sep, p, r))
            else:
                for k, p, o, r in zip(
                    self.key, self.file_path, self.offset, range_spec
                ):
                    f.write("%s%s%s%s%d%s\n" % (k, sep, p, offset_sep, o, r))

    @staticmethod
    def parse_script(script, offset_sep):
        """Parses the parts of the second field of the scp text file.

        Args:
          script: Second column of scp file.
          offset_sep: Separtor between file_path and offset.

        Returns:
          file_path, offset and range_spec.
        """
        file_range = [f.split(sep="[", maxsplit=1) for f in script]
        offset = None
        range_spec = None

        file_offset = [f[0].split(sep=offset_sep, maxsplit=1) for f in file_range]
        file_path = [f[0] for f in file_offset]

        if len(file_offset[0]) == 2:
            offset = [int(f[1]) if len(f) == 2 else -1 for f in file_offset]
            if -1 in offset:
                raise ValueError("Missing data position for %s" % f[0])

        do_range = False
        for f in file_range:
            if len(f) == 2:
                do_range = True
                break

        if do_range:
            range_spec1 = [
                f[1].rstrip("]").split(sep=":", maxsplit=1) if len(f) == 2 else None
                for f in file_range
            ]
            range_spec21 = [
                int(f[0]) if f is not None and f[0].isdecimal() else 0
                for f in range_spec1
            ]
            range_spec22 = [
                int(f[1]) if f is not None and f[1].isdecimal() else None
                for f in range_spec1
            ]
            range_spec = [
                [a, b - a + 1] if b is not None else [a, 0]
                for a, b in zip(range_spec21, range_spec22)
            ]
            range_spec = np.array(range_spec, dtype=np.int64)

        return file_path, offset, range_spec

    @classmethod
    def load(cls, file_path, sep=" ", offset_sep=":", is_wav=False):
        """Loads script list from text file.

        Args:
          file_path: File to read the list.
          sep: Separator between the key and file_path in the text file.
          offset_sep: Separator between file_path and offset.

        Returns:
          SCPList object.
        """
        with open(file_path, "r") as f:
            fields = [line.rstrip().split(sep=sep, maxsplit=1) for line in f]

        key = [f[0] for f in fields]
        script = [f[1] for f in fields]
        del fields
        if is_wav:
            file_path = script
            offset = None
            range_spec = None
        else:
            file_path, offset, range_spec = SCPList.parse_script(script, offset_sep)
        return cls(key, file_path, offset, range_spec)

    def split(self, idx, num_parts, group_by_key=True):
        """Splits SCPList into num_parts and return part idx.

        Args:
          idx: Part to return from 1 to num_parts.
          num_parts: Number of parts to split the list.
          group_by_key: If True, all the lines with the same key
                        go to the same part.

        Returns:
          Sub SCPList
        """
        if group_by_key:
            key, idx1 = split_list_group_by_key(self.key, idx, num_parts)
        else:
            key, idx1 = split_list(self.key, idx, num_parts)

        file_path = self.file_path[idx1]
        offset = None
        range_spec = None
        if self.offset is not None:
            offset = self.offset[idx1]
        if self.range_spec is not None:
            range_spec = self.range_spec[idx1]

        return SCPList(key, file_path, offset, range_spec)

    @classmethod
    def merge(cls, scp_lists):
        """Merges several SCPList.

        Args:
          scp_lists: List of SCPLists

        Returns:
          SCPList object concatenation the scp_lists.
        """
        key_list = [item.key for item in scp_lists]
        file_list = [item.file_path for item in scp_lists]
        offset_list = [item.offset for item in scp_lists]
        range_list = [item.range_spec for item in scp_lists]

        key = np.concatenate(tuple(key_list))
        file_path = np.concatenate(tuple(file_list))

        if offset_list[0] is None:
            offset = None
        else:
            offset = np.concatenate(tuple(offset_list))

        if range_list[0] is None:
            range_spec = None
        else:
            range_spec = np.concatenate(tuple(range_list))

        return cls(key, file_path, offset, range_spec)

    def filter(self, filter_key, keep=True):
        """Removes elements from SCPList ojbect by key

        Args:
          filter_key: List with the keys of the elements to keep or remove.
          keep: If True, we keep the elements in filter_key;
                if False, we remove the elements in filter_key;

        Returns:
          SCPList object.
        """
        if not keep:
            filter_key = np.setdiff1d(self.key, filter_key)

        f, _ = ismember(filter_key, self.key)
        assert np.all(f)
        f, _ = ismember(self.key, filter_key)
        key = self.key[f]
        file_path = self.file_path[f]

        offset = None
        range_spec = None
        if self.offset is not None:
            offset = self.offset[f]
        if self.range_spec is not None:
            range_spec = self.range_spec[f]

        return SCPList(key, file_path, offset, range_spec)

    def filter_paths(self, filter_key, keep=True):
        """Removes elements of SCPList by file_path

        Args:
          filter_key: List with the file_path of the elements to keep or remove.
          keep: If True, we keep the elements in filter_key;
                if False, we remove the elements in filter_key;

        Returns:
          SCPList object.
        """

        if not keep:
            filter_key = np.setdiff1d(self.file_path, filter_key)

        f, _ = ismember(filter_key, self.file_path)
        assert np.all(f)
        f, _ = ismember(self.file_path, filter_key)
        key = self.key[f]
        file_path = self.file_path[f]

        offset = None
        range_spec = None
        if self.offset is not None:
            offset = self.offset[f]
        if self.range_spec is not None:
            range_spec = self.range_spec[f]

        return SCPList(key, file_path, offset, range_spec)

    def filter_index(self, index, keep=True):
        """Removes elements of SCPList by index

        Args:
          filter_key: List with the index of the elements to keep or remove.
          keep: If True, we keep the elements in filter_key;
                if False, we remove the elements in filter_key;

        Returns:
          SCPList object.
        """

        if not keep:
            index = np.setdiff1d(np.arange(len(self.key), dtype=np.int64), index)

        key = self.key[index]
        file_path = self.file_path[index]
        offset = None
        range_spec = None
        if self.offset is not None:
            offset = self.offset[index]
        if self.range_spec is not None:
            range_spec = self.range_spec[index]

        return SCPList(key, file_path, offset, range_spec)

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

        self.key = self.key[index]
        self.file_path = self.file_path[index]
        if self.offset is not None:
            self.offset = self.offset[index]
        if self.range_spec is not None:
            self.range_spec = self.range_spec[index]

        self.key_to_index = None
        return index

    def __eq__(self, other):
        """Equal operator"""
        if self.key.size == 0 and other.key.size == 0:
            return True
        eq = self.key.shape == other.key.shape
        eq = eq and np.all(self.key == other.key)
        eq = eq and (self.file_path.shape == other.file_path.shape)
        eq = eq and np.all(self.file_path == other.file_path)

        if (
            self.offset is None
            and other.offset is not None
            or self.offset is not None
            and other.offset is None
        ):
            eq = False
        elif self.offset is not None and other.offset is not None:
            eq = eq and np.all(self.offset == other.offset)

        if (
            self.range_spec is None
            and other.range_spec is not None
            or self.range_spec is not None
            and other.range_spec is None
        ):
            eq = False
        elif self.range_spec is not None and other.range_spec is not None:
            eq = eq and np.all(self.range_spec == other.range_spec)

        return eq

    def __ne__(self, other):
        """Non-equal operator"""
        return not self.__eq__(other)

    def __cmp__(self, other):
        """Comparison operator"""
        if self.__eq__(other):
            return 0
        return 1
