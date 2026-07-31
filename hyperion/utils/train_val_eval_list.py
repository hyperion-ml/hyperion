"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from copy import deepcopy
from typing import Dict, List, Optional, Union

import numpy as np

from .list_utils import *
from .misc import PathLike


class TrainValEvalList:
    """Split a dataset into train/validation/eval (or arbitrary) parts.

    Attributes:
      key: Item identifiers.
      part: Integer part id assigned to each item.
      part_names: Optional names for parts indexed by part id.
      mask: Optional boolean mask selecting valid items.
    """

    def __init__(
        self,
        key: Union[List[str], np.ndarray],
        part: Union[List[int], np.ndarray],
        part_names: Optional[Union[List[str], np.ndarray]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        self.part = part
        self.key = key
        self.part_names = part_names
        self.mask = mask
        self._part2num: Optional[Dict[str, int]] = None
        self.validate()

    def validate(self) -> None:
        """Validate and normalize internal arrays."""
        self.key = list2ndarray(self.key)
        self.part = list2ndarray(self.part)
        if self.part.dtype != int:
            self.part = self.part.astype(int)
        assert len(self.key) == len(self.part)
        if len(self.part) > 0:
            assert len(np.unique(self.part[self.part >= 0])) == np.max(self.part) + 1
        if self.mask is not None:
            self.mask = list2ndarray(self.mask).astype(bool)
            assert len(self.mask) == len(self.part)
        if self.part_names is not None:
            self.part_names = list2ndarray(self.part_names)
            assert len(self.part_names) == self.num_parts()

    def _make_part2num(self) -> None:
        if self._part2num is not None:
            return
        assert self.part_names is not None
        self._part2num = {p: k for k, p in enumerate(self.part_names)}

    def copy(self) -> "TrainValEvalList":
        """Returns a copy of the object."""
        return deepcopy(self)

    def __len__(self) -> int:
        """Returns number of parts."""
        return self.num_parts()

    def num_parts(self) -> int:
        """Returns number of parts."""
        if len(self.part) == 0:
            return 0
        return int(np.max(self.part) + 1)

    def align_with_key(
        self, key: Union[List[str], np.ndarray], raise_missing: bool = True
    ) -> None:
        """Align this list to the order of an external key list.

        Args:
          key: Target key order.
          raise_missing: If True, raise when any key is not found.
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

    def get_part_idx(self, part: Union[int, str]) -> np.ndarray:
        """Return boolean indices selecting items for a given part.

        Args:
          part: Part number or part name.

        Returns:
          Boolean mask with selected items.
        """
        if isinstance(part, str):
            self._make_part2num()
            part = self._part2num[part]

        idx = self.part == part
        if self.mask is not None:
            idx = np.logical_and(idx, self.mask)
        return idx

    def get_part(self, part: Union[int, str]) -> np.ndarray:
        """Return item keys for a given part.

        Args:
          part: Part number or part name.

        Returns:
          Keys belonging to the selected part.
        """
        idx = self.get_part_idx(part)
        return self.key[idx]

    def __getitem__(self, part: Union[int, str]) -> np.ndarray:
        """Return item keys for a given part.

        Args:
          part: Part number or part name.

        Returns:
          Keys belonging to the selected part.
        """
        return self.get_part(part)

    def save(self, file_path: PathLike, sep: str = " ") -> None:
        """Save list to a text file.

        Args:
          file_path: Output text file.
          sep: Separator between fields.
        """
        with open(file_path, "w") as f:
            for p, k in zip(self.part, self.key):
                if self.part_names is None:
                    f.write("%s%s%d\n" % (k, sep, p))
                else:
                    f.write("%s%s%d%s\n" % (k, sep, p, self.part_names[p]))

    @classmethod
    def load(cls, file_path: PathLike, sep: str = " ") -> "TrainValEvalList":
        """Load list from a text file.

        Args:
          file_path: Input text file.
          sep: Separator between fields.

        Returns:
          Loaded TrainValEvalList object.
        """

        with open(file_path, "r") as f:
            fields = [line.rstrip().split(sep=sep, maxsplit=2) for line in f]
        key = np.asarray([f[0] for f in fields])
        part = np.asarray([int(f[1]) for f in fields], dtype=int)
        if len(fields[0]) == 2:
            part_names = None
        else:
            part_names = np.asarray([f[2] for f in fields], dtype=object)
            _, part_idx = np.unique(part, return_index=True)
            part_names = part_names[part_idx]

        return cls(key, part, part_names=part_names)

    @classmethod
    def create(
        cls,
        segment_key: Union[List[str], np.ndarray],
        part_proportions: Union[List[float], np.ndarray],
        part_names: Optional[Union[List[str], np.ndarray]] = None,
        balance_by_key: Optional[Union[List[str], np.ndarray]] = None,
        group_by_key: Optional[Union[List[str], np.ndarray]] = None,
        mask: Optional[np.ndarray] = None,
        shuffle: bool = True,
        seed: int = 1024,
    ) -> "TrainValEvalList":
        """Create a new partition assignment.

        Args:
          segment_key: Item identifiers to partition.
          part_proportions: Fractions for the first ``num_parts - 1`` parts.
            The last part receives the remaining samples.
          part_names: Optional part names. Defaults to
            ``['train', 'val', 'eval']`` for 3 parts or ``['train', 'eval']``
            for 2 parts.
          balance_by_key: Optional class key used to balance class counts across
            parts.
          group_by_key: Optional grouping key forcing all items in the same
            group to stay in one part.
          mask: Optional boolean mask to exclude items from assignment.
          shuffle: If True, shuffle groups before assignment.
          seed: Random seed used when ``shuffle=True``.

        Returns:
          TrainValEvalList object.
        """

        num_parts = len(part_proportions) + 1
        cum_prop = np.hstack(([0], np.cumsum(part_proportions), [1]))

        if part_names is None:
            if num_parts == 3:
                part_names = ["train", "val", "eval"]
            elif num_parts == 2:
                part_names = ["train", "eval"]

        if shuffle:
            rng = np.random.default_rng(seed=seed)

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
        return cls(segment_key, parts, part_names=part_names, mask=mask)
