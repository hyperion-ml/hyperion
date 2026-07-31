"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, Optional, Sequence, Tuple, Union, overload

import numpy as np
import pandas as pd

from .list_utils import *
from .misc import PathLike


class Utt2Info:
    """Class to manipulate utt2spk, utt2lang, etc. files.

    Attributes:
      key: Utterance keys.
      info: Info values associated to each key.
      key_to_index: Dictionary that returns the row position of each key.
    """

    def __init__(self, utt_info: pd.DataFrame) -> None:
        self.utt_info = utt_info
        self.validate()
        self.utt_info.index = self.utt_info.key
        self.key_to_index: Optional[Dict[str, int]] = None

    def validate(self) -> None:
        """Validates the attributes of the Utt2Info object."""
        assert "key" in self.utt_info.columns
        assert self.utt_info.shape[1] >= 2
        # assert self.utt_info['key'].nunique() == self.utt_info.shape[0]

    @classmethod
    def create(
        cls,
        key: Union[Sequence[str], np.ndarray],
        info: Union[Sequence[object], np.ndarray],
    ) -> "Utt2Info":
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
    def num_info_fields(self) -> int:
        return self.utt_info.shape[1] - 1

    @property
    def key(self) -> np.ndarray:
        return np.asarray(self.utt_info["key"])

    @property
    def info(self) -> np.ndarray:
        if self.utt_info.shape[1] > 2:
            return np.asarray(self.utt_info.iloc[:, 1:])
        else:
            return np.asarray(self.utt_info[1])

    def copy(self) -> "Utt2Info":
        """Makes a copy of the object."""
        return deepcopy(self)

    def __len__(self) -> int:
        """Returns the number of elements in the list."""
        return len(self.utt_info)

    def len(self) -> int:
        """Returns the number of elements in the list."""
        return len(self.utt_info)

    def _create_dict(self) -> None:
        """Creates dictionary that returns the position of
        a segment in the list.
        """
        self.key_to_index = OrderedDict(
            (k, i) for i, k in enumerate(self.utt_info.index)
        )

    def get_index(self, key: str) -> int:
        """Returns the position of key in the list."""
        if self.key_to_index is None:
            self._create_dict()
        return self.key_to_index[key]

    def __contains__(self, key: str) -> bool:
        """Returns ``True`` if the list contains ``key``."""
        return key in self.utt_info.index

    @overload
    def __getitem__(self, key: str) -> Union[object, np.ndarray]: ...

    @overload
    def __getitem__(
        self, key: Union[int, np.integer]
    ) -> Union[Tuple[str, object], Tuple[str, np.ndarray]]: ...

    def __getitem__(
        self, key: Union[str, int, np.integer]
    ) -> Union[object, np.ndarray, Tuple[str, object], Tuple[str, np.ndarray]]:
        """Return entry by key or index.

        Args:
          key: Utterance key or row index.

        Returns:
          If ``key`` is a string, returns the info value(s) for that key.
          If ``key`` is an integer, returns ``(key, info)`` for that row.
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

    def sort(self, field: Union[int, str] = 0) -> None:
        """Sort rows by key (field=0) or by a selected info field."""
        if field == 0:
            self.utt_info.sort_index(ascending=True, inplace=True)
        else:
            idx = np.argsort(self.utt_info[field])
            self.utt_info = self.utt_info.iloc[idx]
        self.key_to_index = None

    def save(self, file_path: PathLike, sep: str = " ") -> None:
        """Save utt2info table to text file.

        Args:
          file_path: Destination file path.
          sep: Field separator.
        """
        self.utt_info.to_csv(file_path, sep=sep, header=False, index=False)

    @classmethod
    def load(
        cls,
        file_path: PathLike,
        sep: str = " ",
        dtype: Optional[Dict[int, object]] = None,
    ) -> "Utt2Info":
        """Load an utt2info table from a text file.

        Args:
          file_path: File to read.
          sep: Field separator.
          dtype: Optional dictionary with pandas dtypes by column index.

        Returns:
          Loaded ``Utt2Info`` object.
        """
        if dtype is None:
            dtype = {0: np.str_, 1: np.str_}
        df = pd.read_csv(file_path, sep=sep, header=None, dtype=dtype)
        df = df.rename(index=str, columns={0: "key"})
        return cls(df)

    def split(
        self, idx: int, num_parts: int, group_by_field: Union[int, str] = 0
    ) -> "Utt2Info":
        """Split table into ``num_parts`` and return part ``idx``.

        Args:
          idx: Part to return from 1 to num_parts.
          num_parts: Number of parts to split the list.
          group_by_field: If non-zero, rows with the same value in this field
            are kept in the same split.

        Returns:
          Sub Utt2Info object.
        """
        if group_by_field == 0:
            _, idx1 = split_list(self.utt_info["key"], idx, num_parts)
        else:
            _, idx1 = split_list_group_by_key(
                self.utt_info[group_by_field], idx, num_parts
            )

        utt_info = self.utt_info.iloc[idx1]
        return Utt2Info(utt_info)

    @classmethod
    def merge(cls, info_lists: Sequence["Utt2Info"]) -> "Utt2Info":
        """Merge several ``Utt2Info`` tables.

        Args:
          info_lists: List of ``Utt2Info`` objects.

        Returns:
          Concatenated ``Utt2Info`` object.
        """
        df_list = [u2i.utt_info for u2i in info_lists]
        utt_info = pd.concat(df_list)
        return cls(utt_info)

    def filter(
        self, filter_key: Union[Sequence[str], np.ndarray], keep: bool = True
    ) -> "Utt2Info":
        """Filter rows by utterance key.

        Args:
          filter_key: Keys to keep or remove.
          keep: If ``True``, keep keys in ``filter_key``. If ``False``,
            remove keys in ``filter_key``.

        Returns:
          Filtered ``Utt2Info`` object.
        """
        if not keep:
            filter_key = np.setdiff1d(self.utt_info["key"], filter_key)
        utt_info = self.utt_info.loc[filter_key]
        return Utt2Info(utt_info)

    def filter_info(
        self,
        filter_key: Union[Sequence[object], np.ndarray],
        field: Union[int, str] = 1,
        keep: bool = True,
    ) -> "Utt2Info":
        """Filter rows by value in an info field.

        Args:
          filter_key: Info values to keep or remove.
          field: Column index or name to filter on.
          keep: If ``True``, keep values in ``filter_key``. If ``False``,
            remove values in ``filter_key``.

        Returns:
          Filtered ``Utt2Info`` object.
        """
        if not keep:
            filter_key = np.setdiff1d(self.utt_info[field], filter_key)
        f, _ = ismember(filter_key, self.utt_info[field])
        if not np.all(f):
            for k in filter_key[f == False]:
                logging.error("info %s not found in field %s" % (k, field))
            raise Exception("not all keys were found in field %s" % (field))

        f, _ = ismember(self.utt_info[field], filter_key)
        utt_info = self.utt_info.iloc[f]
        return Utt2Info(utt_info)

    def filter_index(
        self, index: Union[Sequence[int], np.ndarray], keep: bool = True
    ) -> "Utt2Info":
        """Filter rows by positional index.

        Args:
          index: Integer indices to keep or remove.
          keep: If ``True``, keep ``index``. If ``False``, remove ``index``.

        Returns:
          Filtered ``Utt2Info`` object.
        """

        if not keep:
            index = np.setdiff1d(np.arange(len(self.key), dtype=np.int64), index)

        utt_info = self.utt_info.iloc[index]
        return Utt2Info(utt_info)

    def shuffle(
        self, seed: int = 1024, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """Shuffles the elements of the list.

        Args:
          seed: Seed for random number generator.
          rng: numpy random number generator object.

        Returns:
          Index used to shuffle the list.
        """
        if rng is None:
            rng = np.random.default_rng(seed=seed)
        index = np.arange(len(self.key))
        rng.shuffle(index)
        self.utt_info = self.utt_info.iloc[index]
        self.key_to_index = None
        return index

    def __eq__(self, other: object) -> bool:
        """Return ``True`` when both tables are equal."""
        if not isinstance(other, Utt2Info):
            return False
        if self.utt_info.shape[0] == 0 and other.utt_info.shape[0] == 0:
            return True
        eq = self.utt_info.equals(other.utt_info)
        return eq

    def __ne__(self, other: object) -> bool:
        """Return ``True`` when tables are different."""
        return not self.__eq__(other)

    def __cmp__(self, other: object) -> int:
        """Compatibility comparison method."""
        if self.__eq__(other):
            return 0
        return 1
