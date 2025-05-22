"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import re
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype

from .list_utils import split_list, split_list_group_by_key
from .misc import PathLike

T = TypeVar("T", bound="InfoTable")


class _InfoTableIndexer:
    """
    Wrapper for DataFrame indexers (like .loc and .iloc) that ensures any
    DataFrame result is returned as an InfoTable (or subclass) instance.
    """

    def __init__(self, parent: T, indexer: Any):
        """
        Args:
            parent (InfoTable): The InfoTable instance that owns this indexer.
            indexer (Any): The pandas indexer (e.g., df.loc or df.iloc).
        """
        self.parent = parent
        self.indexer = indexer

    def __getitem__(self, key: Union[str, int, slice, list, Tuple]) -> Any:
        """
        Gets item(s) from the DataFrame indexer.

        Args:
            key: A key or index/slice used for accessing rows or columns.

        Returns:
            InfoTable if result is a DataFrame, otherwise the native pandas result.
        """
        result = self.indexer[key]
        if isinstance(result, pd.DataFrame) and "id" in result:
            # convert only if "id" is in the result otherwise return a regular dataframe
            # since wihtout "id" it is not a valid InfoTable
            return self.parent.__class__(result)
        return result

    def __setitem__(self, key: Union[str, int, slice, list, Tuple], value: Any) -> None:
        """
        Sets item(s) in the DataFrame via the indexer.

        Args:
            key: A key or index/slice to locate the value to update.
            value: The new value(s) to assign.
        """
        self.indexer[key] = value


class _InfoTableAtIndexer:
    """
    Wrapper for DataFrame .at and .iat accessors, which always return scalars
    and do not require wrapping.
    """

    def __init__(self, parent: T, indexer: Any):
        """
        Args:
            parent (InfoTable): The InfoTable instance that owns this indexer.
            indexer (Any): The pandas indexer (e.g., df.at or df.iat).
        """
        self.parent = parent
        self.indexer = indexer

    def __getitem__(self, key: Tuple[Any, Any]) -> Any:
        """
        Gets a single scalar value from the DataFrame.

        Args:
            key: A tuple of (row label/index, column label/index).

        Returns:
            The scalar value at the specified location.
        """

        return self.indexer[key]

    def __setitem__(self, key: Tuple[Any, Any], value: Any) -> None:
        """
        Sets a scalar value in the DataFrame.

        Args:
            key: A tuple of (row label/index, column label/index).
            value: The new value to assign.
        """
        self.indexer[key] = value


class InfoTable:
    """
    Base class for storing structured metadata in a tabular format.

    This class wraps a pandas DataFrame and adds helper methods for working
    with audio-visual dataset metadata such as recordings, segments,
    and features. Maintains a consistent interface for operations like filtering,
    merging, and indexing.

    Attributes:
        df (pd.DataFrame): The internal DataFrame storing the metadata.
    """

    def __init__(self, df: Union[pd.DataFrame, T]):
        """
        Initialize an InfoTable from a DataFrame or another InfoTable.

        Args:
            df (Union[pd.DataFrame, InfoTable]): Input data.
        """
        if isinstance(df, InfoTable):
            df = df.df

        assert "id" in df, f"info_table={df}"
        self.df = df
        self.fix_dtypes()
        self.df.set_index("id", drop=False, inplace=True)

    def fix_dtypes(self) -> None:
        """
        Ensure the 'id' column is of string type.
        """
        if infer_dtype(self.df.id) != "string":
            self.df["id"] = self.df["id"].astype(str)

    def convert_col_to_str(self, column: str) -> None:
        """
        Ensure a specific column is of string type.

        Args:
            column (str): Column name to convert.
        """
        if infer_dtype(self.df[column]) != "string":
            self.df[column] = self.df[column].astype(str)

    def copy(self) -> T:
        """
        Return a deep copy of the InfoTable.

        Returns:
            InfoTable: A deep copy.
        """
        return deepcopy(self)

    def clone(self) -> T:
        """
        Alias for copy().

        Returns:
            InfoTable: A deep copy.
        """
        return deepcopy(self)

    def __iter__(self):
        return iter(self.df)

    def __len__(self):
        return len(self.df)

    def __str__(self):
        return str(self.df)

    def __repr__(self):
        return repr(self.df)

    # @property
    # def __len__(self):
    #     return self.df.__len__

    # @property
    # def __str__(self):
    #     return self.df.__str__

    # @property
    # def __repr__(self):
    #     return self.df.__repr__

    @property
    def loc(self):
        """
        Access a group of rows and columns by label(s).

        Returns:
            _InfoTableIndexer: Indexer that wraps .loc.
        """
        return _InfoTableIndexer(self, self.df.loc)

    @property
    def iloc(self):
        """
        Access a group of rows and columns by integer position(s).

        Returns:
            _InfoTableIndexer: Indexer that wraps .iloc.
        """
        return _InfoTableIndexer(self, self.df.iloc)

    @property
    def at(self):
        """
        Access a single value for a row/column label pair.

        Returns:
            _InfoTableAtIndexer: Indexer that wraps .at.
        """
        return _InfoTableAtIndexer(self, self.df.at)

    @property
    def iat(self):
        """
        Access a single value for a row/column integer position pair.

        Returns:
            _InfoTableAtIndexer: Indexer that wraps .iat.
        """
        return _InfoTableAtIndexer(self, self.df.iat)

    def __getitem__(self, key: Any) -> Union[T, pd.Series]:
        """
        Get item from the internal DataFrame.

        Args:
            key: Key used for indexing.

        Returns:
            Union[InfoTable, pd.Series]: Sub-table or Series.
        """
        result = self.df[key]
        if isinstance(result, pd.DataFrame) and "id" in result:
            return self.__class__(result)
        return result

    def __setitem__(self, key: Any, value: Any) -> None:
        """
        Set item in the internal DataFrame.

        Args:
            key: Column label or index.
            value: Value to assign.
        """
        self.df[key] = value

    def __contains__(self, key: Any) -> bool:
        """
        Check whether key is in the DataFrame columns.

        Args:
            key: Key to check.

        Returns:
            bool: True if key is in columns.
        """
        return key in self.df

    # @property
    # def iat(self):
    #     return self.df.iat

    # @property
    # def at(self):
    #     return self.df.at

    # @property
    # def iloc(self):
    #     return self.df.iloc

    # @property
    # def loc(self):
    #     return self.df.loc

    # @property
    # def __getitem__(self):
    #     return self.df.__getitem__

    # @property
    # def __setitem__(self):
    #     return self.df.__setitem__

    # @property
    # def __contains__(self):
    #     return self.df.__contains__

    @property
    def index(self) -> pd.Index:
        """
        Get the index of the DataFrame.

        Returns:
            pd.Index: DataFrame index.
        """
        return self.df.index

    @property
    def eval(self) -> Callable:
        """
        Return the DataFrame.eval method.

        Returns:
            Callable: The eval method for evaluating expressions.
        """
        return self.df.eval

    @property
    def iterrows(self) -> Callable:
        """
        Return the DataFrame.iterrows generator.

        Returns:
            Callable: Yields (index, Series) pairs.
        """
        return self.df.iterrows

    def dropna(self, *args, **kwargs) -> T:
        """
        Return a new InfoTable with missing values dropped.

        Args:
            *args, **kwargs: Passed to pandas.DataFrame.dropna.

        Returns:
            InfoTable: A new instance with rows (or columns) with NA removed.
        """
        result = self.df.dropna(*args, **kwargs)
        if result is None:
            return None
        return self.__class__(result)

    def query(self, expr: str, **kwargs) -> T:
        """
        Filters rows using a boolean expression string.

        Args:
            expr (str): A string expression to evaluate, using column names as variables.
            **kwargs: Passed through to `pandas.DataFrame.query()`.

        Returns:
            InfoTable: A new InfoTable with filtered rows.
        """
        result = self.df.query(expr, **kwargs)
        return self.__class__(result)

    def xs(
        self, key, axis: int = 0, level: Union[int, str] = None, drop_level: bool = True
    ) -> Union[T, pd.Series]:
        """
        Returns a cross-section (row or column) from the DataFrame.

        Args:
            key: Label or tuple of labels to select.
            axis (int): Axis to retrieve from (0 for index, 1 for columns).
            level: Level in MultiIndex to use.
            drop_level (bool): Whether to drop the level(s) from the result.

        Returns:
            InfoTable or Series: If the result is a DataFrame, wraps it as InfoTable.
        """
        result = self.df.xs(key, axis=axis, level=level, drop_level=drop_level)
        if isinstance(result, pd.DataFrame):
            return self.__class__(result)
        return result

    def head(self: T, n: int = 5) -> T:
        """
        Return the first n rows.

        Args:
            n (int): Number of rows.

        Returns:
            InfoTable: A new InfoTable.
        """
        return self.__class__(self.df.head(n))

    def tail(self: T, n: int = 5) -> T:
        """
        Return the last n rows.

        Args:
            n (int): Number of rows.

        Returns:
            InfoTable: A new InfoTable.
        """
        return self.__class__(self.df.tail(n))

    def sample(self: T, n: int = 1, random_state=None) -> T:
        """
        Return a random sample of rows.

        Args:
            n (int): Number of rows.
            random_state: Seed for reproducibility.

        Returns:
            InfoTable: Sampled InfoTable.
        """
        return self.__class__(self.df.sample(n=n, random_state=random_state))

    def drop(self, labels=None, axis=0, columns=None, inplace=False) -> Optional[T]:
        """
        Drop specified labels.

        Args:
            labels: Index or column labels.
            axis (int): Whether to drop rows (0) or columns (1).
            columns: Column labels to drop.
            inplace (bool): Modify in place.

        Returns:
            Optional[InfoTable]: Modified InfoTable or None.
        """
        result = self.df.drop(
            labels=labels, axis=axis, columns=columns, inplace=inplace
        )
        if inplace:
            return None
        return self.__class__(result)

    def save(self, file_path: PathLike, sep: Optional[str] = None) -> None:
        """
        Save the InfoTable to a file.

        Args:
            file_path (str or Path): Path to save the file.
            sep (Optional[str]): Column separator (default inferred from extension).
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        ext = file_path.suffix
        if ext in ["", ".scp"] or re.match(r"\.[0-9]+$", ext):
            # if no extension we save as kaldi utt2spk file
            assert len(self.df.columns) == 2
            self.df.to_csv(file_path, sep=" ", header=False, index=False)
            return

        if sep is None:
            sep = "\t" if ".tsv" in ext else ","

        self.df.to_csv(file_path, sep=sep, index=False)

    @classmethod
    def from_lists(
        cls: Type[T],
        ids: List[str],
        column_names: List[str],
        column_data: List[List[Any]],
    ) -> T:
        """
        Create InfoTable from lists of IDs and corresponding column data.

        Args:
            ids (List[str]): List of IDs.
            column_names (List[str]): List of column names.
            column_data (List[List[Any]]): Column values.

        Returns:
            InfoTable: Constructed table.
        """
        df_dict = {"id": ids}
        assert len(column_names) == len(column_data)
        for name, data in zip(column_names, column_data):
            assert len(ids) == len(data)
            df_dict[name] = data
        df = pd.DataFrame(df_dict)
        return cls(df)

    @classmethod
    def from_dict(cls: Type[T], df_dict: Dict[str, List[Any]]) -> T:
        """
        Create InfoTable from a dictionary.

        Args:
            df_dict (Dict[str, List[Any]]): Column data including 'id'.

        Returns:
            InfoTable: Constructed table.
        """
        assert "id" in df_dict
        df = pd.DataFrame(df_dict)
        return cls(df)

    @classmethod
    def load(
        cls: Type[T],
        file_path: PathLike,
        sep: Optional[str] = None,
        name: str = "class_id",
    ) -> T:
        """
        Load InfoTable from file.

        Args:
            file_path (str or Path): Path to the input file.
            sep (Optional[str]): Column separator.
            name (str): Name of the second column (used for Kaldi format).

        Returns:
            InfoTable: Loaded table.
        """
        file_path = Path(file_path)
        ext = file_path.suffix
        if ext in ["", ".scp"]:
            # if no extension we load as kaldi utt2spk file
            df = pd.read_csv(
                file_path,
                sep=" ",
                header=None,
                names=["id", name],
                dtype={"id": str, name: str},
            )
        else:
            if sep is None:
                sep = "\t" if ".tsv" in ext else ","

            # we enforce these dtypes
            fixed_dtypes = {
                "id": str,
                "speaker": str,
                "language": str,
                "gender": str,
                "duration": float,
                "storage_path": str,
                "storage_byte": int,
                "num_frames": int,
                "video_ids": str,
                "language_est": str,
            }
            df = pd.read_csv(file_path, sep=sep, dtype=fixed_dtypes)

        return cls(df)

    def sort(
        self, column: str = "id", ascending: bool = True, inplace: bool = True
    ) -> Optional[T]:
        """
        Sort the InfoTable by a specific column.

        Args:
            column (str): Column name to sort by.
            ascending (bool): Sort in ascending order.
            inplace (bool): Sort in place or return a new object.

        Returns:
            Optional[InfoTable]: Sorted InfoTable or None if inplace.
        """
        if column == "id":
            r = self.df.sort_index(inplace=inplace, ascending=ascending)
        else:
            r = self.df.sort_values(by=column, inplace=inplace, ascending=ascending)

        if inplace:
            return None
        return self.__class__(r)

    def split(self, idx: int, num_parts: int, group_by: Optional[str] = None) -> T:
        """
        Split the InfoTable into parts and return the selected part.

        Args:
            idx (int): Part to return (1-based).
            num_parts (int): Total number of parts.
            group_by (Optional[str]): Column to group by when splitting.

        Returns:
            InfoTable: The selected part of the split.
        """
        if group_by is None or group_by == "id":
            _, idx1 = split_list(self.df["id"], idx, num_parts)
        else:
            _, idx1 = split_list_group_by_key(self.df[group_by], idx, num_parts)

        df = self.df.iloc[idx1]
        return self.__class__(df)

    @classmethod
    def cat(cls: Type[T], tables: List[T]) -> T:
        """
        Concatenate multiple InfoTables.

        Args:
            tables (List[InfoTable]): List of InfoTable objects to concatenate.

        Returns:
            InfoTable: Concatenated InfoTable.

        Raises:
            AssertionError: If resulting DataFrame has duplicate IDs.
        """
        df_list = [table.df for table in tables]
        df = pd.concat(df_list)
        assert df[
            "id"
        ].is_unique, """there are duplicated ids in the tables we are concatenating"""
        return cls(df)

    def filter(
        self: T,
        predicate: Optional[
            Callable[[pd.DataFrame], Union[pd.Series, np.ndarray]]
        ] = None,
        items: Optional[List[Any]] = None,
        iindex: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
        by: str = "id",
        keep: bool = True,
        raise_if_missing: bool = True,
    ) -> T:
        """
        Filter the InfoTable based on a predicate, item list, index list, or column subset.

        Args:
            predicate (Callable, optional): Function that returns a boolean mask, e.g.:
              lambda df: df["duration"] > 1.0.
            items (List[Any], optional): Items to include/exclude from the 'by' column like
              df.loc[items, by], used only if predicate is None
            iindex (np.ndarray, optional): Integer indices to include/exclude like
              df.iloc[iindex], used if predicate and items are None
            columns (List[str], optional): Columns to retain or remove.
            by (str): Column name to use with 'items'.
            keep (bool): Whether to keep or exclude matched rows/columns.
            raise_if_missing (bool): Raise error if items are missing.

        Returns:
            InfoTable: Filtered InfoTable.

        Raises:
            Exception: If items are not found and raise_if_missing is True.
        """
        assert (
            predicate is not None
            or items is not None
            or iindex is not None
            or columns is not None
        ), "predicate, items, iindex and columns cannot be not None at the same time"
        df = self.df

        if predicate is not None:
            mask = predicate(self.df)

        if not keep:
            if predicate is not None:
                mask = np.logical_not(mask)
            elif items is not None:
                items = np.setdiff1d(df[by], items)
            elif iindex is not None:
                iindex = np.setdiff1d(np.arange(len(df)), iindex)

            if columns is not None:
                columns = np.setdiff1d(df.columns, columns)
        else:
            if columns is not None:
                if "id" in df and "id" not in columns:
                    columns = ["id"] + columns

        if predicate is not None:
            if columns is None:
                df = df.loc[mask]
            else:
                df = df.loc[mask, columns]
        elif items is not None:
            if by != "id":
                missing = [False if v in df[by] else True for v in items]
                if any(missing) and raise_if_missing:
                    raise Exception(f"{items[missing]} not found in table")
                items = [True if v in items else False for v in df[by]]
            elif not raise_if_missing:
                items = [item for item in items if item in df.index]

            if columns is None:
                df = df.loc[items]
            else:
                df = df.loc[items, columns]
        else:
            if not raise_if_missing:
                iindex = iindex[iindex < len(df)]

            if iindex is not None:
                df = self.df.iloc[iindex]

            if columns is not None:
                df = df[columns]

        return self.__class__(df.copy())

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

    def shuffle(
        self, seed: int = 1024, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Shuffle the rows of the InfoTable.

        Args:
            seed (int): Seed for random number generator.
            rng (np.random.Generator, optional): Numpy random generator.

        Returns:
            np.ndarray: Shuffled indices.
        """
        if rng is None:
            rng = np.random.default_rng(seed=seed)
        index = np.arange(len(self.df))
        rng.shuffle(index)
        self.df = self.df.iloc[index]
        return index

    def set_index(
        self, keys: Union[str, List[str]], inplace: bool = True
    ) -> Optional[T]:
        """
        Set the DataFrame index using one or more columns.

        Args:
            keys (str or list): Column(s) to use as index.
            inplace (bool): Whether to modify in place.

        Returns:
            Optional[InfoTable]: Modified InfoTable if not inplace.
        """
        if inplace:
            self.df.set_index(keys, drop=False, inplace=True)
            return None

        df = self.df.set_index(keys, drop=False, inplace=False)
        return type(self)(df)

    def reset_index(self) -> None:
        """
        Reset the DataFrame index to the 'id' column.

        Returns:
            None
        """
        self.df.set_index("id", drop=False, inplace=True)

    def get_loc(
        self, keys: Union[str, List[str], np.ndarray]
    ) -> Union[int, np.ndarray, List[int]]:
        """
        Get integer location(s) for the given key(s).

        Args:
            keys (str, list, or np.ndarray): Index label(s).

        Returns:
            int, np.ndarray, or List[int]: Location(s) in the index.
        """
        if isinstance(keys, (list, np.ndarray)):
            return self.df.index.get_indexer(keys)

        loc = self.df.index.get_loc(keys)
        if isinstance(loc, int):
            return loc

        if isinstance(loc, np.ndarray) and loc.dtype == bool:
            return np.nonzero(loc)[0]

        return list(range(loc.start, loc.stop, loc.step))

    def get_col_idx(self, keys: Union[str, List[str]]) -> Union[int, np.ndarray]:
        """
        Get the integer index position(s) of the specified column(s).

        Args:
            keys (str or list): Column name(s).

        Returns:
            int or np.ndarray: Position(s) of the column(s).
        """
        return self.df.columns.get_loc(keys)

    def add_columns(
        self,
        right_table: Union[T, pd.DataFrame],
        column_names: Union[None, str, List[str], np.ndarray] = None,
        on: Union[str, List[str], np.ndarray] = "id",
        right_on: Union[None, str, List[str], np.ndarray] = None,
        remove_missing: bool = False,
    ) -> None:
        """
        Add new columns from another InfoTable or DataFrame.

        Args:
            right_table (Union[InfoTable, pd.DataFrame]): The table to merge columns from.
            column_names (str or list, optional): Columns to include from the right table.
            on (str or list): Key(s) from the current table.
            right_on (str or list, optional): Key(s) from the right table.
            remove_missing (bool): Use inner join (drop unmatched rows) if True.
        """
        if isinstance(right_table, InfoTable):
            right_table = right_table.df

        if column_names is not None:
            right_table = right_table[column_names]

        if right_on is None:
            right_on = on

        how = "inner" if remove_missing else "left"
        left_index = False
        right_index = False
        if on == "id" or on == ["id"]:
            on = None
            left_index = True

        if (right_on == "id" or right_on == ["id"]) and "id" in right_table:
            right_on = None
            right_index = True

        self.df = self.df.merge(
            right_table,
            how=how,
            left_on=on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            suffixes=(None, "_right"),
        )

    def replace_columns(
        self,
        right_table: Union[T, pd.DataFrame],
        column_names: Union[None, str, List[str], np.ndarray] = None,
    ) -> None:
        """
        Replace column values with those from another table.

        Args:
            right_table (Union[InfoTable, pd.DataFrame]): Table to source values from.
            column_names (str or list, optional): Columns to replace. If None, all.
        """
        if isinstance(right_table, InfoTable):
            right_table = right_table.df

        if column_names is None:
            column_names = right_table.columns

        for column in column_names:
            if column == "id":
                continue

            dtype = self.df.dtypes[column]
            if column in self.df and column in right_table:
                dtype_right = right_table.dtypes[column]
                if dtype in [np.dtype("int64"), np.dtype("int32")] and dtype_right in [
                    np.dtype("float64"),
                    np.dtype("float32"),
                ]:
                    self.df[column] = self.df[column].astype(dtype_right)

            self.df.loc[right_table.id, column] = right_table[column].astype(dtype)

    @classmethod
    def merge(
        cls: Type[T],
        left_table: Union[T, pd.DataFrame],
        right_table: Union[T, pd.DataFrame],
        how: str = "inner",
        on: Union[str, List[str], None] = None,
        left_on: Union[str, List[str], None] = None,
        right_on: Union[str, List[str], None] = None,
        left_index: bool = False,
        right_index: bool = False,
        sort: bool = False,
        suffixes: Tuple[str, str] = ("_x", "_y"),
        copy: Optional[bool] = None,
        indicator: Union[str, bool] = False,
        validate: Optional[str] = None,
    ) -> T:
        """
        Merge two InfoTables or DataFrames into a new InfoTable.

        Args:
            left_table (Union[InfoTable, pd.DataFrame]): Left-hand table.
            right_table (Union[InfoTable, pd.DataFrame]): Right-hand table.
            how (str): Merge method (e.g., 'inner', 'outer', 'left', 'right').
            on (Union[str, List[str], None]): Column(s) to join on.
            left_on (Union[str, List[str], None]): Column(s) from the left table to join on.
            right_on (Union[str, List[str], None]): Column(s) from the right table to join on.
            left_index (bool): Use index from the left table as join key.
            right_index (bool): Use index from the right table as join key.
            sort (bool): Sort the result by the join keys.
            suffixes (Tuple[str, str]): Suffixes to apply to overlapping column names.
            copy (Optional[bool]): If False, avoid copying data where possible.
            indicator (Union[str, bool]): Adds a column to the output DataFrame called '_merge'.
            validate (Optional[str]): Check if merge is of specified type.

        Returns:
            InfoTable: A new merged InfoTable.
        """
        if isinstance(left_table, InfoTable):
            left_table = left_table.df

        if isinstance(right_table, InfoTable):
            right_table = right_table.df

        df = pd.merge(
            left=left_table,
            right=right_table,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            sort=sort,
            suffixes=suffixes,
            copy=copy,
            indicator=indicator,
            validate=validate,
        )
        return cls(df)

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
