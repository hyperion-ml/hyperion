"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import re
from collections import OrderedDict
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray
from pandas.api.types import infer_dtype, is_numeric_dtype

from .list_utils import split_list, split_list_group_by_key
from .misc import PathLike

T = TypeVar("T", bound="InfoTable")

_NULL_SCAN_CHUNK = 1024 * 1024  # 1MB


def _sanitize_dataframe_null_chars(df: pd.DataFrame) -> None:
    """
    Remove '\\x00' characters from string-like columns in the dataframe.

    Args:
        df (pd.DataFrame): DataFrame to sanitize.
    """
    if df.columns.has_duplicates:
        dup_cols = df.columns[df.columns.duplicated()]
        dup_labels = ", ".join(map(str, dup_cols))
        raise ValueError(
            "InfoTable contains duplicated column names, which is unsupported: "
            f"{dup_labels}"
        )

    nulls_found = False

    for col_idx, column in enumerate(df.columns):
        # Use positional indexing so that duplicate column names are handled
        # one at a time and we always get a Series instead of a DataFrame.
        series = df.iloc[:, col_idx]
        mask = series.map(
            lambda value: (isinstance(value, str) and "\x00" in value)
            or isinstance(value, (bytes, bytearray))
        )
        if mask.isna().any():
            mask = mask.fillna(False)
        if bool(mask.any()):
            nulls_found = True
            df.loc[mask, column] = series.loc[mask].map(
                lambda value: (
                    value.replace("\x00", "")
                    if isinstance(value, str)
                    else (
                        value.decode("utf-8", errors="replace")
                        if isinstance(value, (bytes, bytearray))
                        else value
                    )
                )
            )

    if nulls_found:
        logging.warning("Removed NULL characters from dataframe before saving.")


def _file_has_null_bytes(path: Path) -> bool:
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(_NULL_SCAN_CHUNK), b""):
            if b"\x00" in chunk:
                return True
    return False


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
        if isinstance(result, pd.DataFrame) and self.parent.__class__.is_valid_df(
            result
        ):
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

    @staticmethod
    def is_valid_df(df: pd.DataFrame) -> bool:
        """
        Check if the DataFrame is valid for InfoTable.

        Args:
            df (pd.DataFrame): DataFrame to check.

        Returns:
            bool: True if valid, False otherwise.
        """
        return "id" in df

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
            self.df[column] = self.df[column].astype("string")

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
    def columns(self):
        """
        Get the DataFrame columns.

        Returns:
            pd.Index: Column names.
        """
        return self.df.columns

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
        if isinstance(result, pd.DataFrame) and self.is_valid_df(result):
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

        _sanitize_dataframe_null_chars(self.df)

        is_kaldi_format = ext in ["", ".scp"] or re.match(r"\.[0-9]+$", ext)
        if is_kaldi_format:
            assert len(self.df.columns) == 2
            sep_to_use = " "
            header = False
        else:
            sep_to_use = sep if sep is not None else ("\t" if ".tsv" in ext else ",")
            header = True

        read_kwargs: Dict[str, Any] = {
            "sep": sep_to_use,
            "dtype": str,
            "keep_default_na": False,
            "engine": "python",
            "on_bad_lines": "error",
        }
        if not header:
            read_kwargs["header"] = None

        # First attempt: regular pandas write to disk.
        self.df.to_csv(
            file_path,
            sep=sep_to_use,
            index=False,
            header=header,
            encoding="utf-8",
        )

        needs_fallback = False
        try:
            disk_df = pd.read_csv(file_path, **read_kwargs)
            if len(disk_df) != len(self.df):
                logging.warning(
                    "Row count mismatch after writing %s. Expected %d rows, got %d.",
                    file_path,
                    len(self.df),
                    len(disk_df),
                )
                needs_fallback = True
        except Exception as read_error:
            logging.warning(
                "Verification read failed for %s after direct write: %s",
                file_path,
                read_error,
            )
            needs_fallback = True

        if not needs_fallback and not _file_has_null_bytes(file_path):
            return

        logging.warning(
            "Rewriting %s using in-memory CSV serialization due to detected issues.",
            file_path,
        )

        csv_buffer = StringIO()
        self.df.to_csv(
            csv_buffer,
            sep=sep_to_use,
            index=False,
            header=header,
            lineterminator="\n",
        )
        csv_content = csv_buffer.getvalue()

        if "\x00" in csv_content:
            logging.warning(
                "Removing NULL characters from serialized content for %s.", file_path
            )
            csv_content = csv_content.replace("\x00", "")

        read_df = pd.read_csv(StringIO(csv_content), **read_kwargs)
        if len(read_df) != len(self.df):
            raise ValueError(
                f"Row count mismatch when serializing {file_path} in-memory. "
                f"Expected {len(self.df)}, got {len(read_df)}."
            )

        with file_path.open("w", encoding="utf-8", newline="") as out_file:
            out_file.write(csv_content)

        if _file_has_null_bytes(file_path):
            raise ValueError(
                f"NULL bytes detected in {file_path} after in-memory rewrite."
            )

        disk_df = pd.read_csv(file_path, **read_kwargs)
        if len(disk_df) != len(self.df):
            raise ValueError(
                f"Row count mismatch when saving {file_path}. "
                f"Expected {len(self.df)}, got {len(disk_df)}."
            )

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
                "speaker": "string",
                "language": "string",
                "gender": "string",
                "duration": "float",
                "storage_path": "string",
                "storage_byte": "int",
                "num_frames": "int",
                "video_ids": "string",
                "language_est": "string",
                "accent": "string",
                "age_decade": "string",
                "sample_freq": "Int64",
                "up_votes": "Int64",
                "down_votes": "Int64",
                "accents": "string",
                "gender_extended": "string",
                "age": pd.UInt8Dtype(),
                "transcript": "string",
                "transcript_normalized": "string",
                "sentence_domain": "string",
                "iarpa_arts_age": "string",
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
        if not df["id"].is_unique:
            duplicated_ids = df.loc[df["id"].duplicated(keep=False), "id"].tolist()
            raise AssertionError(
                "there are duplicated ids in the tables we are concatenating: "
                f"{duplicated_ids}"
            )
        return cls(df)

    def filter(
        self: T,
        predicate: Union[
            Callable[[pd.DataFrame], Union[pd.Series, np.ndarray]],
            str,
            None,
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
            if isinstance(predicate, str):
                if not keep:
                    predicate = f"not ({predicate})"
                    df = df.query(predicate)

                predicate = None
            else:
                mask = predicate(df)

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
                df = df.iloc[iindex]

            if columns is not None:
                df = df[columns]

        return self.__class__(df.copy())

    def histogram(
        self,
        column: str,
        bins: Optional[int] = None,
        density: bool = True,
        kind: str = "bar",
        color: str = "C0",
        output_file: Optional[PathLike] = None,
        dropna: bool = True,
    ) -> None:
        """
        Plot a histogram of a column, handling string or numeric data.

        Args:
            column (str): Column name to plot. Title will include the column name.
            bins (Optional[int]): Number of bins for numeric columns. Ignored for strings.
            density (bool): If True, plot density/relative frequency; otherwise counts.
            kind (str): Histogram style, either "bar" or "line".
            color (str): Matplotlib color spec for the plot.
            output_file (Optional[PathLike]): If provided, save the figure; otherwise show it.
            dropna (bool): If True, ignore NA values before plotting.
        """
        import matplotlib.pyplot as plt

        if column not in self.df.columns:
            raise KeyError(f"Column '{column}' not found in table")

        series = self.df[column]
        if dropna:
            series = series.dropna()

        if series.empty:
            raise ValueError(
                f"No data available to plot histogram for column '{column}'"
            )

        kind = kind.lower()
        if kind not in ("bar", "line"):
            raise ValueError("kind must be either 'bar' or 'line'")

        data_type = infer_dtype(series, skipna=True)
        fig, ax = plt.subplots()

        if np.issubdtype(series.dtype, np.number):
            num_bins = bins if bins is not None else 10
            if kind == "bar":
                ax.hist(
                    series,
                    bins=num_bins,
                    density=density,
                    color=color,
                    edgecolor="black",
                )
            else:
                counts, bin_edges = np.histogram(series, bins=num_bins, density=density)
                centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax.plot(centers, counts, color=color)
        elif data_type in ["string", "unicode", "bytes", "mixed-integer", "mixed"]:
            counts = series.value_counts(dropna=dropna, sort=False)
            if density:
                counts = counts / counts.sum()
            counts = counts.sort_index()
            x = counts.index.tolist()
            y = counts.values
            if kind == "bar":
                ax.bar(x, y, color=color)
            else:
                ax.plot(x, y, color=color, marker="o")
        else:
            raise TypeError(f"Unsupported dtype '{series.dtype}' for histogram")

        ax.set_title(f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Density" if density else "Count")
        plt.xticks(
            rotation=(
                45
                if data_type in ["string", "unicode", "bytes", "mixed-integer", "mixed"]
                else 0
            )
        )
        ax.grid(True)
        plt.tight_layout()

        if output_file is None:
            plt.show()
        else:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path)
            plt.close(fig)

    def scatter2d(
        self,
        x_column: str,
        y_column: str,
        color: str = "C0",
        marker: str = "o",
        sample_frac: float = 1.0,
        output_file: Optional[PathLike] = None,
        dropna: bool = True,
    ) -> None:
        """
        Plot a 2D scattergram for two numeric columns.

        Args:
            x_column (str): Column name for the x-axis.
            y_column (str): Column name for the y-axis.
            color (str): Matplotlib color for points.
            marker (str): Matplotlib marker style.
            sample_frac (float): Fraction (0,1] of points to plot, sampled randomly.
            output_file (Optional[PathLike]): If provided, save figure; otherwise show it.
            dropna (bool): Drop NA rows before plotting.
        """
        import matplotlib.pyplot as plt

        if not (0 < sample_frac <= 1):
            raise ValueError("sample_frac must be in the range (0, 1]")

        df = self.df[[x_column, y_column]]
        if dropna:
            df = df.dropna()
        if sample_frac < 1:
            df = df.sample(frac=sample_frac)

        if df.empty:
            raise ValueError("No data available to plot scatter2d")

        fig, ax = plt.subplots()
        ax.scatter(df[x_column], df[y_column], c=color, marker=marker)
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(f"{y_column} vs {x_column}")
        ax.grid(True)
        plt.tight_layout()

        if output_file is None:
            plt.show()
        else:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path)
            plt.close(fig)

    def scatter3d(
        self,
        x_column: str,
        y_column: str,
        z_column: str,
        color: str = "C0",
        marker: str = "o",
        sample_frac: float = 1.0,
        output_file: Optional[PathLike] = None,
        dropna: bool = True,
    ) -> None:
        """
        Plot a 3D scattergram for three numeric columns.

        Args:
            x_column (str): Column name for the x-axis.
            y_column (str): Column name for the y-axis.
            z_column (str): Column name for the z-axis.
            color (str): Matplotlib color for points.
            marker (str): Matplotlib marker style.
            sample_frac (float): Fraction (0,1] of points to plot, sampled randomly.
            output_file (Optional[PathLike]): If provided, save figure; otherwise show it.
            dropna (bool): Drop NA rows before plotting.
        """
        import matplotlib.pyplot as plt

        if not (0 < sample_frac <= 1):
            raise ValueError("sample_frac must be in the range (0, 1]")

        df = self.df[[x_column, y_column, z_column]]
        if dropna:
            df = df.dropna()
        if sample_frac < 1:
            df = df.sample(frac=sample_frac)

        if df.empty:
            raise ValueError("No data available to plot scatter3d")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(df[x_column], df[y_column], df[z_column], c=color, marker=marker)
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_zlabel(z_column)
        ax.set_title(f"{z_column} vs {y_column} vs {x_column}")
        ax.grid(True)
        plt.tight_layout()

        if output_file is None:
            plt.show()
        else:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path)
            plt.close(fig)

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
        if isinstance(
            keys, (list, tuple, np.ndarray, pd.Index, pd.Series, ExtensionArray)
        ):
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

    def __getattr__(self, name: str):
        """
        Provide attribute-style access to DataFrame columns and attributes.

        Falls back to self.df[name] if name is a column.
        Falls back to getattr(self.df, name) if it's a DataFrame method or attribute.
        Special methods (e.g., __setstate__) raise AttributeError immediately to
        allow correct behavior during pickling/unpickling.

        Args:
            name (str): The attribute being accessed.

        Returns:
            Any: The corresponding column, method, or attribute.

        Raises:
            AttributeError: If the attribute is not found.
        """
        # Avoid recursion for Python magic methods like __setstate__, __reduce__, etc.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")

        try:
            df = object.__getattribute__(self, "df")
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}' and 'df' is not yet initialized."
            )

        if name in df.columns:
            return df[name]
        if hasattr(df, name):
            return getattr(df, name)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def add_columns(
        self,
        right_table: Union[T, pd.DataFrame],
        column_names: Union[None, str, List[str], np.ndarray] = None,
        on: Union[str, List[str], np.ndarray] = "id",
        right_on: Union[None, str, List[str], np.ndarray] = None,
        replace_overlapping: bool = False,
        ignore_overlapping: bool = False,
        remove_missing: bool = False,
    ) -> None:
        """
        Add new columns from another InfoTable or DataFrame.

        Args:
            right_table (Union[InfoTable, pd.DataFrame]): The table to merge columns from.
            column_names (str or list, optional): Columns to include from the right table.
            on (str or list): Key(s) from the current table.
            right_on (str or list, optional): Key(s) from the right table.
            replace_overlapping (bool): Replace overlapping columns if True.
            ignore_overlapping (bool): If True, skip adding columns that already exist.
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

        # 'id' exists in every InfoTable, so merging with another table will
        # produce an unnecessary 'id_right' copy when pandas applies suffixes.
        if "id_right" in self.df.columns:
            self.df.drop(columns=["id_right"], inplace=True)

        if ignore_overlapping:
            # Drop the extra right-side columns without touching existing data
            for col in right_table.columns:
                extra_col = f"{col}_right"
                if extra_col in self.df.columns:
                    self.df.drop(columns=[extra_col], inplace=True)
            return

        if not replace_overlapping:
            return

        # For overlapping columns: update only where right side is not null
        for col in right_table.columns:
            if col == "id":
                continue  # skip the key

            update_col = f"{col}_right"
            if update_col in self.df.columns:
                self.df[col] = self.df[update_col].combine_first(self.df[col])
                self.df.drop(columns=[update_col], inplace=True)

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

    def harmonize_columns_by_majority_vote(
        self, voter_columns: Union[str, List[str]], target_columns: Union[str, List[str]]
    ) -> None:
        """
        Force all rows sharing ``voter_columns`` to take the majority value on ``target_columns``.

        Args:
            voter_columns (Union[str, List[str]]): Column(s) that define the voting group (e.g., speaker).
            target_columns (Union[str, List[str]]): Column or columns to harmonize via majority vote.

        Raises:
            KeyError: If the voter or target columns are missing.
        """
        if isinstance(voter_columns, str):
            voter_columns = [voter_columns]

        missing_voters = [col for col in voter_columns if col not in self.df.columns]
        if missing_voters:
            raise KeyError(
                "Voting columns not found in table: " + ", ".join(map(str, missing_voters))
            )

        if isinstance(target_columns, str):
            target_columns = [target_columns]

        missing_columns = [col for col in target_columns if col not in self.df.columns]
        if missing_columns:
            raise KeyError(
                "Columns not found in table: " + ", ".join(map(str, missing_columns))
            )

        def _majority_vote(series: pd.Series) -> Any:
            votes = series.dropna()
            if votes.empty:
                return pd.NA
            counts = votes.value_counts()
            return counts.index[0]

        valid_voters_mask = self.df[voter_columns].notna().all(axis=1)
        if not valid_voters_mask.any():
            return

        valid_df = self.df.loc[valid_voters_mask]
        grouped = valid_df.groupby(voter_columns, dropna=False, sort=False)
        group_indices = grouped.groups

        for column in target_columns:
            majority_buffer = pd.Series(pd.NA, index=valid_df.index, dtype="object")
            prob_column = f"{column}_prob" if f"{column}_prob" in self.df.columns else None
            prob_buffer = (
                pd.Series(np.nan, index=valid_df.index, dtype="float64")
                if prob_column is not None
                else None
            )

            for idx in group_indices.values():
                group_series = valid_df.loc[idx, column]
                winner = _majority_vote(group_series)
                if pd.isna(winner):
                    continue

                majority_buffer.loc[idx] = winner

                if prob_buffer is None:
                    continue

                group_probs = valid_df.loc[idx, prob_column]
                winner_mask = group_series == winner
                if not winner_mask.any():
                    continue

                winner_probs = group_probs[winner_mask]
                winner_probs = pd.to_numeric(winner_probs, errors="coerce")
                winner_probs = winner_probs.dropna()
                winner_probs = winner_probs[winner_probs > 0]
                if winner_probs.empty:
                    continue

                log_mean = np.log(winner_probs.astype(float)).mean()
                prob_value = float(np.exp(log_mean))
                prob_buffer.loc[idx] = prob_value

            mask = majority_buffer.notna()
            if mask.any():
                values = majority_buffer[mask]
                try:
                    values = values.astype(self.df[column].dtype)
                except (TypeError, ValueError):
                    pass
                self.df.loc[values.index, column] = values

            if prob_buffer is not None:
                prob_mask = prob_buffer.notna()
                if prob_mask.any():
                    prob_values = prob_buffer[prob_mask]
                    try:
                        prob_values = prob_values.astype(self.df[prob_column].dtype)
                    except (TypeError, ValueError):
                        pass
                    self.df.loc[prob_values.index, prob_column] = prob_values

    def harmonize_columns_by_average(
        self, voter_columns: Union[str, List[str]], target_columns: Union[str, List[str]]
    ) -> None:
        """
        Force all rows sharing ``voter_columns`` to take the average value on ``target_columns``.

        Args:
            voter_columns (Union[str, List[str]]): Column(s) that define the group used for averaging.
            target_columns (Union[str, List[str]]): Numeric column(s) to harmonize via averaging.

        Raises:
            KeyError: If the voter or target columns are missing.
            TypeError: If any target column is not numeric.
        """
        if isinstance(voter_columns, str):
            voter_columns = [voter_columns]

        missing_voters = [col for col in voter_columns if col not in self.df.columns]
        if missing_voters:
            raise KeyError(
                "Voting columns not found in table: " + ", ".join(map(str, missing_voters))
            )

        if isinstance(target_columns, str):
            target_columns = [target_columns]

        missing_columns = [col for col in target_columns if col not in self.df.columns]
        if missing_columns:
            raise KeyError(
                "Columns not found in table: " + ", ".join(map(str, missing_columns))
            )

        for column in target_columns:
            if not is_numeric_dtype(self.df[column]):
                raise TypeError(
                    f"Column '{column}' must be numeric to harmonize by averaging."
                )

        valid_voters_mask = self.df[voter_columns].notna().all(axis=1)
        if not valid_voters_mask.any():
            return

        grouped = self.df.loc[valid_voters_mask].groupby(voter_columns, dropna=False)
        for column in target_columns:
            averages = grouped[column].transform("mean")
            mask = averages.notna()
            if not mask.any():
                continue

            update_index = self.df.index[valid_voters_mask][mask]
            averaged_values = averages[mask]
            try:
                averaged_values = averaged_values.astype(self.df[column].dtype)
            except (TypeError, ValueError):
                pass

            self.df.loc[update_index, column] = averaged_values

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

    def replace(
        self: T, to_replace=None, value=None, inplace: bool = False, **kwargs
    ) -> Optional[T]:
        """
        Replace values in the DataFrame.

        Args:
            to_replace: What to replace.
            value: Value to replace with.
            inplace (bool): Whether to modify the table in-place.
            **kwargs: Additional keyword args passed to pd.DataFrame.replace().

        Returns:
            Optional[InfoTable]: New InfoTable if not inplace, else None.
        """
        result = self.df.replace(
            to_replace=to_replace, value=value, inplace=inplace, **kwargs
        )
        if inplace:
            return None
        return self.__class__(result)
