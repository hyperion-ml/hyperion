"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import numpy as np
import pandas as pd

from .info_table import InfoTable

T = TypeVar("T", bound="ClassInfo")


class ClassInfo(InfoTable):
    """
    A subclass of InfoTable for managing classification metadata.

    Ensures that each entry has a unique class index and maintains class weights.

    Attributes:
        df (pd.DataFrame): Underlying DataFrame containing:
            - 'id' (str): Unique identifier for each class entry.
            - 'class_idx' (int): Unique index assigned to each class.
            - 'weights' (float): Class weights normalized to sum to 1.
    """

    def __init__(self, df: Union[pd.DataFrame, T]):
        """
        Initialize ClassInfo with automatic class index and normalized weights.

        Args:
            df (pd.DataFrame or InfoTable): Input data.
        """
        super().__init__(df)
        if "class_idx" not in self.df:
            self.add_class_idx()

        if "weights" not in self.df:
            self.set_uniform_weights()
        else:
            self.df["weights"] /= self.df["weights"].sum()

    def add_class_idx(self, sort_by_id: bool = False):
        """
        Assign a unique integer class index to each row.
        """
        if sort_by_id:
            self.sort()
        self.df["class_idx"] = [i for i in range(len(self.df))]

    def set_uniform_weights(self):
        """
        Set uniform weights across all classes.
        """
        self.df["weights"] = 1 / len(self.df)

    def set_weights(self, weights: Union[pd.Series, np.ndarray]):
        """
        Set class weights and normalize them.

        Args:
            weights (pd.Series or np.ndarray): Raw weights.
        """
        self.df["weights"] = weights / weights.sum()

    def renorm_weights(self):
        """
        Renormalize existing weights to ensure they sum to 1.
        """
        weights = self.df["weights"]
        self.df["weights"] = weights / weights.sum()

    def exp_weights(self, x):
        """
        Raise weights to the power of x and re-normalize.

        Args:
            x (float): Exponent to apply to weights.
        """
        weights = self.df["weights"] ** x
        self.set_weights(weights)

    def set_zero_weight(self, ids: Union[List[str], np.ndarray]) -> None:
        """
        Set weights of selected IDs to zero and renormalize the rest.

        Args:
            ids (list or np.ndarray): List of IDs to zero out.
        """
        self.df.loc[ids, "weights"] = 0
        self.df["weights"] /= self.df["weights"].sum()

    @property
    def weights(self, ids: Union[str, List[str]]) -> Union[float, pd.Series]:
        """
        Get the weight(s) for given ID(s).

        Args:
            ids (str or list): Single ID or list of IDs.

        Returns:
            float or pd.Series: Corresponding weight(s).
        """
        return self.df.loc[ids, "weights"]

    @property
    def num_classes(self) -> int:
        """
        Number of distinct classes in the table.

        Returns:
            int: Maximum class index + 1.
        """
        return self.df["class_idx"].values.max() + 1

    def sort_by_idx(self, ascending: bool = True) -> None:
        """
        Sort entries by class index.

        Args:
            ascending (bool): Whether to sort in ascending order.
        """
        self.sort("class_idx", ascending)

    @classmethod
    def load(cls: Type[T], file_path: Union[str, Path], sep: Optional[str] = None) -> T:
        """
        Load ClassInfo from file.

        Args:
            file_path (str or Path): Path to the input file.
            sep (Optional[str]): Column separator.

        Returns:
            ClassInfo: Loaded ClassInfo instance.
        """
        file_path = Path(file_path)
        ext = file_path.suffix
        if ext == "":
            # if no extension we load as kaldi utt2spk file
            df = pd.read_csv(
                file_path,
                sep=" ",
                header=None,
                names=["id"],
                dtype={"id": str},
            )
            return cls(df)

        return super().load(file_path, sep)

    @classmethod
    def cat(cls: Type[T], tables: List[T]) -> T:
        """
        Concatenate multiple ClassInfo tables.

        Args:
            tables (List[ClassInfo]): List of ClassInfo objects.

        Returns:
            ClassInfo: Concatenated and validated ClassInfo.
        """
        df_list = [table.df for table in tables]
        df = pd.concat(df_list)
        if not df["id"].is_unique:
            logging.warning(
                """there are duplicated ids in original tables, 
                            removing duplicated rows"""
            )
            df.drop_duplicates(subset="id", keep="first", inplace=True)

        if not df["class_idx"].is_unique:
            logging.warning(
                """class_idx in concat tables are not unique, 
                we will assign new class_idx"""
            )
            df.drop(columns=["class_idx"], inplace=True)
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
        rebuild_idx: bool = False,
    ) -> T:
        """
        Filter rows from ClassInfo with optional index rebuilding.

        Args:
            predicate (Callable, optional): Boolean function to filter rows.
            items (list, optional): Items to filter by.
            iindex (np.ndarray, optional): Row indices.
            columns (list, optional): Columns to keep.
            by (str): Column to apply item filter on.
            keep (bool): Whether to keep or exclude matching rows.
            rebuild_idx (bool): Reassign class_idx after filtering.

        Returns:
            ClassInfo: Filtered and optionally reindexed ClassInfo.
        """
        new_class_info = super().filter(predicate, items, iindex, columns, by, keep)
        if rebuild_idx:
            new_class_info.add_class_idx()

        return new_class_info
