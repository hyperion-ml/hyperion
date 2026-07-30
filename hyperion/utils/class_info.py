"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Type, TypeVar, Union

import numpy as np
import pandas as pd

from .info_table import InfoTable
from .misc import PathLike

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

    Examples:
        >>> import pandas as pd
        >>> from hyperion.utils.class_info import ClassInfo
        >>> df = pd.DataFrame({"id": ["spk1", "spk2", "spk3"]})
        >>> ci = ClassInfo(df)
        >>> ci.num_classes
        3
        >>> ci.weights(["spk1", "spk2"]).tolist()
        [0.3333333333333333, 0.3333333333333333]
        >>> ci.set_zero_weight(["spk1"])
        >>> ci.weights("spk1")
        0.0
        >>> ci2 = ci.filter(items=["spk2", "spk3"], rebuild_idx=True)
        >>> ci2.num_classes
        2
    """

    def __init__(self, df: Union[pd.DataFrame, T]) -> None:
        """
        Initialize ClassInfo with automatic class index and normalized weights.

        Args:
            df (pd.DataFrame or ClassInfo): Input data.
        """
        super().__init__(df)
        if "class_idx" not in self.df:
            self.add_class_idx()

        if "weights" not in self.df:
            self.set_uniform_weights()
        else:
            self.renorm_weights()

    def add_class_idx(self, sort_by_id: bool = False) -> None:
        """
        Assign a unique integer class index to each row.
        """
        if sort_by_id:
            self.sort()
        self.df["class_idx"] = [i for i in range(len(self.df))]

    def set_uniform_weights(self) -> None:
        """
        Set uniform weights across all classes.
        """
        if self.df.empty:
            self.df["weights"] = pd.Series(dtype="float64")
            return
        self.df["weights"] = 1 / len(self.df)

    def set_weights(self, weights: Union[pd.Series, np.ndarray]) -> None:
        """
        Set class weights and normalize them.

        Args:
            weights (pd.Series or np.ndarray): Raw weights.
        """
        if self.df.empty:
            self.df["weights"] = pd.Series(dtype="float64")
            return

        total = weights.sum()
        if total is None or not np.isfinite(total) or total <= 0:
            raise ValueError(f"weights must have a finite positive sum, got {total}")

        self.df["weights"] = weights / total

    def renorm_weights(self) -> None:
        """
        Renormalize existing weights to ensure they sum to 1.
        """
        if self.df.empty:
            self.df["weights"] = pd.Series(dtype="float64")
            return

        weights = self.df["weights"]
        total = weights.sum()
        if total is None or not np.isfinite(total) or total <= 0:
            raise ValueError(f"weights must have a finite positive sum, got {total}")

        self.df["weights"] = weights / total

    def exp_weights(self, x: Union[int, float]) -> None:
        """
        Raise weights to the power of x and re-normalize.

        Args:
            x (int or float): Exponent to apply to weights.
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
        self.renorm_weights()

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
        if self.df.empty:
            return 0

        class_idx = self.df["class_idx"].dropna()
        if class_idx.empty:
            return 0

        return int(class_idx.max()) + 1

    def sort_by_idx(self, ascending: bool = True) -> None:
        """
        Sort entries by class index.

        Args:
            ascending (bool): Whether to sort in ascending order.
        """
        self.sort("class_idx", ascending)

    @classmethod
    def load(cls: Type[T], file_path: PathLike, sep: Optional[str] = None) -> T:
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
        if len(tables) == 0:
            raise ValueError("tables must contain at least one ClassInfo")

        df_list = [table.df for table in tables]
        df = pd.concat(df_list)
        if not df["id"].is_unique:
            logging.warning("""there are duplicated ids in original tables, 
                            removing duplicated rows""")
            df.drop_duplicates(subset="id", keep="first", inplace=True)

        if not df["class_idx"].is_unique:
            logging.warning("""class_idx in concat tables are not unique, 
                we will assign new class_idx""")
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
