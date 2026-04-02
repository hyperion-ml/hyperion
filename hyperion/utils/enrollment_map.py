"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from pathlib import Path
from typing import List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd

from .info_table import InfoTable
from .list_utils import split_list_group_by_key
from .misc import PathLike

T = TypeVar("T", bound="EnrollmentMap")


class EnrollmentMap(InfoTable):
    """
    Mapping between enrollment model IDs and segment IDs.

    Required columns are ``id`` (model identifier) and ``segmentid``.

    Examples:
        >>> import pandas as pd
        >>> from hyperion.utils.enrollment_map import EnrollmentMap
        >>> df = pd.DataFrame({"id": ["m1", "m1", "m2"], "segmentid": ["s1", "s2", "s3"]})
        >>> emap = EnrollmentMap(df)
        >>> uniq, inv = emap.model_idx()
        >>> uniq.tolist()
        ['m1', 'm2']
        >>> emap_part = emap.split(1, 2)
        >>> isinstance(emap_part, EnrollmentMap)
        True
        >>> merged = EnrollmentMap.cat([emap_part, emap.split(2, 2)])
        >>> len(merged) == len(emap)
        True
    """

    def __init__(self, df: Union[pd.DataFrame, T]) -> None:
        """
        Initialize the enrollment map.

        Args:
            df (pd.DataFrame or EnrollmentMap): Input mapping table.
        """
        if "modelid" in df:
            df.rename(columns={"modelid": "id"}, inplace=True)
        assert "segmentid" in df
        super().__init__(df)

    def split(self, idx: int, num_parts: int) -> "EnrollmentMap":
        """
        Split the map into ``num_parts`` and return partition ``idx``.

        Args:
            idx (int): 1-based partition index to return.
            num_parts (int): Total number of partitions.

        Returns:
            EnrollmentMap: Requested partition.
        """
        _, idx1 = split_list_group_by_key(self.df["id"], idx, num_parts)

        df = self.df.iloc[idx1]
        return EnrollmentMap(df)

    def save(
        self,
        file_path: PathLike,
        sep: Optional[str] = None,
        nist_compatible: bool = True,
    ) -> None:
        """
        Save the enrollment map to disk.

        Args:
            file_path (PathLike): Output path.
            sep (Optional[str]): Optional delimiter override for non-``.scp`` files.
            nist_compatible (bool): If True, save ``id`` as ``modelid``.
        """
        if not nist_compatible:
            super().save(file_path, sep)
            return

        # For compatibility with NIST SRE files the index column "id"
        # is saved as modelid.
        self.df.rename(columns={"id": "modelid"}, inplace=True)
        try:
            super().save(file_path, sep)
        finally:
            # Always restore the in-memory schema even if save fails.
            if "modelid" in self.df.columns and "id" not in self.df.columns:
                self.df.rename(columns={"modelid": "id"}, inplace=True)

    @classmethod
    def load(cls: Type[T], file_path: PathLike, sep: Optional[str] = None) -> T:
        """
        Load an EnrollmentMap from file.

        Args:
            file_path (PathLike): File to read.
            sep (Optional[str]): Delimiter for text/CSV/TSV formats.

        Returns:
            EnrollmentMap: Loaded enrollment map.
        """
        file_path = Path(file_path)
        ext = file_path.suffix
        if ext in ["", ".scp"]:
            # if no extension we load as kaldi utt2spk file
            df = pd.read_csv(
                file_path,
                sep=" ",
                header=None,
                names=["segmentid", "modelid"],
                dtype={"segmentid": str, "modelid": str},
            )
            df = df[["modelid", "segmentid"]]
        else:
            if sep is None:
                sep = "\t" if ".tsv" in ext else ","

            df = pd.read_csv(file_path, sep=sep)

        return cls(df)

    @classmethod
    def cat(cls: Type[T], tables: List[T]) -> T:
        """
        Concatenate several enrollment maps.

        Args:
            tables (List[EnrollmentMap]): Input tables.

        Returns:
            EnrollmentMap: Concatenated table.
        """
        if len(tables) == 0:
            raise ValueError("tables must contain at least one EnrollmentMap")

        df_list = [table.df for table in tables]
        df = pd.concat(df_list)
        return cls(df)

    def model_idx(
        self, modelids: Optional[Union[List[str], np.ndarray]] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Return mapping from segments to model indices.

        Args:
            modelids (Optional[Union[List[str], np.ndarray]]): Ordered model IDs
            used to assign integer indices. If ``None``, IDs are inferred with
            ``np.unique``.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
            If ``modelids`` is ``None``, returns
            ``(unique_modelids, segment_to_model_index)``.
            Otherwise returns only ``segment_to_model_index`` and assigns ``-1``
            to rows whose model ID does not appear in ``modelids``.
        """
        if modelids is None:
            return np.unique(self.df["id"], return_inverse=True)

        enroll_idx = -np.ones((len(self.df)), dtype=int)
        for i, modelid in enumerate(modelids):
            idx = self.df["id"] == modelid
            enroll_idx[idx] = i

        return enroll_idx

    def get_unique_modelid_df(self) -> pd.DataFrame:
        """
        Return a DataFrame with sorted unique model IDs.

        Returns:
            pd.DataFrame: Unique rows over the model-id columns.
        """
        df = self.df.drop(columns=["segmentid"])
        df = df.drop_duplicates().sort_index()
        return df
