"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from pathlib import Path
from typing import Optional, Type, TypeVar, Union

import numpy as np
import pandas as pd

from .info_table import InfoTable
from .misc import PathLike

T = TypeVar("T", bound="FeatureSet")


class FeatureSet(InfoTable):
    """
    InfoTable specialization for feature manifests.

    The table must contain ``id`` and ``storage_path`` columns.

    Examples:
        >>> import pandas as pd
        >>> from hyperion.utils.feature_set import FeatureSet
        >>> df = pd.DataFrame({"id": ["utt1"], "storage_path": ["feats.ark:123"]})
        >>> feats = FeatureSet(df)
        >>> feats.add_prefix_to_storage_path("/mnt/data")
        >>> feats.df.loc["utt1", "storage_path"]
        '/mnt/data/feats.ark:123'
        >>> feats2 = feats.filter(items=["utt1"])
        >>> len(feats2)
        1
    """

    def __init__(self, df: Union[pd.DataFrame, T]) -> None:
        """
        Initialize a feature set.

        Args:
            df (pd.DataFrame or FeatureSet): Input metadata table.
        """
        super().__init__(df)
        assert "storage_path" in df

    def add_prefix_to_storage_path(self, prefix: PathLike) -> None:
        """
        Prepend a directory prefix to ``storage_path`` values.

        Args:
            prefix (PathLike): Prefix path to join with each storage path.
        """
        prefix = Path(prefix)
        self.df["storage_path"] = self.df["storage_path"].apply(
            lambda x: str(prefix / x)
        )

    def save(self, file_path: PathLike, sep: Optional[str] = None) -> None:
        """
        Save the feature set to disk.

        Args:
            file_path (PathLike): Output file path.
            sep (Optional[str]): Delimiter for non-``.scp`` files.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        ext = file_path.suffix
        if ext == ".scp":
            # if no extension we save as kaldi feats.scp file
            from .scp_list import SCPList

            offset = self.df["storage_byte"] if "storage_byte" in self.df else None
            range_spec = None
            if "start" in self.df and "num_frames" in self.df:
                range_spec = [
                    np.array([s, n], dtype=np.int64)
                    for s, n in self.df[["start", "num_frames"]]
                ]
            scp = SCPList(
                self.df["id"].values, self.df["storage_path"].values, offset, range_spec
            )
            scp.save(file_path)
            return

        super().save(file_path, sep)

    @classmethod
    def load(cls: Type[T], file_path: PathLike, sep: Optional[str] = None) -> T:
        """
        Load a feature set from disk.

        Args:
            file_path (PathLike): Input file path.
            sep (Optional[str]): Delimiter for non-``.scp`` files.

        Returns:
            FeatureSet: Loaded feature set.
        """
        file_path = Path(file_path)
        ext = file_path.suffix
        if ext == ".scp":
            # if no extension we load as kaldi feats.scp file
            from .scp_list import SCPList

            scp = SCPList.load(file_path)
            df_dict = {"id": scp.key, "storage_path": scp.file_path}
            df = pd.DataFrame(df_dict)
            if scp.offset is not None:
                df["storage_byte"] = scp.offset

            if scp.range_spec is not None:
                df["start"] = [r[0] for r in scp.range_spec]
                df["num_frames"] = [r[1] for r in scp.range_spec]

            return cls(df)

        return super().load(file_path, sep)
