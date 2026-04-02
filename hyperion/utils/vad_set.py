"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from pathlib import Path
from typing import TypeVar, Union

import numpy as np
import pandas as pd

from .feature_set import FeatureSet
from .info_table import InfoTable
from .misc import PathLike

T = TypeVar("T", bound="VADSet")


class VADSet(FeatureSet):
    """
    FeatureSet specialization for voice-activity-detection manifests.

    Examples:
        >>> import pandas as pd
        >>> from hyperion.utils.vad_set import VADSet
        >>> df = pd.DataFrame({"id": ["utt1"], "storage_path": ["vad.ark:10"]})
        >>> vad = VADSet(df)
        >>> list(vad.columns)
        ['id', 'storage_path']
        >>> vad.add_prefix_to_storage_path("/mnt/vad")
        >>> vad.df.loc["utt1", "storage_path"]
        '/mnt/vad/vad.ark:10'
    """

    def __init__(self, df: Union[pd.DataFrame, T]) -> None:
        """
        Initialize a VAD set.

        Args:
            df (pd.DataFrame or VADSet): Input VAD metadata table.
        """
        super().__init__(df)
