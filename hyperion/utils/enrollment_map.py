"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import re
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from .list_utils import split_list, split_list_group_by_key
from .info_table import InfoTable


class EnrollmentMap(InfoTable):
    """Class to store the mapping between enrollment id
       and segmentids
    """

    def __init__(self, df):
        if "modelid" in df:
            df.rename(columns={"modelid": "id"}, inplace=True)
        super().__init__(df)

    def split(self, idx, num_parts):
        """Splits the mapping into num_parts and return part idx.

        Args:
          idx: Part to return from 1 to num_parts.
          num_parts: Number of parts to split the list.
          group_by: All the lines with the same value in column
                          groub_by_field go to the same part

        Returns:
          Sub InfoTable object
        """
        _, idx1 = split_list_group_by_key(self.df["id"], idx, num_parts)

        df = self.df.iloc[idx1]
        return EnrollmentMap(df)

    def save(self, file_path, sep=None, nist_compatible=True):
        if nist_compatible:
            # For compatibility with NIST SRE files the index column "id"
            # is saved as modelid
            self.df.rename(columns={"id": "modelid"}, inplace=True)

        super().save(file_path, sep)
        if nist_compatible:
            self.df.rename(columns={"modelid": "id"}, inplace=True)

    @classmethod
    def load(cls, file_path, sep=None):
        """Loads EnrollmentMap from file.

        Args:
          file_path: File to read the list.
          sep: Separator between the key and file_path in the text file.
          dtype: Dictionary with the dtypes of each column.
          name: name for the data to be loaded
        Returns:
          EnrollmentMap object
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
                dtype={"segmentid": np.str, "modelid": np.str},
            )
            df = df[["modelid", "segmentid"]]
        else:
            if sep is None:
                sep = "\t" if ".tsv" in ext else ","

            df = pd.read_csv(file_path, sep=sep)

        return cls(df)
