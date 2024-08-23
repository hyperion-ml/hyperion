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

from .info_table import InfoTable
from .list_utils import split_list, split_list_group_by_key


class EnrollmentMap(InfoTable):
    """Class to store the mapping between enrollment id
    and segmentids
    """

    def __init__(self, df):
        if "modelid" in df:
            df.rename(columns={"modelid": "id"}, inplace=True)
        assert "segmentid" in df
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
                dtype={"segmentid": str, "modelid": str},
            )
            df = df[["modelid", "segmentid"]]
        else:
            if sep is None:
                sep = "\t" if ".tsv" in ext else ","

            df = pd.read_csv(file_path, sep=sep)

        return cls(df)

    @classmethod
    def cat(cls, tables):
        """Concatenates several tables.

        Args:
          tables: List of InfoTables

        Returns:
          InfoTable object concatenating the tables.
        """
        df_list = [table.df for table in tables]
        df = pd.concat(df_list)
        return cls(df)

    def model_idx(self, modelids=None):
        """Returns mapping from segments to model indexes

        Args:
          modelids: sorted list model ids, used to assign integer indexes
            to ids. If None, modelids are sorted alphabetically with np.unique

        Return:
           If modelids is None, np.array sorted unique modelids, and
           np.array mapping segments to model indexes.
           otherwise, just them model indexes
        """
        if modelids is None:
            return np.unique(self.df["id"], return_inverse=True)

        enroll_idx = np.zeros((len(self.df)), dtype=int)
        for i, modelid in enumerate(modelids):
            idx = self.df["id"] == modelid
            enroll_idx[idx] = i

        return enroll_idx
