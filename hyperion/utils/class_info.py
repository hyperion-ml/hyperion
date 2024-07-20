"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .info_table import InfoTable


class ClassInfo(InfoTable):
    def __init__(self, df):
        super().__init__(df)
        if "class_idx" not in self.df:
            self.add_class_idx()

        if "weights" not in self.df:
            self.set_uniform_weights()
        else:
            self.df["weights"] /= self.df["weights"].sum()

    def add_class_idx(self):
        self.sort()
        self.df["class_idx"] = [i for i in range(len(self.df))]

    def set_uniform_weights(self):
        self.df["weights"] = 1 / len(self.df)

    def set_weights(self, weights):
        self.df["weights"] = weights / weights.sum()

    def renorm_weights(self):
        weights = self.df["weights"]
        self.df["weights"] = weights / weights.sum()

    def exp_weights(self, x):
        weights = self.df["weights"] ** x
        self.set_weights(weights)

    def set_zero_weight(self, ids):
        self.df.loc[ids, "weights"] = 0
        self.df["weights"] /= self.df["weights"].sum()

    @property
    def weights(self, ids):
        return self.df.loc[ids, "weights"]

    @property
    def num_classes(self):
        return self.df["class_idx"].values.max() + 1

    def sort_by_idx(self, ascending=True):
        self.sort("class_idx", ascending)

    @classmethod
    def load(cls, file_path, sep=None):
        """Loads utt2info list from text file.

        Args:
          file_path: File to read the list.
          sep: Separator between the key and file_path in the text file.
          dtype: Dictionary with the dtypes of each column.
        Returns:
          Utt2Info object
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
    def cat(cls, tables):
        """Concatenates several tables.

        Args:
          info_lists: List of InfoTables

        Returns:
          InfoTable object concatenation the info_lists.
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
        self,
        predicate=None,
        items=None,
        iindex=None,
        columns=None,
        by="id",
        keep=True,
        rebuild_idx=False,
    ):
        new_class_info = super().filter(predicate, items, iindex, columns, by, keep)
        if rebuild_idx:
            new_class_info.add_class_idx()

        return new_class_info
