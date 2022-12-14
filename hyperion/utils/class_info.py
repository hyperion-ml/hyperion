"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
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

    def set_zero_weight(self, id):
        self.df.loc[id, "weights"] = 0
        self.df["weights"] /= self.df["weights"].sum()

    @property
    def weights(self, id):
        return self.df.loc[id, "weights"]

    @property
    def num_classes(self):
        return self.df["class_idx"].values.max() + 1

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
                file_path, sep=" ", header=None, names=["id"], dtype={"id": np.str},
            )
            return cls(df)

        return super().load(file_path, sep)
