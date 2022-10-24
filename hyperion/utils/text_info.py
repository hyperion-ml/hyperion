"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .info_table import InfoTable


def read_2column_text(path: Union[Path, str]) -> Dict[str, str]:
    """Read a text file having 2 column as dict object.

    Examples:
        wav.scp:
            key1 /some/path/a.wav
            key2 /some/path/b.wav

        >>> read_2column_text('wav.scp')
        {'key1': '/some/path/a.wav', 'key2': '/some/path/b.wav'}

    """
    assert check_argument_types()

    data = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.rstrip().split(maxsplit=1)
            if len(sps) == 1:
                k, v = sps[0], ""
            else:
                k, v = sps
            if k in data:
                raise RuntimeError(f"{k} is duplicated ({path}:{linenum})")
            data[k] = v
    return data



class TextInfo(InfoTable):
    def __init__(self, df):
        super().__init__(df)
        if "weights" not in self.df:
            self.set_uniform_weights()
        else:
            self.df["weights"] /= self.df["weigths"].sum()

    def set_uniform_weights(self):
        self.df["weights"] = 1 / len(self.df)

    def set_weights(self, weights):
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
    def load(cls, file_path, sp, sep=None):
        """Loads utt2info list from text file.

        Args:
          file_path: File to read the list.
          sp: SentencePieceProcessor from the BPE model
          sep: Separator between the key and file_path in the text file.
          dtype: Dictionary with the dtypes of each column.
        Returns:
          Utt2Info object
        """
        #TODO: load text information
        """Loads utt2info list from text file.

        Args:
          file_path: File to read the list.
          sp: SentencePieceProcessor for bpe.
        Returns:
          Utt2Info object
        """            
        # # y: k2.RaggedTensor,
        # # A ragged tensor with 2 axes [utt][label]. It contains labels of each utterance.
        # y = sp.encode(texts, out_type=int)
        # y = k2.RaggedTensor(y).to(device)
        file_path = Path(file_path)
        text_df = super().load(file_path, sep, name="text_label")
        # for i, text in enumerate(text_df["text_label"]):
        #     y = sp.encode(text, out_type=int)
        #     y = k2.RaggedTensor(y).to(device)
        #     text_df["text_label"][i] = y

        return text_df
