"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .info_table import InfoTable


class RecordingSet(InfoTable):
    def __init__(self, df):
        super().__init__(df)
        assert "storage_path" in df

    def save(self, file_path, sep=None):
        """Saves info table to file

        Args:
          file_path: File to write the list.
          sep: Separator between the key and file_path in the text file.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        ext = file_path.suffix
        if ext == ".scp":
            # if no extension we save as kaldi feats.scp file
            from .scp_list import SCPList

            scp = SCPList(self.df["id"].values, self.df["storage_path"].values)
            scp.save(file_path)
            return

        super().save(file_path, sep)

    @classmethod
    def load(cls, file_path, sep=None):
        """Loads utt2info list from text file.

        Args:
          file_path: File to read the list.
          sep: Separator between the key and file_path in the text file.
        Returns:
          RecordingSet object
        """
        file_path = Path(file_path)
        ext = file_path.suffix
        if ext == ".scp":
            # if no extension we load as kaldi feats.scp file
            from .scp_list import SCPList

            scp = SCPList.load(file_path)
            df_dict = {"id": scp.key, "storage_path": scp.file_path}
            df = pd.DataFrame(df_dict)

            return cls(df)

        return super().load(file_path, sep)
