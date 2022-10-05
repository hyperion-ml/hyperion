"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .info_table import InfoTable


class FeatureSet(InfoTable):
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

            offset = self.df["storage_byte"] if "storage_byte" in self.df else None
            range = None
            if "start" and "num_frames" in self.df:
                range = [
                    np.array([s, n], dtype=np.int64)
                    for s, n in self.df[["start", "num_frames"]]
                ]
            scp = SCPList(
                self.df["id"].values, self.df["storage_path"].values, offset, range
            )
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
          FeatureSet object
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

            if scp.range is not None:
                df["start"] = [r[0] for r in scp.range]
                df["num_frames"] = [r[0] for r in scp.range]

            return cls(df)

        return super().load(file_path, sep)
