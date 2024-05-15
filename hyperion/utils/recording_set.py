"""
 Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
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

    @staticmethod
    def _get_durations(recordings, i, n):
        from ..io import SequentialAudioReader as AR

        durations = []
        fss = []
        with AR(recordings, part_idx=i + 1, num_parts=n) as reader:
            for data in reader:
                key, x, fs = data
                duration = x.shape[0] / fs
                fss.append(fs)
                durations.append(duration)

        return fss, durations

    def get_durations(self, num_threads: int = 16):

        import itertools
        from concurrent.futures import ThreadPoolExecutor

        from tqdm import tqdm

        futures = []
        num_threads = min(num_threads, len(self.df))
        logging.info("submitting threats...")
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            for i in tqdm(range(num_threads)):
                future = pool.submit(RecordingSet._get_durations, self, i, num_threads)
                futures.append(future)

        logging.info("waiting threats...")
        res = [f.result() for f in tqdm(futures)]
        fss = list(itertools.chain(*[r[0] for r in res]))
        durations = list(itertools.chain(*[r[1] for r in res]))

        self.df["duration"] = durations
        self.df["sample_freq"] = fss
