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

    @staticmethod
    def is_valid_df(df: pd.DataFrame) -> bool:
        """
        Check if the DataFrame is valid for InfoTable.

        Args:
            df (pd.DataFrame): DataFrame to check.

        Returns:
            bool: True if valid, False otherwise.
        """
        return "id" in df and "storage_path" in df

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
    def _get_durations_old(recordings, i, n):
        from ..io import SequentialAudioReader as AR

        durations = []
        fss = []
        with AR(recordings=recordings, part_idx=i + 1, num_parts=n) as reader:
            for data in reader:
                key, x, fs = data
                duration = x.shape[0] / fs
                fss.append(fs)
                durations.append(duration)

        return fss, durations

    @staticmethod
    def _get_durations(recordings, i, n):
        import torchaudio

        from ..io import RandomAccessAudioReader as AR

        ids = []
        durations = []
        fss = []
        recordings = recordings.split(i, n)
        with AR(recordings=recordings) as reader:
            for _id, audio_file in zip(recordings["id"], recordings["storage_path"]):
                try:
                    info = torchaudio.info(audio_file)
                    num_samples = info.num_frames
                    sample_rate = info.sample_rate
                except Exception as e:
                    x, fs = reader.read(_id)
                    num_samples = x[0].shape[0]
                    sample_rate = fs[0]

                duration = num_samples / sample_rate
                ids.append(_id)
                fss.append(sample_rate)
                durations.append(duration)

        return ids, fss, durations

    def get_durations_old(self, num_threads: int = 16):
        import itertools
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from tqdm import tqdm

        futures = []
        num_threads = min(num_threads, len(self.df))
        logging.info("submitting threads...")

        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            for i in tqdm(range(num_threads)):
                future = pool.submit(RecordingSet._get_durations, self, i, num_threads)
                futures.append(future)

        logging.info("waiting threads...")
        for handler in logging.getLogger().handlers:
            handler.flush()
        res = []
        for f in tqdm(as_completed(futures), total=len(futures)):
            res.append(f.result())

        fss = list(itertools.chain.from_iterable(r[0] for r in res))
        durations = list(itertools.chain.from_iterable(r[1] for r in res))

        self.df["duration"] = durations
        self.df["sample_freq"] = fss

    def get_durations(self, num_threads: int = 16):
        import itertools
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from tqdm import tqdm

        futures = []
        num_threads = min(num_threads, len(self.df))
        logging.info("submitting threads...")

        with ProcessPoolExecutor(max_workers=num_threads) as pool:
            for i in tqdm(range(num_threads), desc="Submitting threads"):
                future = pool.submit(RecordingSet._get_durations, self, i, num_threads)
                futures.append(future)

            logging.info("waiting threads...")
            for handler in logging.getLogger().handlers:
                handler.flush()
            res = []
            for f in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Receiving results",
            ):
                res.append(f.result())

        # Unpack and flatten
        ids = list(itertools.chain.from_iterable(r[0] for r in res))
        fss = list(itertools.chain.from_iterable(r[1] for r in res))
        durations = list(itertools.chain.from_iterable(r[2] for r in res))

        self.df.loc[ids, "duration"] = durations
        self.df.loc[ids, "sample_freq"] = fss

        # import itertools
        # from concurrent.futures import ThreadPoolExecutor

        # from tqdm import tqdm

        # futures = []
        # num_threads = min(num_threads, len(self.df))
        # logging.info("submitting threats...")
        # with ThreadPoolExecutor(max_workers=num_threads) as pool:
        #     for i in tqdm(range(num_threads)):
        #         future = pool.submit(RecordingSet._get_durations, self, i, num_threads)
        #         futures.append(future)

        # logging.info("waiting threats...")
        # res = [f.result() for f in tqdm(futures)]
        # fss = list(itertools.chain(*[r[0] for r in res]))
        # durations = list(itertools.chain(*[r[1] for r in res]))

        # self.df["duration"] = durations
        # self.df["sample_freq"] = fss
