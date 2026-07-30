"""
Copyright 2022 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import multiprocessing
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio

from .info_table import InfoTable
from .misc import PathLike

T = TypeVar("T", bound="RecordingSet")


class RecordingSet(InfoTable):
    """
    InfoTable specialization for audio-recording manifests.

    The table must contain ``id`` and ``storage_path`` columns.

    Examples:
        >>> import pandas as pd
        >>> from hyperion.utils.recording_set import RecordingSet
        >>> df = pd.DataFrame({"id": ["utt1"], "storage_path": ["/audio/utt1.wav"]})
        >>> recs = RecordingSet(df)
        >>> recs.is_valid_df(recs.df)
        True
        >>> recs2 = recs.filter(items=["utt1"])
        >>> len(recs2)
        1
    """

    def __init__(self, df: Union[pd.DataFrame, T]) -> None:
        """
        Initialize a recording set.

        Args:
            df (pd.DataFrame or RecordingSet): Input metadata table.
        """
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

    def save(self, file_path: PathLike, sep: Optional[str] = None) -> None:
        """
        Save the recording manifest to disk.

        Args:
            file_path (PathLike): Output file path.
            sep (Optional[str]): Delimiter for non-``.scp`` files.
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
    def load(cls: Type[T], file_path: PathLike, sep: Optional[str] = None) -> T:
        """
        Load a recording manifest from disk.

        Args:
            file_path (PathLike): Input file path.
            sep (Optional[str]): Delimiter for non-``.scp`` files.

        Returns:
            RecordingSet: Loaded recording set.
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
    def _get_durations_old(
        recordings: "RecordingSet", i: int, n: int
    ) -> Tuple[List[int], List[float]]:
        """
        Legacy duration extraction helper based on sequential audio reads.

        Args:
            recordings (RecordingSet): Source recordings table.
            i (int): 1-based partition index.
            n (int): Number of partitions.

        Returns:
            Tuple[List[int], List[float]]: Sample rates and durations.
        """
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
    def _get_durations(
        recordings: "RecordingSet",
        i: int,
        n: int,
        progress: Optional[Any] = None,
        report_every: int = 1000,
    ) -> Tuple[List[str], List[int], List[float]]:
        """
        Duration extraction helper with file-header and fallback decoding logic.

        Args:
            recordings (RecordingSet): Source recordings table.
            i (int): 1-based partition index.
            n (int): Number of partitions.
            progress (Optional[Any]): Shared counter proxy with a ``value`` field.
            report_every (int): Counter update interval.

        Returns:
            Tuple[List[str], List[int], List[float]]: Recording IDs, sample rates,
            and durations.
        """
        from ..io import RandomAccessAudioReader as AR

        ids = []
        durations = []
        fss = []
        recordings = recordings.split(i, n)
        processed = 0
        reported = 0
        with AR(recordings=recordings) as reader:
            for _id, audio_file in zip(recordings["id"], recordings["storage_path"]):
                num_samples = None
                sample_rate = None
                suffix = Path(audio_file).suffix.lower()
                if suffix in {".wav", ".flac"}:
                    try:
                        info = sf.info(audio_file)
                        num_samples = info.frames
                        sample_rate = info.samplerate
                    except Exception:
                        num_samples = None
                        sample_rate = None

                if num_samples is None or sample_rate is None or sample_rate <= 0:
                    try:
                        info = torchaudio.info(audio_file)
                        num_samples = info.num_frames
                        sample_rate = info.sample_rate

                    except Exception:
                        num_samples = None
                        sample_rate = None

                if num_samples is None or sample_rate is None or sample_rate <= 0:
                    x, fs = reader.read(_id)
                    num_samples = x[0].shape[0]
                    sample_rate = fs[0]

                if sample_rate is None or sample_rate <= 0:
                    raise ValueError(
                        f"Invalid sample rate {sample_rate} for recording '{_id}' at '{audio_file}'"
                    )

                duration = num_samples / sample_rate
                ids.append(_id)
                fss.append(sample_rate)
                durations.append(duration)
                processed += 1
                if (
                    progress is not None
                    and report_every > 0
                    and processed - reported >= report_every
                ):
                    increment = processed - reported
                    progress.value += increment
                    reported += increment

        if progress is not None and processed > reported:
            increment = processed - reported
            progress.value += increment

        return ids, fss, durations

    # def get_durations_old(self, num_threads: int = 16) -> None:
    #     """
    #     Estimate recording duration and sample rate with a thread pool.

    #     Args:
    #         num_threads (int): Number of worker threads.
    #     """
    #     import itertools
    #     from concurrent.futures import ThreadPoolExecutor, as_completed

    #     from tqdm import tqdm

    #     futures = []
    #     num_threads = min(num_threads, len(self.df))
    #     logging.info("submitting threads...")

    #     with ThreadPoolExecutor(max_workers=num_threads) as pool:
    #         for i in tqdm(range(num_threads)):
    #             future = pool.submit(RecordingSet._get_durations, self, i, num_threads)
    #             futures.append(future)

    #     logging.info("waiting threads...")
    #     for handler in logging.getLogger().handlers:
    #         handler.flush()
    #     res = []
    #     for f in tqdm(as_completed(futures), total=len(futures)):
    #         res.append(f.result())

    #     fss = list(itertools.chain.from_iterable(r[0] for r in res))
    #     durations = list(itertools.chain.from_iterable(r[1] for r in res))

    #     self.df["duration"] = durations
    #     self.df["sample_freq"] = fss

    def get_durations(self, num_threads: int = 16, report_every: int = 5000) -> None:
        """
        Estimate recording duration and sample rate with a process pool.

        This version periodically reports progress and writes ``duration`` and
        ``sample_freq`` columns back into the table.

        Args:
            num_threads (int): Number of worker processes.
            report_every (int): Progress update interval in processed recordings.
        """
        import itertools
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from tqdm import tqdm

        total = len(self.df)
        if total == 0:
            logging.info("No recordings available; skipping duration estimation.")
            return

        if num_threads < 1:
            raise ValueError("num_threads must be >= 1")

        num_threads = min(num_threads, total)

        manager = multiprocessing.Manager()
        progress = manager.Value("i", 0)
        stop_event = threading.Event()
        progress_interval = 60.0

        def heartbeat() -> None:
            while not stop_event.wait(progress_interval):
                value = progress.value
                percent = (100.0 * value / total) if total else 0.0
                logging.info(
                    "Duration estimation progress: %d/%d recordings (%.1f%%)",
                    value,
                    total,
                    percent,
                )
            value = progress.value
            percent = (100.0 * value / total) if total else 0.0
            logging.info(
                "Duration estimation progress: %d/%d recordings (%.1f%%)",
                value,
                total,
                percent,
            )

        heartbeat_thread = threading.Thread(
            target=heartbeat, name="duration-heartbeat", daemon=True
        )
        heartbeat_thread.start()

        futures = []
        logging.info("submitting threads...")

        try:
            with ProcessPoolExecutor(max_workers=num_threads) as pool:
                for i in tqdm(range(num_threads), desc="Submitting threads"):
                    future = pool.submit(
                        RecordingSet._get_durations,
                        self,
                        i + 1,
                        num_threads,
                        progress,
                        report_every,
                    )
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
        finally:
            stop_event.set()
            heartbeat_thread.join()
            manager.shutdown()

        # Unpack and flatten
        ids = list(itertools.chain.from_iterable(r[0] for r in res))
        fss = list(itertools.chain.from_iterable(r[1] for r in res))
        durations = list(itertools.chain.from_iterable(r[2] for r in res))

        self.df.loc[ids, "duration"] = durations
        self.df.loc[ids, "sample_freq"] = fss
        self.df["sample_freq"] = self.df["sample_freq"].astype("Int64")

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
