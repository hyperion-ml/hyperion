"""
Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

from jsonargparse import ActionYesNo, ArgumentParser
from tqdm import tqdm

from ..utils import PathLike, RecordingSet


class DataPrep:
    """
    Base class for data preparation. Handles parallel audio processing and metadata generation.

    Attributes:
        corpus_dir (Path): Path to input data directory.
        output_dir (Path): Path to output data directory.
        use_kaldi_ids (bool): Whether to prefix segment IDs with speaker IDs (Kaldi style).
        target_sample_freq (int): Target audio sampling frequency.
        num_threads (int): Number of threads for parallel processing.
    """

    registry = {}

    def __init__(
        self,
        corpus_dir: PathLike,
        output_dir: PathLike,
        use_kaldi_ids: bool = False,
        target_sample_freq: Optional[int] = None,
        num_threads: int = 10,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.use_kaldi_ids = use_kaldi_ids
        self.target_sample_freq = target_sample_freq
        self.num_threads = num_threads

        self.output_dir.mkdir(exist_ok=True, parents=True)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry[cls.dataset_name()] = cls

    @staticmethod
    def dataset_name() -> str:
        """Returns a unique identifier for the dataset."""
        raise NotImplementedError("Subclasses must implement dataset_name().")

    @staticmethod
    def _get_recording_duration(
        recordings: RecordingSet, i: int, n: int
    ) -> Tuple[List[float], List[float]]:
        """Helper function to calculate duration and sample rate for audio chunks."""
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

    def get_recording_duration(self, recording_set: RecordingSet):
        """
        Computes and appends duration and sampling frequency for each recording.

        Args:
            recording_set (dict): A dictionary expected to be updated with 'duration' and 'sample_freq'.
        """
        import itertools

        futures = []
        logging.info("submitting threats...")
        with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
            for i in tqdm(range(self.num_threads)):
                future = pool.submit(
                    DataPrep._get_recording_duration, recording_set, i, self.num_threads
                )
                futures.append(future)

        logging.info("waiting threats...")
        res = [f.result() for f in tqdm(futures)]
        fss = list(itertools.chain(*[r[0] for r in res]))
        durations = list(itertools.chain(*[r[1] for r in res]))

        recording_set["duration"] = durations
        recording_set["sample_freq"] = fss

    @staticmethod
    def add_class_args(parser: ArgumentParser) -> None:
        """
        Adds command-line arguments to the parser for configuring DataPrep.

        Args:
            parser (ArgumentParser): Argument parser to which arguments will be added.
        """
        parser.add_argument(
            "--corpus-dir", required=True, help="Path to the input corpus directory"
        )
        parser.add_argument(
            "--output-dir",
            required=True,
            help="Path to the output directory where prepared data will be saved",
        )
        parser.add_argument(
            "--use-kaldi-ids",
            default=False,
            action=ActionYesNo,
            help="If True, prefixes segment IDs with speaker IDs (Kaldi style).",
        )
        parser.add_argument(
            "--target-sample-freq",
            default=None,
            type=int,
            help="Target sampling frequency to which audio files should be converted.",
        )
        parser.add_argument(
            "--num-threads",
            default=10,
            type=int,
            help="Number of parallel threads to use for audio processing.",
        )
