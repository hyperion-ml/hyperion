"""
Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import glob
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..utils import HyperDataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike, urlretrieve_progress
from .data_prep import DataPrep


class MusanDataPrep(DataPrep):
    """
    Prepares the MUSAN dataset into structured tables for noise augmentation or speech synthesis.

    This class supports the standard MUSAN subsets:
    - noise
    - music
    - speech

    It builds `RecordingSet` and `SegmentSet` with duration metadata, and stores the
    result in the HyperDataset format.

    Attributes:
        corpus_dir (PathLike): Root directory of the MUSAN dataset.
        subset (str): Subset name to prepare ("noise", "music", or "speech").
        output_dir (PathLike): Directory to save processed dataset files.
        target_sample_freq (Optional[int]): Target sample rate (in Hz) if resampling is desired.
        num_threads (int): Number of parallel threads to use for duration extraction.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        subset: str,
        output_dir: PathLike,
        target_sample_freq: Optional[int] = None,
        num_threads: int = 10,
        **kwargs,
    ):
        """
        Initializes the MUSAN data preparation pipeline.

        Args:
            corpus_dir (PathLike): Input MUSAN corpus directory.
            subset (str): One of 'noise', 'music', or 'speech'.
            output_dir (PathLike): Where to save the prepared dataset.
            target_sample_freq (Optional[int]): Resample frequency if desired.
            num_threads (int): Number of threads for duration extraction.
            **kwargs: Additional keyword arguments (ignored here).
        """
        super().__init__(corpus_dir, output_dir, False, target_sample_freq, num_threads)
        self.subset = subset

    @staticmethod
    def dataset_name() -> str:
        """
        Returns:
            str: Identifier name for this dataset.
        """
        return "musan"

    @staticmethod
    def add_class_args(parser) -> None:
        """
        Adds MUSAN-specific CLI arguments.

        Args:
            parser: An instance of ArgumentParser.
        """
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--subset",
            choices=["noise", "music", "speech"],
            help="""musan subset in [noise, music, speech]""",
            required=True,
        )

    def prepare(self) -> None:
        """
        Executes the full data preparation pipeline for the MUSAN subset:
        - Finds audio files (*.wav)
        - Extracts durations
        - Builds and saves RecordingSet and SegmentSet
        """
        logging.info(
            "Peparing Musan %s corpus_dir:%s -> data_dir:%s",
            self.subset,
            self.corpus_dir,
            self.output_dir,
        )
        rec_dir = self.corpus_dir / self.subset
        assert rec_dir.is_dir(), f"Subset directory not found: {rec_dir}"

        logging.info("searching audio files in %s", str(rec_dir))
        rec_files = list(rec_dir.glob("**/*.wav"))
        if not rec_files:
            # symlinks? try glob
            rec_files = [
                Path(f) for f in glob.iglob(f"{rec_dir}/**/*.wav", recursive=True)
            ]

        assert len(rec_files) > 0, "recording files not found"

        rec_ids = [f.with_suffix("").name for f in rec_files]
        storage_paths = [str(f) for f in rec_files]
        logging.info("making RecordingSet")
        recs = pd.DataFrame({"id": rec_ids, "storage_path": storage_paths})
        recs = RecordingSet(recs)
        recs.get_durations(self.num_threads)
        recs.sort()

        if self.target_sample_freq:
            recs["target_sample_freq"] = self.target_sample_freq

        logging.info("making SegmentsSet")
        segments = pd.DataFrame(
            {
                "id": rec_ids,
                "duration": recs.loc[rec_ids, "duration"].values,
                "noise_type": self.subset,
            }
        )
        segments["original_bandwidth"] = 8000
        segments = SegmentSet(segments)
        segments.sort()
        logging.info("making dataset")
        dataset = HyperDataset(
            segments,
            recordings=recs,
        )
        logging.info("saving dataset at %s", self.output_dir)
        dataset.save(self.output_dir)
        logging.info(
            "datasets containts %d segments",
            len(segments),
        )
