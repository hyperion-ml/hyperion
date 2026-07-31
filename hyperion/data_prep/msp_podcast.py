"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..utils import ClassInfo, HyperDataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike
from .data_prep import DataPrep


class MSPPodcastDataPrep(DataPrep):
    """
    Prepares the MSP-Podcast dataset into structured tables.

    Supports filtering by subset (e.g., 'train', 'test', 'eval') and generates
    segment, recording, and speaker metadata for each subset.

    Attributes:
        corpus_dir (PathLike): Root directory of the MSP-Podcast corpus.
        subset (str): Subset to prepare (e.g., 'train', 'test', 'eval', 'all').
        output_dir (PathLike): Where to write processed dataset.
        use_kaldi_ids (bool): Whether to prepend speaker ID to segment ID.
        target_sample_freq (Optional[int]): Resampling target frequency.
        num_threads (int): Number of threads for audio processing.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        subset: str,
        output_dir: PathLike,
        use_kaldi_ids: bool = False,
        target_sample_freq: Optional[int] = None,
        num_threads: int = 10,
    ):
        super().__init__(
            corpus_dir, output_dir, use_kaldi_ids, target_sample_freq, num_threads
        )
        self.subset = subset.lower()

    @staticmethod
    def dataset_name() -> str:
        return "msp_podcast"

    @staticmethod
    def add_class_args(parser) -> None:
        """
        Adds MSP-Podcast-specific arguments to the CLI parser.

        Args:
            parser: ArgumentParser object.
        """
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--subset",
            choices=["train", "dev", "test", "eval", "all"],
            default="all",
            help="Subset of MSP-Podcast to prepare (default: all).",
        )

    def prepare(self) -> None:
        """
        Runs the complete MSP-Podcast data preparation pipeline:
        - Loads metadata from CSV
        - Filters by subset (if specified)
        - Builds SegmentSet, RecordingSet, ClassInfo
        - Saves the dataset
        """
        logging.info("Preparing MSP-Podcast subset=%s", self.subset)

        metadata_file = self.corpus_dir / "labels.csv"
        audio_root = self.corpus_dir / "Audio"

        if not metadata_file.exists():
            metadata_file = self.corpus_dir / "segments.csv"
            assert metadata_file.exists(), "Missing both labels.csv and segments.csv"

        logging.info("Loading metadata from %s", metadata_file)
        df = pd.read_csv(metadata_file)

        # Subset filtering if applicable
        if self.subset != "all" and "Split_Set" in df.columns:
            df = df[df["Split_Set"].str.lower() == self.subset]

        assert "FileName" in df.columns, "Missing 'FileName' column in metadata"
        df["id"] = df["FileName"].apply(lambda x: Path(x).with_suffix("").name)
        df["storage_path"] = df["FileName"].apply(
            lambda x: str((audio_root / x).resolve())
        )

        if "Speaker" in df.columns:
            df["speaker"] = df["Speaker"].astype(str)
        else:
            df["speaker"] = "msp"

        if self.use_kaldi_ids:
            df["id"] = df.apply(lambda row: f"{row['speaker']}-{row['id']}", axis=1)

        recs = pd.DataFrame({"id": df["id"], "storage_path": df["storage_path"]})
        recs = RecordingSet(recs)
        recs.get_durations(self.num_threads)

        if self.target_sample_freq:
            recs["target_sample_freq"] = self.target_sample_freq

        df["duration"] = df["id"].map(recs.set_index("id")["duration"])

        segments = SegmentSet(df[["id", "speaker", "duration"]])
        segments.sort()

        speakers = ClassInfo(pd.DataFrame({"id": df["speaker"].unique()}))

        dataset = HyperDataset(
            segments=segments,
            recordings=recs,
            classes={"speaker": speakers},
        )
        dataset.save(self.output_dir)

        logging.info(
            "Dataset contains %d segments, %d speakers", len(segments), len(speakers)
        )
