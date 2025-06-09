"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import glob
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..utils import ClassInfo, HypDataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike
from .data_prep import DataPrep


class MLSDatasetDataPrep(DataPrep):
    """
    Prepares the MLS (Multilingual LibriSpeech) dataset into structured tables for training or evaluation.

    This class reads the metadata from `.tsv` files, matches it with audio recordings,
    extracts durations, and builds `RecordingSet`, `SegmentSet`, and class tables for speaker and language.

    Attributes:
        corpus_dir (PathLike): Path to the root MLS directory (e.g., contains folders like 'en/', 'fr/').
        language (str): Language code (e.g., 'en', 'de', 'fr') corresponding to the subdirectory in corpus_dir.
        split (str): Data split to use: 'train', 'dev', or 'test'.
        output_dir (PathLike): Directory to save the processed dataset files.
        use_kaldi_ids (bool): Whether to prepend speaker ID to each segment ID.
        target_sample_freq (Optional[int]): Optional sample rate to convert audio to.
        num_threads (int): Number of parallel threads for duration extraction.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        language: str,
        split: str,
        output_dir: PathLike,
        use_kaldi_ids: bool = False,
        target_sample_freq: Optional[int] = None,
        num_threads: int = 10,
    ) -> None:
        """
        Initializes the MLS dataset preparation class.

        Args:
            corpus_dir (PathLike): Base directory containing the MLS dataset.
            language (str): Language code (e.g., 'en').
            split (str): Dataset split: 'train', 'dev', or 'test'.
            output_dir (PathLike): Directory where processed output will be saved.
            use_kaldi_ids (bool): Whether to prepend speaker ID to segment IDs.
            target_sample_freq (Optional[int]): Resample audio to this frequency if specified.
            num_threads (int): Number of threads for parallel audio duration extraction.
        """
        super().__init__(
            corpus_dir, output_dir, use_kaldi_ids, target_sample_freq, num_threads
        )
        self.language = language
        self.split = split

    @staticmethod
    def dataset_name() -> str:
        """
        Returns:
            str: Dataset name identifier used in metadata.
        """
        return "mls"

    @staticmethod
    def add_class_args(parser) -> None:
        """
        Adds MLS-specific arguments to a CLI parser.

        Args:
            parser (ArgumentParser): The argument parser to which MLS-specific arguments will be added.
        """
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--language",
            type=str,
            required=True,
            help="MLS language code (e.g., 'en', 'de').",
        )
        parser.add_argument(
            "--split",
            choices=["train", "dev", "test"],
            required=True,
            help="Dataset split.",
        )

    def prepare(self) -> None:
        """
        Executes the full MLS data preparation pipeline:

        - Reads metadata from the TSV file (e.g., 'train.tsv').
        - Builds full paths to audio files and validates them.
        - Extracts audio durations using parallel processing.
        - Constructs RecordingSet, SegmentSet, and ClassInfo tables.
        - Saves the dataset to the specified output directory.
        """
        logging.info(
            "Preparing MLS dataset lang=%s split=%s corpus_dir=%s -> data_dir=%s",
            self.language,
            self.split,
            self.corpus_dir,
            self.output_dir,
        )

        base_dir = self.corpus_dir / self.language / self.split
        tsv_file = base_dir / f"{self.split}.tsv"
        audio_dir = base_dir / "audio"

        assert tsv_file.is_file(), f"Missing transcript file: {tsv_file}"
        assert audio_dir.is_dir(), f"Missing audio directory: {audio_dir}"

        # Load metadata
        df = pd.read_csv(tsv_file, sep="\t")
        df["speaker"] = df["client_id"].apply(lambda x: f"mls-{x}")
        df["id"] = df["path"].apply(lambda x: Path(x).with_suffix("").name)
        df["id"] = "mls-" + df["id"]

        if self.use_kaldi_ids:
            df["id"] = df.apply(lambda row: f"{row['speaker']}-{row['id']}", axis=1)

        # Build audio paths
        df["storage_path"] = df["path"].apply(lambda p: str(audio_dir / p))

        logging.info("Creating RecordingSet")
        recs = pd.DataFrame({"id": df["id"], "storage_path": df["storage_path"]})
        recs = RecordingSet(recs)
        recs.get_durations(self.num_threads)
        if self.target_sample_freq:
            recs["target_sample_freq"] = self.target_sample_freq

        logging.info("Creating SegmentsSet")
        df["duration"] = df["id"].map(recs.set_index("id")["duration"])
        df["language"] = self.language
        segments = SegmentSet(df[["id", "speaker", "text", "duration", "language"]])
        segments.sort()

        logging.info("Creating ClassInfo tables")
        speakers = ClassInfo(pd.DataFrame({"id": df["speaker"].unique()}))
        languages = ClassInfo(pd.DataFrame({"id": [self.language]}))

        logging.info("Saving dataset")
        dataset = HypDataset(
            segments=segments,
            recordings=recs,
            classes={"speaker": speakers, "language": languages},
        )
        dataset.save(self.output_dir)
        logging.info(
            "Dataset contains %d segments, %d speakers",
            len(segments),
            len(speakers),
        )
