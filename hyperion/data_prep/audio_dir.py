"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import ast
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from jsonargparse import ActionYesNo, ArgumentParser

from ..utils import ClassInfo, HyperDataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike
from .data_prep import DataPrep


class AudioDir(DataPrep):
    """
    Prepares a dataset from a directory of audio files (wav, flac, mp3, ogg).

    Attributes:
        corpus_dir (PathLike): Input directory with audio files.
        output_dir (PathLike): Output directory to save the dataset.
        prefix (Optional[str]): Optional prefix added to recording/segment IDs.
        metadata (Optional[Dict[str, str]]): Global metadata applied to all segments.
        use_kaldi_ids (bool): Whether to prepend speaker ID to segment ID.
        target_sample_freq (Optional[int]): Target sample rate for audio files.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        output_dir: PathLike,
        prefix: Optional[str] = None,
        metadata: Union[str, Dict[str, str], None] = None,
        use_kaldi_ids: bool = False,
        target_sample_freq: Optional[int] = None,
        num_threads: int = 10,
    ):
        """
        Initialize the AudioDir data preparation class.

        Args:
            corpus_dir (PathLike): Directory containing raw audio.
            output_dir (PathLike): Directory to store the output dataset.
            prefix (Optional[str]): Optional prefix to prepend to recording IDs.
            metadata (str | dict | None): Metadata to apply to all segments.
            use_kaldi_ids (bool): Whether to use Kaldi-style IDs.
            target_sample_freq (Optional[int]): Target sample rate.
            num_threads (int): Number of threads for duration calculation.
        """
        super().__init__(corpus_dir, output_dir, False, target_sample_freq, num_threads)
        if isinstance(metadata, str):
            metadata = ast.literal_eval(metadata)

        self.prefix = prefix
        self.metadata = metadata

    @staticmethod
    def dataset_name() -> str:
        """Returns a name identifier for the dataset."""
        return "audio-dir"

    @staticmethod
    def add_class_args(parser: ArgumentParser) -> None:
        """
        Adds command-line arguments for AudioDir data preparation.

        Args:
            parser: Argument parser to which the options are added.
        """
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--prefix",
            default=None,
            help="""optional prefix for recording/segment ids""",
        )
        parser.add_argument(
            "--metadata",
            default=None,
            help="""metadata applied to all files in the directory, as str describing a dictionary like {'speaker': 'spk1', 'language':'eng', 'age':20}""",
        )

    def prepare(self):
        """
        Prepares a dataset from audio files found in the corpus directory.
        Creates RecordingSet and SegmentSet, assigns metadata, and saves the dataset.
        """
        logging.info(
            "Peparing Audio Directory corpus_dir:%s -> data_dir:%s",
            self.corpus_dir,
            self.output_dir,
        )

        rec_dir = self.corpus_dir
        logging.info("searching audio files in %s", str(rec_dir))
        rec_files = []

        # Collect all matching extensions
        for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
            rec_files.extend(rec_dir.rglob(ext))  # Use glob() for non-recursive

        # Optionally sort or filter
        rec_files = sorted(rec_files)

        assert len(rec_files) > 0, "recording files not found"

        if self.prefix is None:
            rec_ids = [f.with_suffix("").name for f in rec_files]
        else:
            rec_ids = [self.prefix + f.with_suffix("").name for f in rec_files]

        if self.use_kaldi_ids and self.metadata and "speaker" in self.metadata:
            rec_ids = [f"{s}-{f}" for f, s in zip(rec_ids, self.metadata["speaker"])]

        file_paths = [str(r) for r in rec_files]
        logging.info("making RecordingSet")
        recs = pd.DataFrame({"id": rec_ids, "storage_path": file_paths})
        recs = RecordingSet(recs)
        recs.sort()

        logging.info("getting recording durations")
        recs.get_durations(self.num_threads)
        if self.target_sample_freq:
            recs["target_sample_freq"] = self.target_sample_freq

        logging.info("making SegmentsSet")
        df_segs = pd.DataFrame({"id": rec_ids})
        df_segs["duration"] = recs.loc[df_segs["id"], "duration"].values
        if self.metadata is not None:
            for k, v in self.metadata.items():
                df_segs[k] = v
        segments = SegmentSet(df_segs)
        segments.sort()

        logging.info("making dataset")
        dataset = HyperDataset(
            segments,
            recordings=recs,
        )
        logging.info("saving dataset at %s", self.output_dir)
        dataset.save(self.output_dir)
        logging.info("datasets containts %d segments", len(segments))
