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
from jsonargparse import ActionYesNo
from tqdm import tqdm

from ..utils import HypDataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike, urlretrieve_progress
from .data_prep import DataPrep


class RIRSDataPrep(DataPrep):
    """
    Prepares the RIRS (Room Impulse Response and Noise) dataset into structured tables.

    This class supports both directory-based discovery of RIR audio files and reading
    from a predefined `rir_list` file if it exists in the corpus directory.

    Attributes:
        corpus_dir (PathLike): Root directory of the RIRS dataset.
        output_dir (PathLike): Directory where the processed dataset will be saved.
        target_sample_freq (Optional[int]): If specified, audio will be resampled to this frequency.
        num_threads (int): Number of threads for audio duration extraction.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        output_dir: PathLike,
        target_sample_freq: Optional[int] = None,
        num_threads: int = 10,
        **kwargs,
    ):
        """
        Initializes the RIRS data preparation object.

        Args:
            corpus_dir (PathLike): Directory containing RIR audio or metadata.
            output_dir (PathLike): Destination for the structured dataset.
            target_sample_freq (Optional[int]): Desired sample frequency in Hz (if resampling).
            num_threads (int): Number of parallel threads for duration computation.
            **kwargs: Additional arguments passed to the base class.
        """
        super().__init__(corpus_dir, output_dir, False, target_sample_freq, num_threads)

    @staticmethod
    def dataset_name() -> str:
        """
        Returns:
            str: Identifier name for this dataset ("rirs").
        """
        return "rirs"

    @staticmethod
    def add_class_args(parser) -> None:
        """
        Adds RIRS-specific arguments to the CLI parser.

        Args:
            parser: Argument parser to which RIRS options will be added.
        """
        DataPrep.add_class_args(parser)

    def prepare(self) -> None:
        """
        Executes the full RIRS preparation pipeline:
        - Loads file list from rir_list or recursively finds all .wav files
        - Extracts durations
        - Builds SegmentSet and RecordingSet
        - Saves a HypDataset to the output directory
        """
        logging.info(
            "Peparing RIRS corpus_dir:%s -> data_dir:%s",
            self.corpus_dir,
            self.output_dir,
        )
        rec_dir = self.corpus_dir
        assert rec_dir.is_dir(), f"Recording directory not found: {rec_dir}"
        rirs_file = self.corpus_dir / "rir_list"
        if rirs_file.exists():
            rirs_table = pd.read_csv(
                rirs_file,
                sep=" ",
                header=None,
                names=["dummy1", "rir_id", "dummy2", "room_id", "rec_files"],
            )
            rec_files = [Path(f) for f in rirs_table["rec_files"].values]
            room_ids = rirs_table["room_id"].values
        else:
            logging.info("searching audio files in %s", str(rec_dir))
            rec_files = list(rec_dir.glob("**/*.wav"))
            room_ids = None
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

        # logging.info("getting recording durations")
        # self.get_recording_duration(recs)
        if self.target_sample_freq:
            recs["target_sample_freq"] = self.target_sample_freq

        logging.info("making SegmentsSet")
        segments = pd.DataFrame(
            {
                "id": rec_ids,
                "duration": recs.loc[rec_ids, "duration"].values,
            }
        )
        if room_ids is not None:
            assert len(room_ids) == len(
                rec_ids
            ), "Mismatch between room_ids and recordings"
            segments["room_id"] = room_ids
        segments = SegmentSet(segments)
        segments.sort()
        logging.info("making dataset")
        dataset = HypDataset(
            segments,
            recordings=recs,
        )
        logging.info("saving dataset at %s", self.output_dir)
        dataset.save(self.output_dir)
        logging.info(
            "datasets containts %d segments",
            len(segments),
        )
