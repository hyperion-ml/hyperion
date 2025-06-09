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
from jsonargparse import ArgumentParser

from ..utils import ClassInfo, HypDataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike
from .data_prep import DataPrep


class LibriLight(DataPrep):
    """
    Prepares the LibriLight dataset into structured tables for speaker recognition.

    This class handles:
    - Audio discovery in specified subset directory
    - Extraction of speaker and book metadata
    - Generation of RecordingSet and SegmentSet
    - Optional Kaldi-style IDs and resampling

    Attributes:
        corpus_dir (PathLike): Input directory containing LibriLight dataset structure.
        subset (str): Dataset subset (e.g., 'small', 'medium', 'large').
        output_dir (PathLike): Output directory to write prepared files.
        use_kaldi_ids (bool): Whether to prepend speaker ID to segment ID.
        target_sample_freq (Optional[int]): Optional resampling target frequency.
        num_threads (int): Number of parallel threads used for duration extraction.
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
        """
        Initializes the LibriLight data preprocessor.
        """
        super().__init__(corpus_dir, output_dir, False, target_sample_freq, num_threads)

        self.subset = subset

    @staticmethod
    def dataset_name() -> str:
        """Returns a string identifier for the dataset."""
        return "librilight"

    @staticmethod
    def add_class_args(parser: ArgumentParser) -> None:
        """
        Adds LibriLight-specific arguments to the CLI parser.

        Args:
            parser: The JSONArgParse ArgumentParser object.
        """
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--subset",
            choices=[
                "small",
                "medium",
                "large",
                "small-processed",
                "medium-processed",
                "large-processed",
            ],
            help="""if we prepare the data for ["small", "medium", "large", "small-processed", "medium-processed", "large-processed"]""",
            required=True,
        )

    def prepare(self) -> None:
        """
        Executes the full LibriLight data preparation pipeline:
        - Discovers audio files in the given subset
        - Builds RecordingSet and SegmentSet
        - Extracts metadata (speaker, book, language)
        - Saves dataset in the output directory
        """

        logging.info(
            "Peparing LibriLight %s corpus_dir:%s -> data_dir:%s",
            self.subset,
            self.corpus_dir,
            self.output_dir,
        )

        rec_dir = self.corpus_dir / self.subset
        logging.info("searching audio files in %s", str(rec_dir))
        rec_files = list(rec_dir.glob("**/*.flac"))
        if not rec_files:
            # symlinks? try glob
            rec_files = [
                Path(f) for f in glob.iglob(f"{rec_dir}/**/*.flac", recursive=True)
            ]

        assert len(rec_files) > 0, "recording files not found"

        speakers = ["libri-" + f.parent.parent.name for f in rec_files]
        books = [f.parent.name for f in rec_files]
        rec_ids = ["librilight-" + f.with_suffix("").name for f in rec_files]
        if self.use_kaldi_ids:
            rec_ids = [f"{s}-{f}" for f, s in zip(rec_ids, speakers)]

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
        df_segs = pd.DataFrame({"id": rec_ids, "speaker": speakers, "book": books})
        df_segs["duration"] = recs.loc[df_segs["id"], "duration"].values
        df_segs["language"] = "eng"
        segments = SegmentSet(df_segs)
        segments.sort()

        logging.info("making speaker info file")
        df_spks = pd.DataFrame({"id": np.unique(df_segs["speaker"])})
        speakers = ClassInfo(df_spks)

        logging.info("making book info file")
        df_books = pd.DataFrame({"id": np.unique(df_segs["book"])})
        books = ClassInfo(df_books)

        languages = ClassInfo(pd.DataFrame({"id": ["eng"]}))

        classes = {
            "speaker": speakers,
            "book": books,
            "language": languages,
        }

        logging.info("making dataset")
        dataset = HypDataset(
            segments,
            classes=classes,
            recordings=recs,
        )
        logging.info("saving dataset at %s", self.output_dir)
        dataset.save(self.output_dir)
        logging.info(
            "datasets containts %d segments, %d speakers", len(segments), len(speakers)
        )
