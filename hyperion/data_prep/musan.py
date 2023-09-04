"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
import glob
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from jsonargparse import ActionYesNo
from tqdm import tqdm

from ..utils import Dataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike, urlretrieve_progress
from .data_prep import DataPrep


class MusanDataPrep(DataPrep):
    """Class for preparing Musan database into tables

    Attributes:
      corpus_dir: input data directory
      subset: subset of the data noise, music, speech
      output_dir: output data directory
      target_sample_freq: target sampling frequency to convert the audios to.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        subset: str,
        output_dir: PathLike,
        target_sample_freq: int,
        num_threads: int = 10,
        **kwargs,
    ):
        super().__init__(corpus_dir, output_dir, False, target_sample_freq, num_threads)
        self.subset = subset

    @staticmethod
    def dataset_name():
        return "musan"

    @staticmethod
    def add_class_args(parser):
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--subset",
            choices=["noise", "music", "speech"],
            help="""musan subset in [noise, music, speech]""",
            required=True,
        )

    def prepare(self):
        logging.info(
            "Peparing Musan %s corpus_dir:%s -> data_dir:%s",
            self.subset,
            self.corpus_dir,
            self.output_dir,
        )
        rec_dir = self.corpus_dir / self.subset
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
        recs.sort()

        logging.info("getting recording durations")
        self.get_recording_duration(recs)
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
        segments = SegmentSet(segments)
        segments.sort()
        logging.info("making dataset")
        dataset = Dataset(
            segments,
            recordings=recs,
        )
        logging.info("saving dataset at %s", self.output_dir)
        dataset.save(self.output_dir)
        logging.info(
            "datasets containts %d segments",
            len(segments),
        )
