"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import glob
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from jsonargparse import ActionYesNo
from tqdm import tqdm

from ..utils import HypDataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike, urlretrieve_progress
from .data_prep import DataPrep


class RIRSDataPrep(DataPrep):
    """Class for preparing Musan database into tables

    Attributes:
      corpus_dir: input data directory
      output_dir: output data directory
      target_sample_freq: target sampling frequency to convert the audios to.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        output_dir: PathLike,
        target_sample_freq: int,
        num_threads: int = 10,
        **kwargs,
    ):
        super().__init__(corpus_dir, output_dir, False, target_sample_freq, num_threads)

    @staticmethod
    def dataset_name():
        return "rirs"

    @staticmethod
    def add_class_args(parser):
        DataPrep.add_class_args(parser)

    def prepare(self):
        logging.info(
            "Peparing RIRS corpus_dir:%s -> data_dir:%s",
            self.corpus_dir,
            self.output_dir,
        )
        rec_dir = self.corpus_dir
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
            }
        )
        if room_ids is not None:
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
