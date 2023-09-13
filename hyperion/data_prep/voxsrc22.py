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

from ..utils import ClassInfo, Dataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike, urlretrieve_progress
from .data_prep import DataPrep


class VoxSRC22DataPrep(DataPrep):
    """Class to prepare VoxSRC22 dev/test data
    Attributes:
      corpus_dir: input data directory
      vox1_corpus_dir: input data directory for VoxCeleb1
      subset: subset of the data dev or test
      output_dir: output data directory
      target_sample_freq: target sampling frequency to convert the audios to.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        vox1_corpus_dir: PathLike,
        subset: str,
        output_dir: PathLike,
        use_kaldi_ids: bool,
        target_sample_freq: int,
        num_threads: int = 10,
    ):
        use_kaldi_ids = False
        super().__init__(
            corpus_dir, output_dir, use_kaldi_ids, target_sample_freq, num_threads
        )

        assert (
            vox1_corpus_dir is not None or subset == "test"
        ), "dev set needs the VoxCeleb1 corpus dir"
        self.subset = subset
        self.vox1_corpus_dir = (
            None if vox1_corpus_dir is None else Path(vox1_corpus_dir)
        )

    @staticmethod
    def dataset_name():
        return "voxsrc22"

    @staticmethod
    def add_class_args(parser):
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--subset",
            default="dev",
            choices=["dev", "test"],
            help="""vox2 subset in [dev, test]""",
        )
        parser.add_argument(
            "--vox1-corpus-dir",
            default=None,
            help="""corpus directory of voxceleb 1.""",
        )

    def prepare_track12_dev(self):
        logging.info(
            "Preparing VoxSRC22 %s corpus:%s + %s -> %s",
            self.subset,
            self.corpus_dir,
            self.vox1_corpus_dir,
            self.output_dir,
        )
        logging.info("making trials")
        trials_file = self.corpus_dir / "voxsrc2022_dev.txt"
        df_in = pd.read_csv(
            trials_file,
            header=None,
            sep=" ",
            names=["key", "enroll_file", "test_file"],
        )
        key = ["target" if k == 1 else "nontarget" for k in df_in["key"]]

        modelid = df_in["enroll_file"]
        segmentid = df_in["test_file"]
        df_trials = pd.DataFrame(
            {"modelid": modelid, "segmentid": segmentid, "targettype": key}
        )
        df_trials.sort_values(by=["modelid", "segmentid"], inplace=True)
        file_path = self.output_dir / "trials.csv"
        df_trials.to_csv(file_path, index=False)
        trials = {"trials": file_path}
        modelid = df_trials["modelid"].sort_values().unique()
        uniq_segmentid = df_trials["segmentid"].sort_values().unique()
        uniq_segmentid = np.unique(np.concatenate((uniq_segmentid, modelid), axis=0))

        logging.info("making enrollment map")
        df_enroll = pd.DataFrame({"modelid": modelid, "segmentid": modelid})
        file_path = self.output_dir / "enrollment.csv"
        df_enroll.to_csv(file_path, index=False)
        enrollments = {"enrollment": file_path}

        logging.info("making RecordingSet")
        vox1_segmentid = []
        vox22_segmentid = []
        for s in uniq_segmentid:
            if "VoxSRC2022_dev" in s:
                vox22_segmentid.append(s)
            else:
                vox1_segmentid.append(s)

        vox1_rec_files = [
            glob.glob(f"{self.vox1_corpus_dir}/**/{s}")[0] for s in vox1_segmentid
        ]
        # vox22_rec_files = [
        #     glob.glob(f"{self.corpus_dir}/**/{s}")[0] for s in vox22_segmentid
        # ]
        vox22_rec_files = [f"{self.corpus_dir}/{s}" for s in vox22_segmentid]

        rec_ids = vox22_segmentid + vox1_segmentid
        rec_files = vox22_rec_files + vox1_rec_files

        assert len(vox22_rec_files) > 0, "vox22 recording files not found"
        assert len(vox1_rec_files) > 0, "vox1 recording files not found"

        recs = pd.DataFrame({"id": rec_ids, "storage_path": rec_files})
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
            }
        )
        segments = SegmentSet(segments)
        segments.sort()

        logging.info("making dataset")
        dataset = Dataset(
            segments,
            recordings=recs,
            enrollments=enrollments,
            trials=trials,
            sparse_trials=False,
        )
        logging.info("saving dataset at %s", self.output_dir)
        dataset.save(self.output_dir)
        logging.info(
            "datasets containts %d segments",
            len(segments),
        )

    def prepare_track12_test(self):
        logging.info(
            "Preparing VoxSRC22 %s corpus:%s -> %s",
            self.subset,
            self.corpus_dir,
            self.output_dir,
        )

    def prepare(self):
        if self.subset == "dev":
            self.prepare_track12_dev()
        else:
            self.prepare_track12_test()
