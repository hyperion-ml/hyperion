"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import re
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from jsonargparse import ActionYesNo
from tqdm import tqdm

from ..utils import ClassInfo, EnrollmentMap, HypDataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike
from .data_prep import DataPrep


class LDC2024E41DataPrep(DataPrep):
    """
    Prepares the LDC2024E41 ARTS sample dataset

    This class parses metadata, audio file locations, and optionally builds
    enrollment maps and trial keys. It supports partitioning by task phase.

    Attributes:
        corpus_dir (PathLike): Root directory of the dataset.
        partition (str): One of 'enrollment', 'test', or 'unlabeled'.
        output_dir (PathLike): Directory where the prepared dataset is saved.
        use_kaldi_ids (bool): If True, use Kaldi-style IDs (<speaker>-<segment>).
        target_sample_freq (Optional[int]): Resample recordings to this frequency (Hz).
        num_threads (int): Number of threads for parallel audio processing.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        partition: str,
        output_dir: PathLike,
        use_kaldi_ids: bool = False,
        target_sample_freq: Optional[int] = None,
        num_threads: int = 10,
    ):
        super().__init__(
            corpus_dir, output_dir, use_kaldi_ids, target_sample_freq, num_threads
        )

        self.partition = partition
        # self.use_ldc_langs = use_ldc_langs

    @staticmethod
    def dataset_name() -> str:
        """Returns the dataset name identifier."""
        return "ldc2024e41"

    @staticmethod
    def add_class_args(parser) -> None:
        """Adds CLI arguments specific to the LDC2024E41 ARTS Sample dataset."""
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--partition",
            choices=["enrollment", "test"],
            help="""trial side in [enrollment, test]""",
            required=True,
        )

    def _read_docs(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads and processes metadata files from `docs/`.

        Returns:
            Tuple of DataFrames: (enrollment_df, test_df, trial_df)
        """
        logging.info("loading docs")
        docs_dir = self.corpus_dir / "docs"
        df_enr = pd.read_csv(docs_dir / "ARTS_sample_enroll.tsv", sep="\t")
        df_enr["filename"] = df_enr["segmentid.channel"].apply(
            lambda x: re.sub(".*/", "", re.sub(r"\.sph\.[AB]$", "", x))
        )
        df_enr["channel"] = df_enr["segmentid.channel"].apply(lambda x: x[-1].lower())
        df_enr["segmentid"] = [
            f"{s}-{c.lower()}" for s, c in zip(df_enr["filename"], df_enr["channel"])
        ]
        df_trial = pd.read_csv(docs_dir / "ARTS_sample_test_key.csv", sep=",")
        df_trial.rename(
            columns={
                "subjectid": "modelid",
                " segmentid": "filename",
                " channel": "channel",
                " targettype": "targettype",
                " gender": "gender",
            },
            inplace=True,
        )
        df_trial["segmentid"] = [
            f"{s}-{c.lower()}"
            for s, c in zip(df_trial["filename"], df_trial["channel"])
        ]
        df_trial["gender"] = df_trial["gender"].apply(lambda x: x.strip()[0])
        df_test = df_trial[df_trial["targettype"] == "target"].sort_values(
            by=["segmentid"]
        )
        df_test2 = df_trial[~df_trial["segmentid"].isin(df_test["segmentid"])].copy()
        df_test2["modelid"] = pd.NA
        df_test2 = df_test2.drop_duplicates()
        df_test = pd.concat([df_test, df_test2], ignore_index=True).drop(
            columns=["targettype"], axis=1
        )
        df_test.rename(columns={"modelid": "speaker"}, inplace=True)
        return df_enr, df_test, df_trial

    def make_recording_set(self, df_segs: pd.DataFrame) -> RecordingSet:
        """
        Builds the RecordingSet table from segment metadata.

        Args:
            df_segs (pd.DataFrame): Metadata for segments.

        Returns:
            RecordingSet: Table of audio file paths and durations.
        """
        logging.info("making RecordingSet")
        wav_dir = self.corpus_dir / "data" / self.partition

        def channel_to_num(c):
            if c == "a":
                return 1
            elif c == "b":
                return 2
            else:
                raise ValueError(f"Invalid channel: {c}")

        paths = [
            f"sph2pipe -f wav -p -c {channel_to_num(c)} {wav_dir / s}.sph |"
            for s, c in zip(df_segs["filename"], df_segs["channel"])
        ]
        df_recs = pd.DataFrame({"id": df_segs["segmentid"], "storage_path": paths})
        df_recs["sample_freq"] = 8000
        if self.target_sample_freq is not None:
            df_recs["target_sample_freq"] = self.target_sample_freq

        recordings = RecordingSet(df_recs)
        recordings.get_durations(self.num_threads)
        return recordings

    def make_class_infos(self, df_segs: pd.DataFrame) -> dict[str, ClassInfo]:
        """
        Builds ClassInfo tables for speakers, gender, language, and source type.

        Args:
            df_segs (pd.DataFrame): Segment metadata.

        Returns:
            dict[str, ClassInfo]: Class tables keyed by name.
        """
        logging.info("making ClassInfos")
        df_segs = df_segs.reset_index(drop=True)
        df_spks = df_segs[["speaker", "gender"]].drop_duplicates().dropna()
        df_spks.rename(columns={"speaker": "id"}, inplace=True)
        df_spks.sort_values(by="id", inplace=True)
        speakers = ClassInfo(df_spks)

        languages = ClassInfo(pd.DataFrame({"id": ["eng"]}))
        genders = ClassInfo(pd.DataFrame({"id": ["m", "f"]}))
        sources = ClassInfo(pd.DataFrame({"id": ["cts"]}))
        return {
            "speaker": speakers,
            "language": languages,
            "source_type": sources,
            "gender": genders,
        }

    def make_enrollments(self, df_enr: pd.DataFrame) -> dict[str, EnrollmentMap]:
        """
        Creates an EnrollmentMap for the enrollment partition.

        Args:
            df_enr (pd.DataFrame): Enrollment metadata.

        Returns:
            dict[str, EnrollmentMap]: Map containing model-segment links.
        """
        logging.info("making Enrollment")
        df_enr = df_enr[["modelid", "segmentid"]].copy()
        return {"enrollment": EnrollmentMap(df_enr)}

    def make_trials(self, df_trial: pd.DataFrame) -> dict[str, Path]:
        """
        Builds the main trial file from the trial metadata.

        Args:
            df_trial (pd.DataFrame): Trial metadata with model/segment IDs and targettype.

        Returns:
            dict[str, Path]: Mapping from 'trials' to path of the saved trial CSV.
        """

        logging.info("making Trials")
        df_trial = df_trial[["modelid", "segmentid", "targettype"]].copy()
        output_file = self.output_dir / "trials.csv"
        df_trial.to_csv(output_file, sep=",", index=False)
        trials = {"trials": output_file}
        return trials

    def prepare(self) -> None:
        """
        Executes the full LDC2024E41 preparation pipeline:
        - Loads metadata
        - Creates segment, recording, and class info tables
        - Writes enrollment maps and trial keys as needed
        - Saves HypDataset to output_dir
        """
        logging.info(
            "Peparing LDC2024E41 %s corpus_dir: %s -> data_dir: %s",
            self.partition,
            self.corpus_dir,
            self.output_dir,
        )
        df_enr, df_test, df_trial = self._read_docs()
        enrollments = None
        trials = None
        if self.partition == "enrollment":
            recs = self.make_recording_set(df_enr)
            df_segs = pd.DataFrame(
                {"id": df_enr["segmentid"], "speaker": df_enr["modelid"]}
            )
            df_segs["gender"] = [
                df_test.loc[df_test["speaker"] == s, "gender"].values[0]
                for s in df_segs["speaker"]
            ]
            enrollments = self.make_enrollments(df_enr)
        else:
            recs = self.make_recording_set(df_test)
            df_segs = pd.DataFrame(
                {
                    "id": df_test["segmentid"],
                    "speaker": df_test["speaker"],
                    "gender": df_test["gender"],
                }
            )
            trials = self.make_trials(df_trial)

        df_segs["duration"] = recs.loc[df_segs["id"], "duration"].values
        df_segs["corpusid"] = "mx3"
        df_segs["language"] = "eng"
        df_segs["source_type"] = "cts"
        df_segs["dataset"] = self.dataset_name()

        classes = self.make_class_infos(df_segs)

        segments = SegmentSet(df_segs)

        logging.info("making dataset")
        dataset = HypDataset(
            segments,
            classes,
            recordings=recs,
            enrollments=enrollments,
            trials=trials,
            sparse_trials=False,
        )
        logging.info("saving dataset at %s", self.output_dir)
        dataset.save(self.output_dir)
        dataset.describe()
