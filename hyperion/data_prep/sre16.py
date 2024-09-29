"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import glob
import logging
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from jsonargparse import ActionYesNo
from tqdm import tqdm

from ..utils import ClassInfo, EnrollmentMap, HypDataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike
from .data_prep import DataPrep


class SRE16DataPrep(DataPrep):
    """Class for preparing the SRE16 (LDC2019S20) database into tables"""

    def __init__(
        self,
        corpus_dir: PathLike,
        subset: str,
        partition: str,
        output_dir: PathLike,
        use_kaldi_ids: bool,
        target_sample_freq: int,
        num_threads: int = 10,
        use_ldc_langs: bool = False,
    ):
        super().__init__(
            corpus_dir, output_dir, use_kaldi_ids, target_sample_freq, num_threads
        )
        self.subset = subset
        self.partition = partition
        self.use_ldc_langs = use_ldc_langs
        assert self.subset == "eval" or self.partition in [
            "enrollment",
            "test",
            "train",
        ]

    @staticmethod
    def dataset_name():
        return "sre16"

    @staticmethod
    def add_class_args(parser):
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--subset",
            choices=["dev", "eval"],
            help="""sre16 subset in [dev, eval]""",
            required=True,
        )
        parser.add_argument(
            "--partition",
            choices=[
                "enrollment",
                "test",
                "train",
                "train60",
                "enrollment40",
                "test40",
            ],
            default="test",
            help="""sre16 trial side in [enroll, test]""",
        )
        parser.add_argument(
            "--use-ldc-langs",
            default=False,
            action=ActionYesNo,
            help="convert language id to LDC format",
        )

    def read_segments_metadata(self):
        if self.partition in ["train", "train60"]:
            segments_file = (
                self.corpus_dir / "docs" / f"sre16_{self.subset}_segment_key.tsv"
            )
        elif self.partition in ["enrollment", "enrollment40"]:
            segments_file = (
                self.corpus_dir
                / "docs"
                / f"sre16_{self.subset}_enrollment_segment_key.tsv"
            )
        elif self.partition in ["test", "test40"]:
            segments_file = (
                self.corpus_dir / "docs" / f"sre16_{self.subset}_test_segment_key.tsv"
            )

        logging.info("loading segment metadata from %s", segments_file)
        df_segs = pd.read_csv(segments_file, sep="\t")
        df_segs["dataset"] = "sre16"
        df_segs["corpusid"] = "call_my_net"
        df_segs["source_type"] = "cts"

        metadata_dir = self.corpus_dir / "metadata"
        df_sides = pd.read_csv(metadata_dir / "call_sides.tsv", sep="\t")
        df_calls = pd.read_csv(metadata_dir / "calls.tsv", sep="\t")
        df_spks = pd.read_csv(metadata_dir / "subjects.tsv", sep="\t")
        df_sides = pd.merge(df_sides, df_calls, how="left", on="call_id")
        df_sides = pd.merge(df_sides, df_spks, how="left", on="subject_id")
        df_sides.rename(
            columns={
                "subject_id": "speaker",
                "language_id": "language",
                "phone_id": "phoneid",
                "call_id": "sessionid",
                "sex": "gender",
            },
            inplace=True,
        )
        df_sides["gender"] = df_sides["gender"].str.lower()
        df_sides["language"] = df_sides["language"].str.upper()
        df_segs.rename(columns={"call_id": "sessionid"}, inplace=True)
        df_segs = pd.merge(df_segs, df_sides, how="left", on="sessionid")
        if self.partition in ["train60", "enrollment40", "test40"]:
            for lang in ["YUE", "TGL"]:
                spks = np.unique(df_sides.loc[df_segs["language"] == lang, "speaker"])
                n_spks = len(spks)
                n_60 = round(n_spks * 0.6)
                if self.partition == "train60":
                    spks = spks[:n_60]
                else:
                    spks = spks[n_60:]

                df_segs = df_segs.loc[df_segs["speaker"].isin(spks)]

        if self.use_kaldi_ids:
            df_segs["id"] = df_segs["speaker"] + "-" + df_segs["segment"]
        else:
            df_segs["id"] = df_segs["segment"]

        df_segs.set_index("id", drop=False, inplace=True)
        return df_segs

    def make_recording_set(self, df_segs):
        logging.info("making RecordingSet")
        rec_dir = self.corpus_dir / "data"
        logging.info("searching audio files in %s", str(rec_dir))
        rec_files = [Path(f) for f in glob.iglob(f"{rec_dir}/**/*.sph", recursive=True)]
        rec_ids = [f.stem for f in rec_files]
        df_recs = pd.DataFrame({"segment": rec_ids, "filename": rec_files})
        df_recs.set_index("segment", inplace=True)
        df_recs = df_recs.loc[df_segs["segment"]]
        df_recs["id"] = df_segs["id"]
        df_recs.reset_index(drop=True, inplace=True)
        df_recs["storage_path"] = df_recs["filename"].apply(
            lambda x: f"sph2pipe -f wav -p -c 1 {x} |"
        )
        df_recs.drop(columns=["filename"], inplace=True)
        df_recs["sample_freq"] = 8000

        recordings = RecordingSet(df_recs)
        recordings.get_durations(self.num_threads)
        return recordings

    def make_class_infos(self, df_segs):
        logging.info("making ClassInfos")
        df_segs = df_segs.reset_index(drop=True)
        df_spks = df_segs[["speaker", "gender"]].drop_duplicates()
        df_spks.rename(columns={"speaker": "id"}, inplace=True)
        df_spks.sort_values(by="id", inplace=True)
        speakers = ClassInfo(df_spks)

        df_langs = df_segs[["language"]].drop_duplicates()
        df_langs.rename(columns={"language": "id"}, inplace=True)
        df_langs.sort_values(by="id", inplace=True)
        languages = ClassInfo(df_langs)

        sources = genders = ClassInfo(pd.DataFrame({"id": ["cts"]}))
        genders = ClassInfo(pd.DataFrame({"id": ["m", "f"]}))
        return {
            "speaker": speakers,
            "language": languages,
            "source_type": sources,
            "gender": genders,
        }

    def make_enrollments(self, df_segs):
        logging.info("making Enrollment")
        enroll_file = self.corpus_dir / "docs" / f"sre16_{self.subset}_enrollment.tsv"
        df_enr = pd.read_csv(enroll_file, sep="\t")
        if self.partition == "enrollment40":
            df_enr = df_enr.loc[df_enr.segment.isin(df_segs.segment)]
        if self.use_kaldi_ids:
            segment_ids = [
                df_segs.loc[df_segs["segment"] == s, "id"]
                for s in df_enr["segment"].values
            ]
            df_enr["segmentid"] = segment_ids
            df_enr.drop(columns=["segment"], axis=1, inplace=True)
        else:
            df_enr.rename(columns={"segment": "segmentid"}, inplace=True)

        df_enr.drop(columns=["side"], inplace=True)
        return {"enrollment": EnrollmentMap(df_enr)}

    def make_trials(self, df_segs):
        logging.info("making Trials")
        trial_file = self.corpus_dir / "docs" / f"sre16_{self.subset}_trial_key.tsv"
        df_trial = pd.read_csv(trial_file, sep="\t")
        if self.partition == "enrollment40":
            df_trial = df_trial.loc[df_trial.segment.isin(df_segs.segment)]

        output_file = self.output_dir / "trials_official.tsv"
        df_trial.to_csv(output_file, sep="\t", index=False)
        if self.use_kaldi_ids:
            segment_ids = [
                df_segs.loc[df_segs["segment"] == s, "id"]
                for s in df_trial["segment"].values
            ]
            df_trial["segment"] = segment_ids

        output_file = self.output_dir / "trials_official.tsv"
        df_trial.to_csv(output_file, sep="\t", index=False)

        df_trial.rename(columns={"segment": "segmentid"}, inplace=True)
        output_file = self.output_dir / "trials.tsv"
        df_trial.to_csv(output_file, sep="\t", index=False)
        trials = {"trials": output_file}

        df_trial["gender"] = df_segs.loc[df_trial.segmenid, "gender"]
        df_trial["language"] = df_segs.loc[df_trial.segmenid, "language"]

        df_trial = self._get_extra_trial_conds(df_trial, df_segs)
        output_file = self.output_dir / "trials_ext.tsv"
        df_trial.to_csv(output_file, sep="\t", index=False)
        trials = {"trials_ext": output_file}
        attributes = {
            "gender": ["m", "f"],
            "language": ["CMN", "CEB"] if self.subset == "dev" else ["YUE", "TGL"],
        }
        for att_name, att_vals in attributes.items():
            for val in att_vals:
                file_name = f"trials_{att_name}_{val}"
                output_file = self.output_dir / f"{file_name}.tsv"
                df_trials_cond = df_trial.loc[
                    df_trial[att_name] == val, ["modelid", "segmentid", "targettype"]
                ]
                df_trials_cond.to_csv(output_file, sep="\t", index=False)
                trials[file_name] = output_file

        return trials

    def prepare(self):
        logging.info(
            "Peparing SRE16 %s %s corpus_dir: %s -> data_dir: %s",
            self.subset,
            self.partition,
            self.corpus_dir,
            self.output_dir,
        )
        if not (self.corpus_dir / "metadata").is_dir():
            if self.subset == "eval":
                self.corpus_dir = self.corpus_dir / "data" / "eval" / "R149_0_1"
            else:
                self.corpus_dir = self.corpus_dir / "data" / "dev" / "R148_0_0"

        df_segs = self.read_segments_metadata()
        recs = self.make_recording_set(df_segs)
        df_segs["duration"] = recs.loc[df_segs["id"], "duration"].values

        classes = self.make_class_infos(df_segs)

        enrollments = None
        trials = None
        if self.partition in ["enrollment", "enrollment40"]:
            enrollments = self.make_enrollments(df_segs)
            trials = None
        elif self.partition in ["test", "test40"]:
            trials = self.make_trials(df_segs)

        df_segs.drop(columns=["segment"], inplace=True)
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
