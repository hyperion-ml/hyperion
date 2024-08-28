"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

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


class SRE19CTSDataPrep(DataPrep):
    """Class for preparing the SRE19 CTS Challenge (LDC2019E58/LDC2023S03) database into tables

    Attributes:
      corpus_dir: input data directory
      output_dir: output data directory
      partition: sre18 trial side in [unlabeled, enroll, test]
      use_kaldi_ids: puts speaker-id in front of segment id like kaldi
      target_sample_freq: target sampling frequency to convert the audios to.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        partition: str,
        output_dir: PathLike,
        use_kaldi_ids: bool,
        target_sample_freq: int,
        num_threads: int = 10,
    ):
        super().__init__(
            corpus_dir, output_dir, use_kaldi_ids, target_sample_freq, num_threads
        )

        self.partition = partition
        # self.use_ldc_langs = use_ldc_langs

    @staticmethod
    def dataset_name():
        return "sre19_cts"

    @staticmethod
    def add_class_args(parser):
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--partition",
            choices=["unlabeled", "enrollment", "test"],
            help="""sre18 trial side in [unlabeled, enroll, test]""",
            required=True,
        )
        # parser.add_argument(
        #     "--use-ldc-langs",
        #     default=False,
        #     action=ActionYesNo,
        #     help="convert language id to LDC format",
        # )

    def read_segments_metadata(self):

        source_type = "cts"
        corpusid = "cmn2"
        lang = "ARA-AEB"
        segments_file = (
            self.corpus_dir / "docs" / f"sre19_cts_challenge_segment_key.tsv"
        )
        logging.info("loading segment metadata from %s", segments_file)
        df_segs = pd.read_csv(segments_file, sep="\t")
        df_segs.rename(
            columns={
                "segmentid": "id",
                "subjectid": "speaker",
                "source_type": "cts_source_type",
            },
            inplace=True,
        )
        df_segs = df_segs.loc[(df_segs["partition"] == self.partition)]

        df_segs["gender"] = df_segs["gender"].apply(
            lambda x: "m" if x == "male" else "f"
        )
        df_segs["speaker"] = df_segs["speaker"].astype(str)
        df_segs["corpusid"] = corpusid
        df_segs["language"] = lang
        df_segs["source_type"] = source_type
        df_segs["filename"] = df_segs["id"]
        df_segs["dataset"] = self.dataset_name()
        if self.use_kaldi_ids:
            df_segs["id"] = df_segs[["speaker", "id"]].apply(
                lambda row: "-".join(row.values.astype(str)), axis=1
            )
        df_segs.set_index("id", drop=False, inplace=True)
        return df_segs

    def make_recording_set(self, df_segs):

        logging.info("making RecordingSet")
        wav_dir = self.corpus_dir / "data" / "eval" / self.partition

        df_recs = df_segs[["id"]].copy()
        corpusid = df_segs["corpusid"].values[0]
        df_recs["storage_path"] = df_segs["filename"].apply(
            lambda x: f"sph2pipe -f wav -p -c 1 {wav_dir / x} |"
        )
        df_recs["sample_freq"] = 8000

        if self.target_sample_freq is not None:
            df_recs["target_sample_freq"] = self.target_sample_freq

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

        df_source = df_segs[["source_type"]].drop_duplicates()
        df_source.rename(columns={"source_type": "id"}, inplace=True)
        df_source.sort_values(by="id", inplace=True)
        sources = ClassInfo(df_source)

        df_source = df_segs[["cts_source_type"]].drop_duplicates()
        df_source.rename(columns={"cts_source_type": "id"}, inplace=True)
        df_source.sort_values(by="id", inplace=True)
        cts_sources = ClassInfo(df_source)

        genders = ClassInfo(pd.DataFrame({"id": ["m", "f"]}))
        return {
            "speaker": speakers,
            "language": languages,
            "source_type": sources,
            "cts_source_type": cts_sources,
            "gender": genders,
        }

    def make_enrollments(self, df_segs):
        logging.info("making Enrollment")
        enroll_file = self.corpus_dir / "docs" / f"sre19_cts_challenge_enrollment.tsv"
        df_enr = pd.read_csv(enroll_file, sep="\t")
        df_enr = df_enr[df_enr["segmentid"].isin(df_segs["id"])]
        if self.use_kaldi_ids:
            df_enr["speaker"] = [
                df_segs.loc[df_segs["filename"] == s, "speaker"]
                for s in df_enr["segmentid"].values
            ]
            df_enr["segmentid"] = (
                df_enr["speaker"].astype(str) + "-" + df_enr["segmentid"]
            )
            df_enr.drop(columns=["speaker"], inplace=True)

        assert df_segs["id"].isin(df_enr["segmentid"]).all()
        return {"enrollment": EnrollmentMap(df_enr)}

    def make_trials(self, df_segs):
        logging.info("making Trials")
        trial_file = self.corpus_dir / "docs" / f"sre19_cts_challenge_trial_key.tsv"

        df_trial = pd.read_csv(trial_file, sep="\t")
        if self.use_kaldi_ids:
            df_trial["speaker"] = [
                df_segs.loc[df_segs["filename"] == s, "speaker"]
                for s in df_trial["segmentid"].values
            ]
            df_trial["segmentid"] = (
                df_trial["speaker"].astype(str) + "-" + df_trial["segmentid"]
            )
            df_trial.drop(columns=["speaker"], inplace=True)

        output_file = self.output_dir / "trials.tsv"
        df_trial.to_csv(output_file, sep="\t", index=False)
        trials = {"trials": output_file}
        attributes = {
            "num_enroll_segs": [1, 3],
            "phone_num_match": ["Y", "N"],
            "gender": ["male", "female"],
            "source_type": ["pstn", "voip"],
        }
        for test_set in ["prog", "eval"]:
            file_name = f"trials_{test_set}"
            output_file = self.output_dir / f"{file_name}.tsv"
            df_trial_ts = df_trial[df_trial["test_set"] == test_set]
            df_trial_ts.to_csv(output_file, sep="\t", index=False)
            trials[file_name] = output_file
            # modelid segmentid       side    targettype      num_enroll_segs phone_num_match gender  source_type     data_source     test_set
            # 1001_sre19      aajrhrbk_sre19.sph      a       nontarget       1       N       female  voip    cmn2    eval
            # 1001_sre19      aanxznlm_sre19.sph      a       target  1       N       female  pstn    cmn2    eval
            # 1001_sre19      aayuatag_sre19.sph      a       nontarget       1       N       female  pstn    cmn2    eval
            # 1001_sre19      abdabzqf_sre19.sph      a       nontarget       1       N       female  pstn    cmn2    eval
            # 1001_sre19      abgkmkdd_sre19.sph      a       nontarget       1       N       female  pstn    cmn2    eval
            # 1001_sre19      abkxwtcl_sre19.sph      a       target  1       N       female  pstn    cmn2    eval
            # 1001_sre19      abtdbtgv_sre19.sph      a       nontarget       1       N       female  pstn    cmn2    eval
            # 1001_sre19      acmpkpoo_sre19.sph      a       nontarget       1       N       female  voip    cmn2    eval

            for att_name, att_vals in attributes.items():
                for val in att_vals:
                    file_name = f"trials_{test_set}_{att_name}_{val}"
                    output_file = self.output_dir / f"{file_name}.tsv"
                    df_trials_cond = df_trial_ts.loc[
                        df_trial[att_name] == val,
                        ["modelid", "segmentid", "targettype"],
                    ]
                    df_trials_cond.to_csv(output_file, sep="\t", index=False)
                    trials[file_name] = output_file

        return trials

    def prepare(self):
        logging.info(
            "Peparing SRE19 %s corpus_dir: %s -> data_dir: %s",
            self.partition,
            self.corpus_dir,
            self.output_dir,
        )
        df_segs = self.read_segments_metadata()
        recs = self.make_recording_set(df_segs)
        df_segs["duration"] = recs.loc[df_segs["id"], "duration"].values

        classes = self.make_class_infos(df_segs)

        enrollments = None
        trials = None
        if self.partition == "enrollment":
            enrollments = self.make_enrollments(df_segs)
        elif self.partition == "test":
            trials = self.make_trials(df_segs)

        df_segs.drop(columns=["filename"], inplace=True)
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
