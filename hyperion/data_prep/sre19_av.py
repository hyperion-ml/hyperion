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


class SRE19AVDataPrep(DataPrep):
    """Class for preparing the SRE19 AV dev (LDC2019E56/LDC2023V01) or eval (LDC2019E57/LDC2023V01) database into tables

    Attributes:
      corpus_dir: input data directory
      output_dir: output data directory
      modality: audio, visual, audio-visual
      subset: sre21 subset in [dev, eval]
      partition: sre21 trial side in [enroll, test]
      use_kaldi_ids: puts speaker-id in front of segment id like kaldi
      target_sample_freq: target sampling frequency to convert the audios to.
      ldc_langs: convert language id to LDC format
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        modality: str,
        subset: str,
        partition: str,
        output_dir: PathLike,
        use_kaldi_ids: bool,
        target_sample_freq: int,
        num_threads: int = 10,
        # use_ldc_langs: bool = False,
    ):
        super().__init__(
            corpus_dir, output_dir, use_kaldi_ids, target_sample_freq, num_threads
        )
        self.modality = modality
        self.subset = subset
        self.partition = partition
        # self.use_ldc_langs = use_ldc_langs
        self.docs_dir = self.corpus_dir / "docs" / subset
        if self.docs_dir.is_dir():
            # package is LDC2023V01
            self.data_dir = self.corpus_dir / "data" / subset
        else:
            # package is LDC2019E56/57
            self.docs_dir = self.corpus_dir / "docs"
            self.data_dir = self.corpus_dir / "data"

    @staticmethod
    def dataset_name():
        return "sre19_av"

    @staticmethod
    def add_class_args(parser):
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--modality",
            default="audio",
            choices=["audio", "visual", "audio-visual"],
            help="audio, visual, audio-visual",
        )
        parser.add_argument(
            "--subset",
            choices=["dev", "eval"],
            help="""sre19 av subset in [dev, eval]""",
            required=True,
        )
        parser.add_argument(
            "--partition",
            choices=["enrollment", "test"],
            help="""sre19 av trial side in [enroll, test]""",
            required=True,
        )
        # parser.add_argument(
        #     "--use-ldc-langs",
        #     default=False,
        #     action=ActionYesNo,
        #     help="convert language id to LDC format",
        # )

    def read_segments_metadata(self):

        segments_file = self.docs_dir / f"sre19_av_{self.subset}_segment_key.tsv"
        logging.info("loading segment metadata from %s", segments_file)
        df_segs = pd.read_csv(segments_file, sep="\t")
        df_segs.rename(
            columns={"segmentid": "id", "subjectid": "speaker"},
            inplace=True,
        )
        df_segs["gender"] = df_segs["gender"].apply(
            lambda x: "m" if x == "male" else "f"
        )
        df_segs["speaker"] = df_segs["speaker"].astype(str)
        df_segs = df_segs.loc[df_segs["partition"] == self.partition]

        df_segs["source_type"] = "afv"
        df_segs["filename"] = df_segs["id"].apply(lambda x: f"{x}.mp4")
        df_segs["dataset"] = self.dataset_name()
        df_segs["corpusid"] = "vast"
        df_segs["language"] = "ENG"
        if self.use_kaldi_ids:
            df_segs["id"] = df_segs[["speaker", "id"]].apply(
                lambda row: "-".join(row.values.astype(str)), axis=1
            )
        df_segs.set_index("id", drop=False, inplace=True)
        return df_segs

    def make_recording_set(self, df_segs):

        logging.info("making RecordingSet")
        wav_dir = self.data_dir / self.partition

        df_recs = df_segs[["id"]].copy()
        if self.target_sample_freq is not None:
            ar_opt = f"-ar {self.target_sample_freq} "
            df_recs["sample_freq"] = self.target_sample_freq
        else:
            ar_opt = ""
            df_recs["sample_freq"] = 44100

        df_recs["storage_path"] = df_segs["filename"].apply(
            lambda x: f"ffmpeg -v 8 -i {wav_dir/x} -vn {ar_opt}-ac 1 -f wav - |"
        )

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

        genders = ClassInfo(pd.DataFrame({"id": ["m", "f"]}))
        return {
            "speaker": speakers,
            "language": languages,
            "source_type": sources,
            "gender": genders,
        }

    def make_enrollments(self, df_segs):
        logging.info("making Enrollment")
        if self.modality in ["audio", "audio-visual"]:
            enroll_file = self.docs_dir / f"sre19_av_{self.subset}_enrollment.tsv"
            df_enr = pd.read_csv(enroll_file, sep="\t")
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

        if self.modality == "visual":
            # TODO
            pass

    def make_trials(self, df_segs):
        logging.info("making Trials")
        trial_file = self.docs_dir / f"sre19_av_{self.subset}_trial_key.tsv"

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

        return trials

    def prepare(self):
        logging.info(
            "Peparing SRE19 %s %s %s corpus_dir: %s -> data_dir: %s",
            self.modality,
            self.subset,
            self.partition,
            self.corpus_dir,
            self.output_dir,
        )
        df_segs = self.read_segments_metadata()
        if self.modality != "visual":
            recs = self.make_recording_set(df_segs)
            df_segs["duration"] = recs.loc[df_segs["id"], "duration"].values
        else:
            recs = None

        classes = self.make_class_infos(df_segs)

        if self.partition == "enrollment":
            enrollments = self.make_enrollments(df_segs)
            trials = None
        else:
            enrollments = None
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
