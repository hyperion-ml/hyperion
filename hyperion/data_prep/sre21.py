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

from ..utils import (
    ClassInfo,
    EnrollmentMap,
    HypDataset,
    ImageSet,
    RecordingSet,
    SegmentSet,
    VideoSet,
)
from ..utils.misc import PathLike
from .data_prep import DataPrep


class LangTrialCond(str, Enum):
    ENG_ENG = "ENG_ENG"
    ENG_CMN = "ENG_CMN"
    ENG_YUE = "ENG_YUE"
    CMN_CMN = "CMN_CMN"
    CMN_YUE = "CMN_YUE"
    YUE_YUE = "YUE_YUE"
    OTHER_OTHER = "OTHER_OTHER"
    OTHER_ENG = "OTHER_ENG"
    OTHER_CMN = "OTHER_CMN"
    OTHER_YUE = "OTHER_YUE"

    @staticmethod
    def choices():
        return [o.value for o in LangTrialCond]

    @staticmethod
    def is_eng(val):
        if val in ["ENG", "USE", "english"]:
            return True
        return False

    @staticmethod
    def get_side_cond(val):
        if val in ["ENG", "USE", "english"]:
            return "ENG"
        if val in ["YUE", "cantonese"]:
            return "YUE"
        if val in ["CMN", "mandarin"]:
            return "CMN"

        return "OTHER"

    @staticmethod
    def get_trial_cond(enr, test):
        enr = LangTrialCond.get_side_cond(enr)
        test = LangTrialCond.get_side_cond(test)
        trial = enr + "_" + test
        try:
            return LangTrialCond[trial]
        except:
            trial = test + "_" + enr
            return LangTrialCond[trial]


class SourceTrialCond(str, Enum):
    CTS_CTS = "CTS_CTS"
    CTS_AFV = "CTS_AFV"
    AFV_AFV = "AFV_AFV"

    @staticmethod
    def choices():
        return [o.value for o in SourceTrialCond]

    @staticmethod
    def get_trial_cond(enr, test):
        trial = enr.upper() + "_" + test.upper()
        try:
            return SourceTrialCond[trial]
        except:
            trial = test.upper() + "_" + enr.upper()
            return SourceTrialCond[trial]


class SRE21DataPrep(DataPrep):
    """Class for preparing the SRE21 dev (LDC2021E09) or eval (LDC2021E10) database into tables

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
        use_ldc_langs: bool = False,
    ):
        super().__init__(
            corpus_dir, output_dir, use_kaldi_ids, target_sample_freq, num_threads
        )
        self.modality = modality
        self.subset = subset
        self.partition = partition
        self.use_ldc_langs = use_ldc_langs

    @staticmethod
    def dataset_name():
        return "sre21"

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
            help="""sre21 subset in [dev, eval]""",
            required=True,
        )
        parser.add_argument(
            "--partition",
            choices=["enrollment", "test"],
            help="""sre21 trial side in [enroll, test]""",
            required=True,
        )
        parser.add_argument(
            "--use-ldc-langs",
            default=False,
            action=ActionYesNo,
            help="convert language id to LDC format",
        )

    def read_segments_metadata(self):

        segments_file = (
            self.corpus_dir / "docs" / f"sre21_{self.subset}_segment_key.tsv"
        )
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
        if self.partition == "enrollment":
            jpg = df_segs["id"].str.match(r".*\.jpg$")
            if self.modality in ["audio", "audio-visual"]:
                df_segs = df_segs.loc[~jpg]
            else:
                df_segs = df_segs.loc[jpg]
        else:
            # mp4 = df_segs["filename"].apply(lambda x: x[-3:] != "mp4")
            mp4 = df_segs["id"].str.match(r".*\.mp4$")
            if self.modality == "audio":
                df_segs = df_segs.loc[~mp4]
            elif self.modality in ["visual", "audio-visual"]:
                df_segs = df_segs.loc[mp4]
                df_segs["source_type"] = "afv"

        if self.use_ldc_langs:
            df_segs.replace({"language": "english"}, {"language": "ENG"}, inplace=True)
            df_segs.replace(
                {"language": "cantonese"}, {"language": "YUE"}, inplace=True
            )
            df_segs.replace({"language": "mandarin"}, {"language": "CMN"}, inplace=True)

        # df_segs.loc[df_segs["id"].str.match(r".*\.mp4$"), "source_type"] = "afv"
        df_segs["filename"] = df_segs["id"]
        df_segs["dataset"] = self.dataset_name()
        df_segs["corpusid"] = "we_can_talk"
        if self.use_kaldi_ids:
            df_segs["id"] = df_segs[["speaker", "id"]].apply(
                lambda row: "-".join(row.values.astype(str)), axis=1
            )
        df_segs.set_index("id", drop=False, inplace=True)
        return df_segs

    def make_recording_set(self, df_segs):

        logging.info("making RecordingSet")
        if self.modality == "audio":
            wav_dir = self.corpus_dir / "data" / "audio" / self.partition
        else:
            wav_dir = self.corpus_dir / "data" / "video" / self.partition

        df_recs = df_segs[["id"]].copy()
        if self.modality == "audio":
            cts_idx = df_segs["source_type"] == "cts"
            df_recs.loc[cts_idx, "storage_path"] = df_segs.loc[
                cts_idx, "filename"
            ].apply(lambda x: f"sph2pipe -f wav -p -c 1 {wav_dir / x} |")
            df_recs.loc[~cts_idx, "storage_path"] = df_segs.loc[
                ~cts_idx, "filename"
            ].apply(lambda x: str(wav_dir / x))
            df_recs.loc[cts_idx, "sample_freq"] = 8000
            df_recs.loc[~cts_idx, "sample_freq"] = 16000
            if self.target_sample_freq is not None:
                df_recs["target_sample_freq"] = self.target_sample_freq
        else:
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

    def make_image_set(self, df_segs):

        logging.info("making ImageSet")

        img_dir = self.corpus_dir / "data" / "image" / self.partition
        df_imgs = df_segs[["id"]].copy()
        df_imgs["storage_path"] = df_segs["filename"].apply(lambda x: f"{img_dir/x}")

        images = ImageSet(df_imgs)
        return images

    def make_video_set(self, df_segs):

        logging.info("making VideoSet")
        vid_dir = self.corpus_dir / "data" / "video" / self.partition

        df_vids = df_segs[["id"]].copy()
        df_vids["sample_freq"] = 44100
        df_vids["storage_path"] = df_segs["filename"].apply(lambda x: f"{vid_dir/x}")

        if self.target_sample_freq is not None:
            df_vids["target_sample_freq"] = self.target_sample_freq

        videos = VideoSet(df_vids)
        videos.get_metadata(self.num_threads)
        return videos

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
            enroll_file = (
                self.corpus_dir / "docs" / f"sre21_audio_{self.subset}_enrollment.tsv"
            )
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
            key_file = (
                self.corpus_dir / "docs" / f"sre21_visual_{self.subset}_trial_key.tsv"
            )
            df_key = pd.read_csv(key_file, sep="\t")
            ids = np.unique(df_key["imageid"])
            df_enr = pd.DataFrame({"id": ids, "segmentid": ids})
            return {"enrollment": EnrollmentMap(df_enr)}

    def _get_enroll_conds(self):
        # we read the segments again to get extra conditions
        partition = self.partition
        self.partition = "enrollment"
        df_segs = self.read_segments_metadata()
        self.partition = partition
        enr_map = self.make_enrollments(df_segs)["enrollment"]
        modelids = enr_map["id"].unique()
        langs = []
        sources = []
        for modelid in modelids:
            seg_id = enr_map.loc[modelid, "segmentid"]
            if not isinstance(seg_id, str):
                seg_id = seg_id.values[0]

            lang = df_segs.loc[seg_id, "language"]
            source = df_segs.loc[seg_id, "source_type"]
            langs.append(lang)
            sources.append(source)

        df_enr_conds = pd.DataFrame(
            {"id": modelids, "language": langs, "source_type": sources}
        )
        df_enr_conds.set_index("id", inplace=True)
        return df_enr_conds

    def _get_extra_trial_conds(self, df_trial, df_segs):
        df_enr = self._get_enroll_conds()
        for i, row in df_trial.iterrows():
            modelid = row["modelid"]
            segid = row["segmentid"]
            lang_cond = LangTrialCond.get_trial_cond(
                df_enr.loc[modelid, "language"], df_segs.loc[segid, "language"]
            )
            source_cond = SourceTrialCond.get_trial_cond(
                df_enr.loc[modelid, "source_type"],
                df_segs.loc[segid, "source_type"],
            )
            df_trial.loc[i, "language"] = lang_cond
            df_trial.loc[i, "source_type"] = source_cond

        return df_trial

    def make_trials(self, df_segs):
        logging.info("making Trials")
        trial_file = (
            self.corpus_dir
            / "docs"
            / f"sre21_{self.modality}_{self.subset}_trial_key.tsv"
        )

        df_trial = pd.read_csv(trial_file, sep="\t")
        if self.modality == "visual":
            df_trial.rename(columns={"imageid": "modelid"}, inplace=True)

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

        if self.modality == "visual":
            attributes = {
                "gender": ["male", "female"],
            }
        else:
            df_trial = self._get_extra_trial_conds(df_trial, df_segs)
            output_file = self.output_dir / "trials_ext.tsv"
            df_trial.to_csv(output_file, sep="\t", index=False)
            trials = {"trials_ext": output_file}
            attributes = {
                "num_enroll_segs": [1, 3],
                "phone_num_match": ["Y", "N"],
                "gender": ["male", "female"],
                "source_type_match": ["Y", "N"],
                "language_match": ["N", "Y"],
                "language": LangTrialCond.choices(),
                "source_type": SourceTrialCond.choices(),
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
            "Peparing SRE21 %s %s %s corpus_dir: %s -> data_dir: %s",
            self.modality,
            self.subset,
            self.partition,
            self.corpus_dir,
            self.output_dir,
        )
        df_segs = self.read_segments_metadata()
        recs = None
        imgs = None
        vids = None
        if self.modality != "visual":
            recs = self.make_recording_set(df_segs)
            df_segs["duration"] = recs.loc[df_segs["id"], "duration"].values

        if self.modality != "audio":
            if self.partition == "enrollment":
                imgs = self.make_image_set(df_segs)
            else:
                vids = self.make_video_set(df_segs)

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
            images=imgs,
            videos=vids,
            enrollments=enrollments,
            trials=trials,
            sparse_trials=False,
        )
        logging.info("saving dataset at %s", self.output_dir)
        dataset.save(self.output_dir)
        dataset.describe()
