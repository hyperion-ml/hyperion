"""
Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import re
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from jsonargparse import ActionYesNo, ArgumentParser
from tqdm import tqdm

from ..utils import (
    ClassInfo,
    EnrollmentMap,
    HypDataset,
    RecordingSet,
    SegmentSet,
    VideoSet,
)
from ..utils.misc import PathLike
from .data_prep import DataPrep


class JanusMultimediaDataPrep(DataPrep):
    """
    Prepares the Janus Multimedia (LDC2019E55) dataset into structured tables for speaker recognition.

    This class handles:
    - Metadata parsing
    - Recording and video manifest generation
    - Segment filtering based on trial conditions
    - Trial and enrollment list creation

    Attributes:
        corpus_dir (PathLike): Input directory containing dataset structure.
        output_dir (PathLike): Output directory where the prepared tables are written.
        subset (str): Evaluation subset, either 'dev' or 'eval'.
        condition (str): Trial condition, either 'core' or 'full'.
        partition (str): Either 'enrollment' or 'test'.
        with_videos (bool): If True, prepare video manifest in addition to audio.
        target_sample_freq (Optional[int]): Optional resampling frequency.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        subset: str,
        condition: str,
        partition: str,
        output_dir: PathLike,
        use_kaldi_ids: bool = False,
        target_sample_freq: Optional[int] = None,
        with_videos: bool = False,
        num_threads: int = 10,
    ):
        super().__init__(corpus_dir, output_dir, False, target_sample_freq, num_threads)
        self.subset = subset
        self.condition = condition
        self.partition = partition
        self.with_videos = with_videos

    @staticmethod
    def dataset_name():
        """
        Initialize Janus dataset preprocessor.
        """
        return "janus_multimedia"

    @staticmethod
    def add_class_args(parser: ArgumentParser) -> None:
        """Add command-line arguments to the parser."""
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--subset",
            choices=["dev", "eval"],
            help="""janus subset in [dev, eval]""",
            required=True,
        )
        parser.add_argument(
            "--condition",
            choices=["core", "full"],
            help="""janus condition in [core, full]""",
            default="core",
        )
        parser.add_argument(
            "--partition",
            choices=["enrollment", "test"],
            help="""janus trial side in [enroll, test]""",
            required=True,
        )
        parser.add_argument(
            "--with-videos",
            default=False,
            action=ActionYesNo,
            help="""prepare video manifest""",
        )

    def read_segments_metadata(self) -> pd.DataFrame:
        """
        Loads and parses Janus segment metadata TSV file, filters based on trial subset and partition.

        Returns:
            pd.DataFrame: DataFrame with selected and processed segments.
        """

        segments_file = self.corpus_dir / "docs" / f"janus_multimedia.tsv"
        logging.info("loading segment metadata from %s", segments_file)
        df_segs = pd.read_csv(segments_file, sep="\t")
        df_segs.columns = map(str.lower, df_segs.columns)
        # fix name
        df_segs["video"] = df_segs["video"].apply(
            lambda x: re.sub("video/", "videos/", x)
        )
        df_segs.rename(
            columns={"subject_id": "speaker", "voice_start": "t_start"},
            inplace=True,
        )
        if self.partition == "enrollment":
            sel_column = f"{self.subset}.enroll"
            df_segs["duration"] = df_segs["voice_end"] - df_segs["t_start"]
        else:
            sel_column = ".".join([self.subset, self.condition, "test"])
            df_segs.drop(
                columns=[
                    "t_start",
                    "voice_end",
                    "face_frame",
                    "face_x",
                    "face_y",
                    "face_h",
                    "face_w",
                ],
                inplace=True,
            )

        sel_idx = df_segs[sel_column] == "Y"
        df_segs = df_segs.loc[sel_idx]
        df_segs["id"] = df_segs["video"].apply(
            lambda x: "janus-" + re.sub(".*/", "", x)
        )
        df_segs["video"] = df_segs["video"].apply(lambda x: self.corpus_dir / x)
        df_segs["speaker"] = df_segs["speaker"].astype(str)
        df_segs["language"] = "eng"
        df_segs["dataset"] = self.dataset_name()
        df_segs["corpusid"] = self.dataset_name()
        df_segs["source_type"] = "afv"
        df_segs.set_index("id", drop=False, inplace=True)
        # remove empty file
        df_segs.drop(index=["janus-2931.mp4"], inplace=True, errors="ignore")
        return df_segs

    def make_recording_set(self, df_segs: pd.DataFrame) -> RecordingSet:
        """
        Builds the RecordingSet from filtered segments.

        Args:
            df_segs (pd.DataFrame): DataFrame with segment metadata.

        Returns:
            RecordingSet: Audio recordings manifest.
        """

        logging.info("making RecordingSet")
        df_recs = df_segs[["id"]].copy()
        if self.target_sample_freq is not None:
            ar_opt = f"-ar {self.target_sample_freq} "
            df_recs["sample_freq"] = self.target_sample_freq
        else:
            ar_opt = ""
            df_recs["sample_freq"] = 44100

        df_recs["storage_path"] = df_segs["video"].apply(
            lambda x: f"ffmpeg -v 8 -i {x} -vn {ar_opt}-ac 1 -f wav - |"
        )

        recordings = RecordingSet(df_recs)
        recordings.get_durations(self.num_threads)
        return recordings

    def make_video_set(self, df_segs: pd.DataFrame) -> VideoSet:
        """
        Builds the VideoSet if `with_videos` is enabled.

        Args:
            df_segs (pd.DataFrame): DataFrame with segment metadata.

        Returns:
            VideoSet: Video manifest.
        """

        logging.info("making VideoSet")
        df_vids = df_segs[["id"]].copy()
        df_vids["storage_path"] = df_segs["video"]

        videos = VideoSet(df_vids)
        videos.get_metadata(self.num_threads)
        if self.target_sample_freq is not None:
            videos["target_sample_freq"] = self.target_sample_freq
        return videos

    def make_class_infos(self, df_segs: pd.DataFrame) -> Dict[str, ClassInfo]:
        """
        Extracts speaker, language, and source class labels from segments.

        Args:
            df_segs (pd.DataFrame): Segment metadata.

        Returns:
            Dict[str, ClassInfo]: Dictionary of class info tables.
        """
        logging.info("making ClassInfos")
        df_segs = df_segs.reset_index(drop=True)
        df_spks = df_segs[["speaker"]].drop_duplicates()
        df_spks.rename(columns={"speaker": "id"}, inplace=True)
        df_spks.sort_values(by="id", inplace=True)
        speakers = ClassInfo(df_spks)

        languages = ClassInfo(pd.DataFrame({"id": ["eng"]}))
        sources = ClassInfo(pd.DataFrame({"id": ["afv"]}))
        return {
            "speaker": speakers,
            "language": languages,
            "source_type": sources,
        }

    def make_enrollments(self, df_segs: pd.DataFrame) -> Dict[str, EnrollmentMap]:
        """
        Builds enrollment map using 1:1 segment-to-model assumption.

        Args:
            df_segs (pd.DataFrame): DataFrame with segment metadata.

        Returns:
            Dict[str, EnrollmentMap]: Dictionary with one enrollment list.
        """
        logging.info("making Enrollment")
        seg_ids = df_segs["id"]
        df_enr = pd.DataFrame({"modelid": seg_ids, "segmentid": seg_ids})
        return {"enrollment": EnrollmentMap(df_enr)}

    def make_trials(self) -> Dict[str, Path]:
        """
        Builds trial list(s) by parsing the Janus trial protocol file.

        Args:
            df_segs (pd.DataFrame): Segment metadata (not directly used here).

        Returns:
            Dict[str, Path]: Paths to generated trial CSV or TSV files.
        """
        logging.info("making Trials")
        trial_file = self.corpus_dir / "docs" / f"janus_multimedia_trials.tsv"

        df_trial = pd.read_csv(trial_file, sep="\t")
        df_trial.rename(
            columns={
                "ENROLL_FILE": "modelid",
                "TEST_FILE": "segmentid",
                "LABEL": "targettype",
            },
            inplace=True,
        )
        df_trial["modelid"] = df_trial["modelid"].apply(
            lambda x: "janus-" + re.sub(".*/", "", x)
        )
        df_trial["segmentid"] = df_trial["segmentid"].apply(
            lambda x: "janus-" + re.sub(".*/", "", x)
        )
        # remove empty file
        df_trial = df_trial[~df_trial["segmentid"].isin(["janus-2931.mp4"])]
        df_trial["targettype"] = df_trial["targettype"].apply(
            lambda x: "target" if x == "Y" else "nontarget"
        )
        sel_cond = self.subset.upper() + ".CORE"
        sel_core_idx = df_trial["CONDITION"] == sel_cond
        sel_cond = self.subset.upper() + ".FULL"
        sel_full_idx = df_trial["CONDITION"] == sel_cond
        df_trial.drop(columns=["CONDITION"], inplace=True)
        df_trial_core = df_trial.loc[sel_core_idx]
        df_trial_full = df_trial.loc[sel_full_idx]
        if self.condition == "core":
            output_file = self.output_dir / "trials.tsv"
            df_trial_core.to_csv(output_file, sep="\t", index=False)
            return {"trials": output_file}
        else:
            output_file_core = self.output_dir / "trials_core.tsv"
            df_trial_core.to_csv(output_file_core, sep="\t", index=False)
            output_file_full = self.output_dir / "trials_full.tsv"
            df_trial_full.to_csv(output_file_full, sep="\t", index=False)
            return {"trials_core": output_file_core, "trials_full": output_file_full}

    def prepare(self) -> None:
        """
        Runs the full data preparation pipeline:
        - Loads metadata
        - Generates audio/video manifests
        - Generates segment tables, class labels
        - Builds trials and/or enrollment maps
        - Saves HypDataset object to disk
        """

        logging.info(
            "Peparing Janus %s %s %s corpus_dir: %s -> data_dir: %s",
            self.subset,
            self.condition,
            self.partition,
            self.corpus_dir,
            self.output_dir,
        )
        df_segs = self.read_segments_metadata()
        recs = self.make_recording_set(df_segs)
        if self.with_videos:
            videos = self.make_video_set(df_segs)
        else:
            videos = None

        if self.partition == "test":
            df_segs["duration"] = recs.loc[df_segs["id"], "duration"].values

        classes = self.make_class_infos(df_segs)

        if self.partition == "enrollment":
            enrollments = self.make_enrollments(df_segs)
            trials = None
        else:
            enrollments = None
            trials = self.make_trials(df_segs)

        df_segs.drop(columns=["video"], inplace=True)
        segments = SegmentSet(df_segs)

        logging.info("making dataset")
        dataset = HypDataset(
            segments,
            classes,
            recordings=recs,
            videos=videos,
            enrollments=enrollments,
            trials=trials,
            sparse_trials=False,
        )
        logging.info("saving dataset at %s", self.output_dir)
        dataset.save(self.output_dir)
        dataset.describe()
