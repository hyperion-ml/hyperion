"""
Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from jsonargparse import ActionYesNo
from tqdm import tqdm

from ..utils import ClassInfo, EnrollmentMap, HyperDataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike
from .data_prep import DataPrep


class SRE19AVDataPrep(DataPrep):
    """
    Prepares the SRE19 Audio-Visual (AV) dataset into structured tables.

    Supports the audio, visual, or audio-visual modalities for both development (dev)
    and evaluation (eval) partitions, building metadata for recordings, segments, and
    optional enrollment/trial keys.

    Attributes:
        corpus_dir (PathLike): Base directory of the dataset.
        modality (str): One of 'audio', 'visual', or 'audio-visual'.
        subset (str): Either 'dev' or 'eval'.
        partition (str): Either 'enrollment' or 'test'.
        output_dir (PathLike): Path where processed dataset will be written.
        use_kaldi_ids (bool): If True, IDs are formatted as <speaker>-<segment>.
        target_sample_freq (Optional[int]): Target audio sample rate (Hz).
        num_threads (int): Number of threads for parallel processing.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        modality: str,
        subset: str,
        partition: str,
        output_dir: PathLike,
        use_kaldi_ids: bool = False,
        target_sample_freq: Optional[int] = None,
        num_threads: int = 10,
        # use_ldc_langs: bool = False,
    ):
        """
        Initialize the data preparation logic for SRE19 AV.

        Args:
            corpus_dir (PathLike): Root dataset directory.
            modality (str): 'audio', 'visual', or 'audio-visual'.
            subset (str): 'dev' or 'eval'.
            partition (str): 'enrollment' or 'test'.
            output_dir (PathLike): Where to save the processed dataset.
            use_kaldi_ids (bool): Format IDs with speaker prefix.
            target_sample_freq (Optional[int]): Resample frequency if specified.
            num_threads (int): Number of threads for audio duration extraction.
        """
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
    def dataset_name() -> str:
        """Returns dataset name identifier."""
        return "sre19_av"

    @staticmethod
    def add_class_args(parser) -> None:
        """
        Adds CLI arguments for configuring SRE19 AV data preparation.

        Args:
            parser: ArgumentParser instance.
        """

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

    def read_segments_metadata(self) -> pd.DataFrame:
        """
        Reads and preprocesses segment-level metadata from segment key TSV file.

        Returns:
            pd.DataFrame: Metadata with IDs, speaker, gender, filename, etc.
        """
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
        df_segs["language"] = "eng"
        df_segs["original_bandwidth"] = 8000
        if self.use_kaldi_ids:
            df_segs["id"] = df_segs[["speaker", "id"]].apply(
                lambda row: "-".join(row.values.astype(str)), axis=1
            )
        df_segs.set_index("id", drop=False, inplace=True)
        return df_segs

    def make_recording_set(self, df_segs: pd.DataFrame) -> RecordingSet:
        """
        Constructs RecordingSet with FFmpeg audio extraction pipes.

        Args:
            df_segs (pd.DataFrame): Segment metadata.

        Returns:
            RecordingSet: RecordingSet object with storage paths and durations.
        """
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

    def make_class_infos(self, df_segs: pd.DataFrame) -> dict[str, ClassInfo]:
        """
        Builds ClassInfo tables for speaker, language, source_type, and gender.

        Args:
            df_segs (pd.DataFrame): Segment metadata.

        Returns:
            dict[str, ClassInfo]: Class info tables.
        """

        logging.info("making ClassInfo tables")
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

    def make_enrollments(
        self, df_segs: pd.DataFrame
    ) -> Optional[dict[str, EnrollmentMap]]:
        """
        Builds EnrollmentMap for audio or audio-visual modalities.

        Args:
            df_segs (pd.DataFrame): Segment metadata.

        Returns:
            dict[str, EnrollmentMap] or None: Enrollment map if applicable.
        """

        logging.info("making EnrollmentMap")
        if self.modality in ["audio", "audio-visual"]:
            enroll_file = self.docs_dir / f"sre19_av_{self.subset}_enrollment.tsv"
            df_enr = pd.read_csv(enroll_file, sep="\t")
            if self.use_kaldi_ids:
                df_enr["speaker"] = [
                    df_segs.loc[df_segs["filename"] == s, "speaker"].values[0]
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

    def make_trials(self, df_segs: pd.DataFrame) -> dict[str, Path]:
        """
        Builds trial key and returns path to trials file.

        Args:
            df_segs (pd.DataFrame): Segment metadata.

        Returns:
            dict[str, Path]: Mapping of trial set name to TSV path.
        """
        logging.info("Building Trials")
        trial_file = self.docs_dir / f"sre19_av_{self.subset}_trial_key.tsv"

        df_trial = pd.read_csv(trial_file, sep="\t")
        if self.use_kaldi_ids:
            df_trial["speaker"] = [
                df_segs.loc[df_segs["filename"] == s, "speaker"].values[0]
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

    def prepare(self) -> None:
        """
        Executes the full SRE19 AV data preparation pipeline:
        - Loads metadata
        - Builds segment and recording sets
        - Prepares class info tables
        - Generates enrollment and/or trial files
        - Saves final HyperDataset to disk
        """

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
        dataset = HyperDataset(
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
