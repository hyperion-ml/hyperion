"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..utils import ClassInfo, EnrollmentMap, HyperDataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike
from .data_prep import DataPrep


class VoicesChallengeDataPrep(DataPrep):
    """
    Prepares the Voices Challenge dataset into a structured HyperDataset format.

    Attributes:
        corpus_dir: Root directory of the dataset.
        partition: One of ['enrollment', 'test'].
        output_dir: Directory to save processed dataset.
        use_kaldi_ids: Whether to format IDs as <speaker>-<segment_id>.
        target_sample_freq: Optional target audio sampling frequency.
        num_threads: Number of threads to use for duration extraction.
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

    @staticmethod
    def dataset_name() -> str:
        """Returns the dataset name."""
        return "voices_challenge"

    @staticmethod
    def add_class_args(parser) -> None:
        """Adds CLI arguments specific to the Voices Challenge dataset."""
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--partition",
            choices=["enrollment", "test"],
            required=True,
            help="Which partition of the Voices Challenge to prepare.",
        )

    def read_segments(self) -> pd.DataFrame:
        """
        Loads the segment metadata from `docs/segments.tsv`.

        Returns:
            pd.DataFrame: Segment metadata indexed by segment ID.
        """
        path = Path(self.corpus_dir) / "docs" / "segments.tsv"
        df = pd.read_csv(path, sep="\t")
        df["id"] = df["segment_id"]
        if self.use_kaldi_ids:
            df["id"] = df["speaker"] + "-" + df["segment_id"]
        df["filename"] = df["segment_id"] + ".wav"
        df["dataset"] = self.dataset_name()
        df["corpusid"] = "voices"
        return df.set_index("id", drop=False)

    def read_speaker_labels(self) -> pd.DataFrame:
        """
        Reads speaker-level metadata (e.g. gender) from `docs/speaker_labels.tsv`.

        Returns:
            pd.DataFrame: Speaker-level metadata indexed by speaker ID.
        """
        spk_file = Path(self.corpus_dir) / "docs" / "speaker_labels.tsv"
        df_spk = pd.read_csv(spk_file, sep="\t")
        df_spk["speaker"] = df_spk["speaker"].astype(str)
        df_spk["gender"] = df_spk["gender"].str.lower()
        return df_spk.set_index("speaker")

    def make_recording_set(self, df: pd.DataFrame) -> RecordingSet:
        """
        Constructs the RecordingSet from segment metadata.

        Args:
            df: Segment DataFrame.

        Returns:
            RecordingSet: Table of recordings and durations.
        """
        audio_dir = Path(self.corpus_dir) / "data" / "audio"
        df_rec = df[["id"]].copy()
        df_rec["storage_path"] = df["filename"].apply(lambda x: str(audio_dir / x))
        df_rec["sample_freq"] = 16000
        if self.target_sample_freq:
            df_rec["target_sample_freq"] = self.target_sample_freq
        recs = RecordingSet(df_rec)
        recs.get_durations(self.num_threads)
        return recs

    def make_class_infos(self, df: pd.DataFrame) -> Dict[str, ClassInfo]:
        """
        Constructs ClassInfo for speakers.

        Args:
            df: Segment metadata including speaker info.

        Returns:
            Dict[str, ClassInfo]: ClassInfo tables keyed by class type.
        """
        speakers = df[["speaker"]].drop_duplicates().rename(columns={"speaker": "id"})
        return {"speaker": ClassInfo(speakers)}

    def make_enrollments(self, df: pd.DataFrame) -> Dict[str, EnrollmentMap]:
        """
        Builds the EnrollmentMap from `docs/enrollment.tsv`.

        Args:
            df: Segment metadata (used for Kaldi ID mapping if needed).

        Returns:
            Dict[str, EnrollmentMap]: Enrollment mapping keyed by 'enrollment'.
        """
        path = Path(self.corpus_dir) / "docs" / "enrollment.tsv"
        df_enr = pd.read_csv(path, sep="\t")
        if self.use_kaldi_ids:
            df_enr["segmentid"] = df_enr["speaker"] + "-" + df_enr["segmentid"]
        return {"enrollment": EnrollmentMap(df_enr)}

    def make_trials(self, df: pd.DataFrame) -> Dict[str, Path]:
        """
        Builds the trials file from `docs/trials.tsv`.

        Args:
            df: Segment metadata (used for Kaldi ID mapping if needed).

        Returns:
            Dict[str, Path]: Path to saved trial file.
        """
        path = Path(self.corpus_dir) / "docs" / "trials.tsv"
        df_trial = pd.read_csv(path, sep="\t")
        if self.use_kaldi_ids:
            df_trial["segmentid"] = df_trial["speaker"] + "-" + df_trial["segmentid"]
        output = Path(self.output_dir) / "trials.tsv"
        df_trial.to_csv(output, sep="\t", index=False)
        return {"trials": output}

    def prepare(self) -> None:
        """
        Executes the full data preparation pipeline for Voices Challenge.
        """
        logging.info("Preparing Voices Challenge partition: %s", self.partition)
        df = self.read_segments()
        spk_info = self.read_speaker_labels()
        df = df.join(spk_info, on="speaker")

        recs = self.make_recording_set(df)
        df["duration"] = df["id"].map(recs.set_index("id")["duration"])
        segments = SegmentSet(df.drop(columns=["filename", "segment_id"]))
        segments["original_bandwidth"] = 8000
        segments.sort()
        classes = self.make_class_infos(df)

        enrollments = (
            self.make_enrollments(df) if self.partition == "enrollment" else None
        )
        trials = self.make_trials(df) if self.partition == "test" else None

        dataset = HyperDataset(
            segments=segments,
            recordings=recs,
            classes=classes,
            enrollments=enrollments,
            trials=trials,
            sparse_trials=False,
        )
        dataset.save(self.output_dir)
        dataset.describe()
