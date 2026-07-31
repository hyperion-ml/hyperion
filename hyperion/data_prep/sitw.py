import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..utils import ClassInfo, EnrollmentMap, HyperDataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike
from .data_prep import DataPrep


class SITWDataPrep(DataPrep):
    """
    Prepares the Speakers in the Wild (SITW) dataset into HyperDataset format.

    Supports preparation of either the enrollment or test partitions,
    reading segments, recordings, trials, and class info.

    Attributes:
        corpus_dir: Root SITW data directory.
        partition: One of ['enrollment', 'test'].
        output_dir: Directory to save the processed dataset.
        use_kaldi_ids: If True, format IDs as <speaker>-<segment_id>.
        target_sample_freq: Optional audio resampling rate in Hz.
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
        """Returns the dataset name string."""
        return "sitw"

    @staticmethod
    def add_class_args(parser) -> None:
        """Adds CLI arguments specific to SITW preparation."""
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--partition",
            choices=["enrollment", "test"],
            required=True,
            help="Dataset partition to prepare (enrollment or test)",
        )

    def read_segments(self) -> pd.DataFrame:
        """
        Reads segment metadata from `segments.tsv`.

        Returns:
            DataFrame indexed by segment ID.
        """
        seg_path = Path(self.corpus_dir) / "docs" / "segments.tsv"
        df = pd.read_csv(seg_path, sep="\t")
        df["id"] = df["segment_id"]
        if self.use_kaldi_ids:
            df["id"] = df["speaker"] + "-" + df["segment_id"]
        df["filename"] = df["segment_id"] + ".flac"
        df["dataset"] = self.dataset_name()
        df["corpusid"] = "sitw"
        return df.set_index("id", drop=False)

    def make_recording_set(self, df: pd.DataFrame) -> RecordingSet:
        """
        Creates a RecordingSet from the segment metadata.

        Args:
            df: Segment metadata.

        Returns:
            RecordingSet with audio paths and durations.
        """
        audio_dir = Path(self.corpus_dir) / "data"
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
        Builds ClassInfo for speakers.

        Args:
            df: Segment metadata.

        Returns:
            Dict of ClassInfo with 'speaker'.
        """
        df_spk = df[["speaker"]].drop_duplicates().rename(columns={"speaker": "id"})
        return {"speaker": ClassInfo(df_spk)}

    def make_enrollments(self, df: pd.DataFrame) -> Dict[str, EnrollmentMap]:
        """
        Reads and constructs the EnrollmentMap from TSV.

        Args:
            df: Segment metadata (used for Kaldi-ID conversion).

        Returns:
            Dict[str, EnrollmentMap]: 'enrollment' key with map.
        """
        enr_path = Path(self.corpus_dir) / "docs" / "enrollment.tsv"
        df_enr = pd.read_csv(enr_path, sep="\t")
        if self.use_kaldi_ids:
            df_enr["segmentid"] = df_enr["speaker"] + "-" + df_enr["segmentid"]
        return {"enrollment": EnrollmentMap(df_enr)}

    def make_trials(self, df: pd.DataFrame) -> Dict[str, Path]:
        """
        Loads and optionally modifies the trial key.

        Args:
            df: Segment metadata (for Kaldi-ID handling).

        Returns:
            Dictionary with key 'trials' and trial file path.
        """
        trial_path = Path(self.corpus_dir) / "docs" / "trials.tsv"
        df_trial = pd.read_csv(trial_path, sep="\t")
        if self.use_kaldi_ids:
            df_trial["segmentid"] = df_trial["speaker"] + "-" + df_trial["segmentid"]
        output = Path(self.output_dir) / "trials.tsv"
        df_trial.to_csv(output, sep="\t", index=False)
        return {"trials": output}

    def prepare(self) -> None:
        """
        Executes the full data preparation pipeline for SITW.
        """
        logging.info("Preparing SITW partition: %s", self.partition)

        df = self.read_segments()
        recs = self.make_recording_set(df)
        df["duration"] = df["id"].map(recs.set_index("id")["duration"])

        segments = SegmentSet(df.drop(columns=["filename", "segment_id"]))
        segments["original_bandwidth"] = 8000
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
