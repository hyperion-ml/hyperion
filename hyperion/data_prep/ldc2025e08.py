"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from jsonargparse import ActionYesNo

from hyperion.utils.langcodes import language_to_alpha3

from ..utils import ClassInfo, HyperDataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike
from .data_prep import DataPrep


class LDC2025E08DataPrep(DataPrep):
    """
    Prepares the LDC2025E08 ARTS Phase I Eval2 Initialization Sample dataset.

    Supports segment and recording metadata extraction, class info generation,
    and construction of a HyperDataset for 'init-speaker' and 'init-understand' partitions.

    Attributes:
        corpus_dir (PathLike): Root directory of the dataset.
        output_dir (PathLike): Directory where the prepared dataset is saved.
        extended_meta_file (Optional[PathLike]): Optional path to extended metadata CSV file.
        target_sample_freq (Optional[int]): Optional audio resampling frequency.
        num_threads (int): Number of threads for parallel duration computation.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        output_dir: PathLike,
        extended_meta_file: Optional[PathLike] = None,
        use_kaldi_ids: bool = False,
        target_sample_freq: Optional[int] = None,
        num_threads: int = 10,
    ):
        super().__init__(
            corpus_dir, output_dir, use_kaldi_ids, target_sample_freq, num_threads
        )
        self.corpus_dir = (
            self.corpus_dir / "ARTS_Phase1_Eval2_Initialization_Dataset_V1.0"
        )
        self.extended_meta_file = (
            Path(extended_meta_file)
            if extended_meta_file
            else self.corpus_dir / "docs" / "phase1-3_ta1_initmeta_04072026.csv"
        )

    @staticmethod
    def dataset_name() -> str:
        """Returns the dataset name identifier."""
        return "ldc2025e08"

    @staticmethod
    def add_class_args(parser) -> None:
        """Adds command-line arguments specific to the LDC2025E05 data preparation."""
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--extended-meta-file",
            type=Path,
            default=None,
            help="Path to extended metadata CSV file with additional speaker info.",
        )

    @staticmethod
    def _to_arts_dialect(df_segs: pd.DataFrame) -> pd.Series:
        return df_segs.apply(
            lambda row: (
                "USA"
                if row["native_language"] == "English" and row["dialects"] != "British"
                else (
                    "SSBE"
                    if row["native_language"] == "English"
                    and row["dialects"] == "British"
                    else "L1SPAN" if row["native_language"] == "Spanish" else "L1HINDI"
                )
            ),
            axis=1,
        )

    def _read_docs(self) -> pd.DataFrame:
        """
        Reads and parses the initialization metadata CSV file.

        Returns:
            pd.DataFrame: Segment metadata.
        """
        logging.info("loading docs")

        docs_dir = self.corpus_dir / "docs"
        if self.extended_meta_file.is_file():
            logging.info("Using extended metadata file: %s", self.extended_meta_file)
            # segment_id,speaker_id,channel,native_language,sex,ethnicity,dialects,age,age_group,country_of_birth,state_of_birth,city_of_birth,country_of_residence,state_of_residence,city_of_residence,sample_rate,num_channels
            # d407c8b2e099,548cf5f77b3243f2bd9cf29e631ff63e,a,Spanish,F,Hispanic,,56,Senior(55-85),Chile,,Concepcion,Chile,,Coquimbo,16,1
            # a289b2a87cf1,5a1b6748b135d225422a66c56ebb30b6,a,Spanish,M,Hispanic,,54,Adult(25-54),Colombia,Valle,Sevilla,Colombia,Valle,Sevilla,16,1
            seg_file = self.extended_meta_file
        else:
            # segment_id,speaker_id,channel,sample_rate
            # c43ff7e1306a,e77716f701452b971d0fc9c4bddcc23d,a,8
            # c20a8ec6de46,191765469cd53a1585b38150cbcdbb21,b,8
            # 63083701b09c,0a9e00949fc72b89847c5b179ea7df38,b,8
            seg_file = docs_dir / "phase1-2_ta1_initmeta.csv"

        df_segs = pd.read_csv(seg_file, sep=",")
        if "num_channels" not in df_segs.columns:
            df_segs["num_channels"] = df_segs["sample_rate"].apply(
                lambda sr: 1 if sr == 16 else 2
            )

        df_segs.rename(
            columns={"segment_id": "id", "speaker_id": "speaker"}, inplace=True
        )
        df_segs["channel"] = (
            df_segs["channel"].str.lower().str.replace("ab", "a", regex=False)
        )
        df_segs["gender"] = df_segs["sex"].str.lower()
        df_segs["ta2_dialect"] = self._to_arts_dialect(df_segs)
        df_segs["native_language"] = self._language_to_alpha3(
            df_segs["native_language"]
        )
        df_segs["ethnicity"] = df_segs["ethnicity"].str.lower()
        df_segs["dialects"] = df_segs["dialects"].str.lower()
        df_segs["arts_age_group"] = self._age_to_arts_age_group(df_segs["age"])
        df_segs.drop(columns=["sex"], inplace=True)
        return df_segs

    def make_recording_set_old(self, df_segs: pd.DataFrame) -> RecordingSet:
        """
        Builds the RecordingSet table from segment metadata.

        Args:
            df_segs (pd.DataFrame): Metadata for segments.

        Returns:
            RecordingSet: Table of audio file paths and durations.
        """
        logging.info("making RecordingSet")
        wav_dir = self.corpus_dir / "data"

        def channel_to_num(c):
            if c == "a" or c == "ab":
                return 1
            elif c == "b":
                return 2
            else:
                raise ValueError(f"Invalid channel: {c}")

        for s in df_segs["id"]:
            path = wav_dir / f"arts_{s}.wav"
            if not path.exists():
                logging.warning("Missing audio file: %s", path)

        paths = [wav_dir / f"arts_{s}.wav" for s in df_segs["id"]]
        paths = [
            f"sox {p} -c 1 -t wav - remix {channel_to_num(c)} |" if sr == 8 else p
            for p, c, sr in zip(paths, df_segs["channel"], df_segs["sample_rate"])
        ]
        df_recs = pd.DataFrame(
            {
                "id": df_segs["id"],
                "storage_path": paths,
                "sample_freq": df_segs["sample_rate"] * 1000,
            }
        )
        if self.target_sample_freq is not None:
            df_recs["target_sample_freq"] = self.target_sample_freq

        recordings = RecordingSet(df_recs)
        recordings.get_durations(self.num_threads)
        return recordings

    def make_recording_set(self, df_segs: pd.DataFrame) -> RecordingSet:
        """
        Builds the RecordingSet table from segment metadata.

        Args:
            df_segs (pd.DataFrame): Metadata for segments.

        Returns:
            RecordingSet: Table of audio file paths and durations.
        """
        logging.info("making RecordingSet")
        wav_dir = self.corpus_dir / "data"

        for s in df_segs["id"]:
            path = wav_dir / f"arts_{s}.wav"
            if not path.exists():
                logging.warning("Missing audio file: %s", path)

        paths = [wav_dir / f"arts_{s}.wav" for s in df_segs["id"]]
        df_recs = pd.DataFrame(
            {
                "id": df_segs["id"],
                "storage_path": paths,
                "sample_freq": df_segs["sample_rate"] * 1000,
                "channel": df_segs["channel"],
            }
        )
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
        if "dialects" in df_segs.columns:
            df_spks = df_segs[
                [
                    "speaker",
                    "gender",
                    "native_language",
                    "ta2_dialect",
                    "dialects",
                    "ethnicity",
                    "age",
                    "age_group",
                    "arts_age_group",
                    "country_of_birth",
                    "state_of_birth",
                    "city_of_birth",
                    "country_of_residence",
                    "state_of_residence",
                    "city_of_residence",
                ]
            ].drop_duplicates()
        else:
            df_spks = df_segs[["speaker"]].drop_duplicates().dropna()
        df_spks.rename(columns={"speaker": "id"}, inplace=True)
        df_spks.sort_values(by="id", inplace=True)
        speakers = ClassInfo(df_spks)

        languages = ClassInfo(pd.DataFrame({"id": ["eng"]}))
        sources = ClassInfo(pd.DataFrame({"id": ["cts", "intv"]}))
        return {
            "speaker": speakers,
            "language": languages,
            "source_type": sources,
        }

    def prepare(self) -> None:
        """
        Runs the full LDC2025E05 preparation pipeline:
        - Loads and parses segment metadata.
        - Builds the RecordingSet and computes durations.
        - Constructs ClassInfo and SegmentSet objects.
        - Writes the HyperDataset to disk.
        """
        logging.info(
            "Preparing LDC2025E08 corpus_dir: %s -> data_dir: %s",
            self.corpus_dir,
            self.output_dir,
        )
        df_segs = self._read_docs()
        recs = self.make_recording_set(df_segs)
        df_segs["duration"] = recs.loc[df_segs["id"], "duration"].values
        df_segs["language"] = "eng"
        df_segs["source_type"] = df_segs["sample_rate"].apply(
            lambda x: "cts" if x == 8 else "intv"
        )
        df_segs.drop(columns=["sample_rate", "channel"], inplace=True)
        df_segs["dataset"] = self.dataset_name()

        classes = self.make_class_infos(df_segs)

        segments = SegmentSet(df_segs)

        logging.info("making dataset")
        dataset = HyperDataset(
            segments,
            classes,
            recordings=recs,
        )
        logging.info("saving dataset at %s", self.output_dir)
        dataset.save(self.output_dir)
        dataset.describe()
