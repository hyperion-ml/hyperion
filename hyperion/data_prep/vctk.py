"""
Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import glob
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from jsonargparse import ArgumentParser

from ..utils import ClassInfo, HypDataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike
from ..utils.scp_list import SCPList
from .data_prep import DataPrep


class VCTKDataPrep(DataPrep):
    """
    Data preparation class for the VCTK dataset.

    Converts the raw VCTK corpus into structured metadata tables for training and evaluation,
    including speaker information, audio segment durations, language/dialect/region metadata,
    and text transcripts.

    Attributes:
        corpus_dir (PathLike): Path to the base VCTK directory.
        output_dir (PathLike): Output directory where the processed dataset is saved.
        use_kaldi_ids (bool): Whether to prepend speaker ID to segment ID (Kaldi-style IDs).
        target_sample_freq (Optional[int]): Optional target sampling frequency for audio.
        num_threads (int): Number of threads used for parallel duration computation.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        output_dir: PathLike,
        use_kaldi_ids: bool = False,
        target_sample_freq: Optional[int] = None,
        num_threads: int = 10,
    ):
        """Initializes the VCTK data preparation object."""
        super().__init__(corpus_dir, output_dir, False, target_sample_freq, num_threads)

    @staticmethod
    def dataset_name() -> str:
        """Returns the dataset identifier name."""
        return "vctk"

    @staticmethod
    def add_class_args(parser: ArgumentParser) -> None:
        """Adds VCTK-specific arguments to an argument parser."""
        DataPrep.add_class_args(parser)

    def _get_spks_metadata(self) -> pd.DataFrame:
        """
        Loads speaker metadata from `speaker-info.txt`.

        Returns:
            pd.DataFrame: DataFrame containing speaker attributes including gender,
            accent, dialect, region, and country codes.
        """

        file_path = self.corpus_dir / "speaker-info.txt"
        columns = ["ID", "AGE", "GENDER", "ACCENTS", "REGION"]
        rows = []

        with open(file_path, "r") as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split()

                assert len(parts) >= 4
                id_, age, gender = parts[:3]

                if len(parts) == 4:
                    # Only ACCENTS is present, REGION is missing
                    accent = parts[3]
                    region = None
                elif len(parts) == 5:
                    # Assume single-word ACCENTS and REGION
                    accent = parts[3]
                    if accent == "Australian":
                        region = None
                    else:
                        region = parts[4]
                elif len(parts) >= 6:
                    # Try detecting if ACCENTS is multi-word
                    # Start by assuming ACCENTS is two words
                    accent = parts[3]
                    region = parts[-1]

                rows.append([id_, age, gender, accent, region])

        # Create DataFrame
        df = pd.DataFrame(rows, columns=columns)
        print(rows, df.head(), flush=True)
        df["AGE"] = df["AGE"].astype(int)
        # df = pd.read_csv(
        #     file_path,
        #     sep=" ",
        #     dtype={"ID": str, "AGE": int},
        # )
        df.rename(
            columns={
                "ID": "id",
                "AGE": "age",
                "GENDER": "gender",
                "ACCENTS": "accent",
                "REGION": "region",
            },
            inplace=True,
        )
        df["id"] = df["id"].apply(lambda x: f"vctk-{x}")
        df["language"] = "eng"
        df["accent"] = df["accent"].apply(lambda x: x.lower())
        df["region"] = df["region"].apply(lambda x: x.lower() if x else None)
        df["country"] = df["accent"].apply(
            lambda x: (
                "gbr"
                if x
                in [
                    "english",
                    "englishse",
                    "englishsurrey",
                    "scottish",
                    "welsh",
                    "northernirish",
                ]
                else x
            )
        )
        df["country"] = df["country"].replace(
            {
                "irish": "irl",
                "american": "usa",
                "canadian": "can",
                "australian": "aus",
                "newzealand": "nzl",
                "southafrican": "zaf",
                "indian": "ind",
            }
        )
        df["dialect"] = df["accent"].apply(
            lambda x: (
                "eng-gbr"
                if x in ["english", "englishse", "englishsurrey", "scottish", "welsh"]
                else x
            )
        )
        df["dialect"] = df["dialect"].apply(
            lambda x: "eng-irl" if x in ["irish", "northernirish"] else x
        )
        df["dialect"] = df["dialect"].replace(
            {
                "canadian": "eng-can",
                "american": "eng-usa",
                "australian": "eng-aus",
                "newzealand": "eng-nzl",
                "southafrican": "eng-zaf",
                "indian": "eng-ind",
            }
        )
        df["gender"] = df["gender"].apply(lambda x: x.lower())

        return df

    def _get_transcripts(self) -> pd.DataFrame:
        """
        Loads transcript files and maps them to recording IDs.

        Returns:
            pd.DataFrame: DataFrame with columns ['id', 'transcript'] for each audio segment.
        """
        trans_dir = self.corpus_dir / "txt"
        logging.info("searching transcript files in %s", str(trans_dir))
        trans_files = list(trans_dir.glob("**/*.txt"))
        if not trans_files:
            # symlinks? try glob
            trans_files = [
                Path(f) for f in glob.iglob(f"{trans_dir}/**/*.txt", recursive=True)
            ]

        assert len(trans_files) > 0, "transcript files not found"
        ids = []
        trans = []
        for trans_file in trans_files:
            id = f"vctk-{trans_file.stem}"
            with open(trans_file, "r", encoding="utf-8") as f:
                transcript = f.readlines()[0].strip().rstrip("\n")

            ids.append(id)
            trans.append(transcript)

        df_trans = pd.DataFrame({"id": ids, "transcript": trans})
        return df_trans

    def prepare(self) -> None:
        """
        Runs the full data preparation pipeline for VCTK:
        - Loads speaker metadata and transcripts
        - Scans audio files and extracts durations
        - Builds SegmentSet and RecordingSet tables
        - Constructs class tables for speaker attributes and metadata
        - Saves the resulting HypDataset to the output directory
        """
        logging.info(
            "Peparing VCTK corpus_dir:%s -> data_dir:%s",
            self.corpus_dir,
            self.output_dir,
        )

        df_spks = self._get_spks_metadata()
        rec_dir = self.corpus_dir / "wav48"
        logging.info("searching audio files in %s", str(rec_dir))
        rec_files = list(rec_dir.glob("**/*.wav"))
        if not rec_files:
            # symlinks? try glob
            rec_files = [
                Path(f) for f in glob.iglob(f"{rec_dir}/**/*.wav", recursive=True)
            ]

        assert len(rec_files) > 0, "recording files not found"

        df_trans = self._get_transcripts()

        speakers = ["vctk-" + f.parent.name[1:] for f in rec_files]
        rec_ids = ["vctk-" + f.with_suffix("").name for f in rec_files]

        file_paths = [str(r) for r in rec_files]
        logging.info("making RecordingSet")
        recs = pd.DataFrame({"id": rec_ids, "storage_path": file_paths})
        recs = RecordingSet(recs)
        recs.sort()

        logging.info("getting recording durations")
        recs.get_durations(self.num_threads)
        if self.target_sample_freq:
            recs["target_sample_freq"] = self.target_sample_freq

        logging.info("making SegmentsSet")
        df_segs = pd.DataFrame({"id": rec_ids, "speaker": speakers})
        df_segs = df_segs.merge(
            df_spks, how="left", left_on="speaker", right_on="id", suffixes=(None, "_y")
        )
        print(df_segs.head(), flush=True)
        df_segs.drop(columns=["id_y"], inplace=True)
        df_segs = df_segs.merge(df_trans, how="left", on="id")
        df_segs["duration"] = recs.loc[df_segs["id"], "duration"].values
        # df_segs.rename(columns={"speaker_x": "speaker"}, inplace=True)
        df_segs["dataset"] = "vctk"
        df_segs["corpusid"] = "vctk"
        segments = SegmentSet(df_segs)
        segments.sort()

        logging.info("making speaker info file")
        speakers = ClassInfo(df_spks)

        logging.info("making language/dialect info files")
        languages = ClassInfo(pd.DataFrame({"id": ["eng"]}))
        dialects = np.sort(df_spks["dialect"].dropna().unique())
        dialects = ClassInfo(pd.DataFrame({"id": dialects}))
        accents = np.sort(df_spks["accent"].dropna().unique())
        accents = ClassInfo(pd.DataFrame({"id": accents}))
        regions = np.sort(df_spks["region"].dropna().unique())
        regions = ClassInfo(pd.DataFrame({"id": regions}))
        countries = np.sort(df_spks["country"].dropna().unique())
        countries = ClassInfo(pd.DataFrame({"id": countries}))

        classes = {
            "speaker": speakers,
            "language": languages,
            "dialect": dialects,
            "accent": accents,
            "region": regions,
            "country": countries,
        }

        logging.info("making dataset")
        dataset = HypDataset(
            segments,
            classes=classes,
            recordings=recs,
        )
        logging.info("saving dataset at %s", self.output_dir)
        dataset.save(self.output_dir)
        dataset.describe()
