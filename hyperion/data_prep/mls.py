"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import glob
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pycountry

from ..utils import ClassInfo, HypDataset, ParallelFileFinder, RecordingSet, SegmentSet
from ..utils.misc import PathLike
from ..utils.scp_list import SCPList
from .data_prep import DataPrep


class MLSDataPrep(DataPrep):
    """
    Prepares the MLS (Multilingual LibriSpeech) dataset into structured tables for training or evaluation.

    This class reads the metadata from `.tsv` files, matches it with audio recordings,
    extracts durations, and builds `RecordingSet`, `SegmentSet`, and class tables for speaker and language.

    Attributes:
        corpus_dir (PathLike): Path to the root MLS directory (e.g., contains folders like 'en/', 'fr/').
        language (str): Language code (e.g., 'en', 'de', 'fr') corresponding to the subdirectory in corpus_dir.
        split (str): Data split to use: 'train', 'dev', or 'test'.
        output_dir (PathLike): Directory to save the processed dataset files.
        use_kaldi_ids (bool): Whether to prepend speaker ID to each segment ID.
        target_sample_freq (Optional[int]): Optional sample rate to convert audio to.
        num_threads (int): Number of parallel threads for duration extraction.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        language: str,
        subset: str,
        output_dir: PathLike,
        use_kaldi_ids: bool = False,
        target_sample_freq: Optional[int] = None,
        num_threads: int = 10,
    ) -> None:
        """
        Initializes the MLS dataset preparation class.

        Args:
            corpus_dir (PathLike): Base directory containing the MLS dataset.
            language (str): Language code (e.g., 'en').
            split (str): Dataset split: 'train', 'dev', or 'test'.
            output_dir (PathLike): Directory where processed output will be saved.
            use_kaldi_ids (bool): Whether to prepend speaker ID to segment IDs.
            target_sample_freq (Optional[int]): Resample audio to this frequency if specified.
            num_threads (int): Number of threads for parallel audio duration extraction.
        """
        super().__init__(
            corpus_dir, output_dir, use_kaldi_ids, target_sample_freq, num_threads
        )
        self.language = language
        self.language_iso = pycountry.languages.get(name=language).alpha_3.lower()
        self.subset = subset

    @staticmethod
    def dataset_name() -> str:
        """
        Returns:
            str: Dataset name identifier used in metadata.
        """
        return "mls"

    @staticmethod
    def add_class_args(parser) -> None:
        """
        Adds MLS-specific arguments to a CLI parser.

        Args:
            parser (ArgumentParser): The argument parser to which MLS-specific arguments will be added.
        """
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--language",
            default="english",
            choices=[
                "dutch",
                "english",
                "french",
                "german",
                "italian",
                "polish",
                "portuguese",
                "spanish",
            ],
            help="MLS language",
        )

        parser.add_argument(
            "--subset",
            choices=["train", "dev", "test"],
            required=True,
            help="Dataset split.",
        )

    def _read_metadata(self, file_path: PathLike) -> pd.DataFrame:
        """
        Reads the metadata from the specified TSV file.

        Args:
            tsv_file (PathLike): Path to the TSV file containing metadata.

        Returns:
            pd.DataFrame: DataFrame containing the metadata.
        """
        logging.info("Reading metadata from %s", file_path)
        # # Read the raw file
        # with open("your_file.txt", "r", encoding="utf-8") as f:
        #     lines = f.readlines()

        # # Replace multi-char delimiters with single-character tokens *outside quotes*
        # import re

        # def split_line(line):
        #     # Replace the delimiter only outside quotes
        #     pattern = r'(?=(?:[^"]*"[^"]*")*[^"]*$)'  # regex trick: match only outside quotes
        #     return re.split(r'\s*\|\|\s*' + pattern, line.strip())

        # # Split all lines
        # rows = [split_line(line) for line in lines]
        # print(rows, flush=True)
        # # Convert to DataFrame
        # df = pd.DataFrame(rows[1:], columns=rows[0])
        df = pd.read_csv(
            file_path, sep="\s\|\s", skipinitialspace=True, engine="python"
        )
        # SPEAKER   |   GENDER   | PARTITION  |  MINUTES   |  BOOK ID   |             TITLE              |            CHAPTER

        # Clean column names and values
        df.columns = df.columns.str.strip()
        df = df.apply(
            lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x)
        )
        # df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        df = df.rename(
            columns={
                "SPEAKER": "speaker",
                "GENDER": "gender",
                "PARTITION": "subset",
                "MINUTES": "minutes",
                "BOOK ID": "book",
                "TITLE": "book_title",
                "CHAPTER": "chapter_title",
            }
        )
        df["speaker"] = df["speaker"].apply(lambda x: f"libri-{x}")
        df["gender"] = df["gender"].str.lower()
        df["book"] = df["book"].astype(str)
        df["language"] = self.language_iso
        return df

    def _read_transcripts(self, file_path: PathLike) -> pd.DataFrame:
        """
        Reads the transcripts from the specified text file.

        Args:
            file_path (PathLike): Path to the text file containing transcripts.

        Returns:
            pd.DataFrame: DataFrame containing the transcripts.
        """
        logging.info("Reading transcripts from %s", file_path)
        scp = SCPList.load(file_path, sep="\t")
        df = pd.DataFrame({"id": scp.key, "transcript": scp.file_path})
        df["id"] = df["id"].apply(lambda x: f"mls-{x}")
        return df

    def prepare(self) -> None:
        """
        Executes the full MLS data preparation pipeline:

        - Reads metadata from the TSV file (e.g., 'train.tsv').
        - Builds full paths to audio files and validates them.
        - Extracts audio durations using parallel processing.
        - Constructs RecordingSet, SegmentSet, and ClassInfo tables.
        - Saves the dataset to the specified output directory.
        """
        logging.info(
            "Preparing MLS dataset lang=%s split=%s corpus_dir=%s -> data_dir=%s",
            self.language,
            self.subset,
            self.corpus_dir,
            self.output_dir,
        )

        base_dir = self.corpus_dir / f"mls_{self.language}"
        meta_file = base_dir / "metainfo.txt"
        df_meta = self._read_metadata(meta_file)
        base_dir = base_dir / self.subset

        trans_file = base_dir / f"transcripts.txt"
        df_trans = self._read_transcripts(trans_file)

        rec_dir = base_dir / "audio"

        assert rec_dir.is_dir(), f"Missing audio directory: {rec_dir}"
        logging.info("searching audio files in %s", str(rec_dir))
        # rec_files = list(rec_dir.glob("**/*.flac"))
        # if not rec_files:
        #     # symlinks? try glob
        #     rec_files = [
        #         Path(f) for f in glob.iglob(f"{rec_dir}/**/*.flac", recursive=True)
        #     ]
        file_finder = ParallelFileFinder(
            root=rec_dir,
            pattern=r".*\.flac$",
            num_threads=self.num_threads,
        )
        rec_files = file_finder()
        assert len(rec_files) > 0, "recording files not found"

        speakers = ["libri-" + f.parent.parent.name for f in rec_files]
        books = [f.parent.name for f in rec_files]
        rec_ids = ["mls-" + f.with_suffix("").name for f in rec_files]
        if self.use_kaldi_ids:
            rec_ids = [f"{s}-{f}" for f, s in zip(rec_ids, speakers)]

        file_paths = [str(r) for r in rec_files]
        logging.info("making RecordingSet")
        recs = pd.DataFrame({"id": rec_ids, "storage_path": file_paths})
        print("recs", recs, flush=True)
        recs = RecordingSet(recs)
        recs.sort()

        recs.get_durations(self.num_threads)
        if self.target_sample_freq:
            recs["target_sample_freq"] = self.target_sample_freq

        logging.info("Creating SegmentSet")
        df_segs = pd.DataFrame({"id": rec_ids, "speaker": speakers, "book": books})
        df_segs["duration"] = recs.loc[df_segs["id"], "duration"].values
        df_segs["language"] = self.language_iso
        df_segs = df_segs.merge(df_meta, on=["speaker", "book"], how="left")
        df_segs.drop(columns=["language_y"], inplace=True)
        df_segs.rename(columns={"language_x": "language"}, inplace=True)
        df_segs = df_segs.merge(df_trans, on="id", how="left")
        print("segs", df_segs, flush=True)
        segments = SegmentSet(
            df_segs[
                [
                    "id",
                    "speaker",
                    "gender",
                    "transcript",
                    "duration",
                    "language",
                    "book",
                    "book_title",
                    "chapter_title",
                ]
            ].copy()
        )
        segments["corpusid"] = "librivox"
        segments["dataset"] = "mls"
        segments.sort()

        logging.info("Creating ClassInfo tables")
        speakers = ClassInfo(pd.DataFrame({"id": np.sort(df_segs["speaker"].unique())}))
        languages = ClassInfo(pd.DataFrame({"id": [self.language_iso]}))
        books = ClassInfo(pd.DataFrame({"id": np.sort(df_segs["book"].unique())}))
        genders = ClassInfo(pd.DataFrame({"id": ["m", "f"]}))

        logging.info("Saving dataset")
        dataset = HypDataset(
            segments=segments,
            recordings=recs,
            classes={
                "speaker": speakers,
                "gender": genders,
                "language": languages,
                "book": books,
            },
        )
        dataset.save(self.output_dir)
        dataset.describe()
