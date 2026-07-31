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

from ..utils import ClassInfo, HyperDataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike
from ..utils.scp_list import SCPList
from .data_prep import DataPrep


class LibriSpeechDataPrep(DataPrep):
    """
    Prepares the LibriSpeech dataset into structured tables.

    This class processes a specified subset of the LibriSpeech corpus, including
    speaker and chapter metadata, transcripts, and audio durations. It produces
    a dataset suitable for training or evaluating speech and speaker recognition systems.

    Attributes:
        corpus_dir (PathLike): Path to the base LibriSpeech directory.
        subset (str): Subset to process (e.g., "train-clean-100").
        output_dir (PathLike): Output directory for the prepared dataset.
        use_kaldi_ids (bool): Whether to prepend speaker ID to segment ID.
        target_sample_freq (Optional[int]): Optional resampling target frequency.
        num_threads (int): Number of threads to use for audio duration extraction.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        subset: str,
        output_dir: PathLike,
        use_kaldi_ids: bool = False,
        target_sample_freq: Optional[int] = None,
        num_threads: int = 10,
    ):
        """Initializes the LibriSpeech data preparation object."""
        super().__init__(corpus_dir, output_dir, False, target_sample_freq, num_threads)

        self.subset = subset

    @staticmethod
    def dataset_name() -> str:
        """Returns the dataset identifier name."""
        return "librispeech"

    @staticmethod
    def add_class_args(parser: ArgumentParser) -> None:
        """Adds LibriSpeech-specific arguments to an argument parser."""
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--subset",
            choices=[
                "train-clean-100",
                "train-clean-360",
                "train-other",
                "dev-clean",
                "dev-other",
                "test-clean",
                "test-other",
            ],
            help="""if we prepare the data for ["train-clean-100", "train-clean-360", "train-other", "dev-clean", "dev-other", "test-clean", "test-other"]""",
            required=True,
        )

    def _get_docs_dir(self) -> Path:
        """
        Resolves the location of the metadata documentation directory.

        Returns:
            Path: Path to the directory containing documentation files.
        """
        docs_dir = self.corpus_dir / "docs"
        if docs_dir.is_dir():
            return docs_dir
        else:
            return self.corpus_dir

    def _get_spks_metadata(self) -> pd.DataFrame:
        """
        Loads speaker metadata from SPEAKERS.TXT.

        Returns:
            pd.DataFrame: DataFrame with speaker ID and gender.
        """

        docs_dir = self._get_docs_dir()
        file_path = docs_dir / "SPEAKERS.TXT"
        # read in two parts because of bad formating of speakers.txt
        df = pd.read_csv(
            file_path,
            sep="\s*\|\s*",
            header=None,
            names=["id", "gender", "subset", "min", "name"],
            skiprows=12,
            engine="python",
            dtype={"id": str},
            on_bad_lines="skip",
        )

        df2 = pd.read_csv(
            file_path,
            sep="\s*\|\s*",
            header=None,
            names=["id", "gender", "subset", "min", "name", "d1", "d2"],
            skiprows=44,
            engine="python",
            dtype={"id": str},
            nrows=1,
        )
        df2.drop(columns=["d1", "d2"], inplace=True)
        df = pd.concat([df, df2], ignore_index=True)
        df["id"] = df["id"].apply(lambda x: f"libri-{x}")
        df = df.loc[df["subset"] == self.subset]
        df["gender"] = df["gender"].apply(lambda x: x.lower())
        df.drop(columns=["subset", "min", "name"], inplace=True)
        return df

    def _get_chapters_metadata(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads metadata for books and chapters.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Book and chapter metadata tables.
        """
        docs_dir = self._get_docs_dir()
        file_path = docs_dir / "BOOKS.TXT"
        df_books = pd.read_csv(
            file_path,
            sep="\s*\|\s*",
            header=None,
            names=["book", "book_title", "dummy"],
            engine="python",
            dtype={"book": str},
        )
        df_books.drop(columns=["dummy"], inplace=True)

        file_path = docs_dir / "CHAPTERS.TXT"
        df_chapters = pd.read_csv(
            file_path,
            sep="\s*\|\s*",
            header=None,
            names=[
                "chapter",
                "speaker",
                "mins",
                "subset",
                "proj",
                "book",
                "chapter_title",
                "proj_title",
            ],
            skiprows=14,
            engine="python",
            dtype={"chapter": str, "book": str, "speaker": str},
        )
        df_chapters = df_chapters.merge(df_books, how="left", on="book")
        df_chapters["speaker"] = df_chapters["speaker"].apply(lambda x: f"libri-{x}")
        df_chapters.drop(columns=["proj", "mins", "proj_title"], inplace=True)

        return df_books, df_chapters

    def _get_transcripts(self, rec_dir: Path) -> pd.DataFrame:
        """
        Loads transcript files and returns a mapping of ID to transcript text.

        Args:
            rec_dir (Path): Directory where audio and transcript files are located.

        Returns:
            pd.DataFrame: DataFrame with columns ['id', 'transcript'].
        """
        logging.info("searching transcript files in %s", str(rec_dir))
        trans_files = list(rec_dir.glob("**/*.trans.txt"))
        if not trans_files:
            # symlinks? try glob
            trans_files = [
                Path(f) for f in glob.iglob(f"{rec_dir}/**/.*trans.txt", recursive=True)
            ]

        assert len(trans_files) > 0, "transcript files not found"
        dfs = []
        for trans_file in trans_files:
            scp = SCPList.load(trans_file)
            df_i = pd.DataFrame({"id": scp.key, "transcript": scp.file_path})
            dfs.append(df_i)

        df_trans = pd.concat(dfs)
        df_trans["id"] = df_trans["id"].apply(lambda x: f"librispeech-{x}")
        return df_trans

    def prepare(self) -> None:
        """
        Executes the full LibriSpeech preparation pipeline:
        - Loads audio files and metadata (speakers, chapters, books, transcripts)
        - Extracts audio durations
        - Builds segment, recording, and class info tables
        - Writes the final dataset to output_dir
        """
        logging.info(
            "Peparing Libri %s corpus_dir:%s -> data_dir:%s",
            self.subset,
            self.corpus_dir,
            self.output_dir,
        )

        df_spks = self._get_spks_metadata()
        df_books, df_chapters = self._get_chapters_metadata()

        rec_dir = self.corpus_dir / self.subset
        logging.info("searching audio files in %s", str(rec_dir))
        rec_files = list(rec_dir.glob("**/*.flac"))
        if not rec_files:
            # symlinks? try glob
            rec_files = [
                Path(f) for f in glob.iglob(f"{rec_dir}/**/*.flac", recursive=True)
            ]

        assert len(rec_files) > 0, "recording files not found"

        df_trans = self._get_transcripts(rec_dir)

        speakers = ["libri-" + f.parent.parent.name for f in rec_files]
        chapters = [f.parent.name for f in rec_files]
        rec_ids = ["librispeech-" + f.with_suffix("").name for f in rec_files]
        if self.use_kaldi_ids:
            rec_ids = [f"{s}-{f}" for f, s in zip(rec_ids, speakers)]

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
        df_segs = pd.DataFrame(
            {"id": rec_ids, "speaker": speakers, "chapter": chapters}
        )
        df_segs = df_segs.merge(
            df_spks, how="left", left_on="speaker", right_on="id", suffixes=(None, "_y")
        )
        df_segs.drop(columns=["id_y"], inplace=True)
        df_segs = df_segs.merge(df_trans, how="left", on="id")
        df_segs["duration"] = recs.loc[df_segs["id"], "duration"].values
        df_segs = df_segs.merge(df_chapters, how="left", on="chapter")
        df_segs.rename(columns={"speaker_x": "speaker"}, inplace=True)
        df_segs.drop(columns=["speaker_y"], inplace=True)
        df_segs["language"] = "eng"
        df_segs["corpusid"] = "librivox"
        df_segs["dataset"] = "librispeech"
        df_segs["original_bandwidth"] = 8000
        segments = SegmentSet(df_segs)
        segments.sort()

        logging.info("making speaker info file")
        df_spks = df_spks.loc[df_spks["id"].isin(df_segs["speaker"].values)]
        speakers = ClassInfo(df_spks)

        logging.info("making book info file")
        df_books.rename(columns={"book": "id"}, inplace=True)
        df_books = df_books.loc[df_books["id"].isin(df_segs["book"].values)]
        books = ClassInfo(df_books)

        logging.info("making chapter info file")
        df_chapters.rename(columns={"chapter": "id"}, inplace=True)
        df_chapters = df_chapters.loc[df_chapters["id"].isin(df_segs["chapter"].values)]
        chapters = ClassInfo(df_chapters)

        logging.info("making language info file")
        languages = ClassInfo(pd.DataFrame({"id": ["eng"]}))
        logging.info("making gender info file")
        genders = ClassInfo(pd.DataFrame({"id": ["m", "f"]}))

        classes = {
            "speaker": speakers,
            "book": books,
            "chapter": chapters,
            "language": languages,
            "gender": genders,
        }

        logging.info("making dataset")
        dataset = HyperDataset(
            segments,
            classes=classes,
            recordings=recs,
        )
        logging.info("saving dataset at %s", self.output_dir)
        dataset.save(self.output_dir)
        dataset.describe()
