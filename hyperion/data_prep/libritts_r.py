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
from .data_prep import DataPrep


def _read_malformed_trans(path: PathLike) -> pd.DataFrame:
    """
    Load a TSV file into a DataFrame, fixing unbalanced quote issues in the process.

    Args:
        path (str or Path): Path to the original TSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame with cleaned rows.
    """
    logging.warning("reading malformed transcript file: %s", str(path))
    path = Path(path)
    fixed_path = Path.cwd() / f"{path.stem}.fixed.tmp"

    def is_balanced_quotes(s: str) -> bool:
        return s.count('"') % 2 == 0

    def fix_quotes(line: str) -> str:
        parts = line.strip("\n").split("\t")
        fixed_parts = []
        for part in parts:
            if not is_balanced_quotes(part):
                part = part.replace('"', "'")
            fixed_parts.append(part)
        return "\t".join(fixed_parts)

    # Step 1: Write the cleaned file
    with path.open("r", encoding="utf-8") as infile, fixed_path.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            if not is_balanced_quotes(line):
                line = fix_quotes(line)
            outfile.write(line if line.endswith("\n") else line + "\n")

    # Step 2: Read the cleaned file into a DataFrame
    df = pd.read_csv(
        fixed_path,
        sep="\t",
        header=None,
        names=["id", "transcript", "transcript_normalized"],
    )

    # Step 3: Delete the temporary file
    fixed_path.unlink()

    return df


class LibriTTS_R_DataPrep(DataPrep):
    """
    Prepares the LibriTTS-R dataset into structured tables for training and evaluation.

    This class supports:
    - Audio file indexing and duration extraction
    - Speaker, book, and chapter metadata integration
    - Transcript parsing from .tsv files
    - Segment and recording table generation

    Attributes:
        corpus_dir (PathLike): Root directory of the LibriTTS-R dataset.
        subset (str): One of ["train-clean-100", "train-clean-360", "dev-clean", "test-clean"].
        output_dir (PathLike): Path to save the prepared dataset.
        use_kaldi_ids (bool): If True, prepend speaker ID to segment IDs.
        target_sample_freq (Optional[int]): Optional resampling frequency in Hz.
        num_threads (int): Number of threads for duration extraction.
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
        """Initializes the LibriTTS-R preparation pipeline."""
        super().__init__(corpus_dir, output_dir, False, target_sample_freq, num_threads)

        self.subset = subset

    @staticmethod
    def dataset_name() -> str:
        """Returns the dataset name identifier."""
        return "libritts-r"

    @staticmethod
    def add_class_args(parser) -> None:
        """Adds command-line arguments for LibriTTS-R-specific options."""
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--subset",
            choices=[
                "train-clean-100",
                "train-clean-360",
                "dev-clean",
                "test-clean",
            ],
            help="""if we prepare the data for ["train-clean-100", "train-clean-360", "dev-clean", "test-clean"]""",
            required=True,
        )

    def _get_docs_dir(self) -> Path:
        """
        Returns the documentation directory.

        Returns:
            Path: Path to 'docs' if it exists, else corpus_dir.
        """
        docs_dir = self.corpus_dir / "docs"
        if docs_dir.is_dir():
            return docs_dir
        else:
            return self.corpus_dir

    def _get_spks_metadata(self) -> pd.DataFrame:
        """
        Loads speaker metadata from speakers.tsv.

        Returns:
            pd.DataFrame: Speaker info with ID and gender.
        """

        docs_dir = self._get_docs_dir()
        file_path = docs_dir / "speakers.tsv"
        df = pd.read_csv(
            file_path,
            sep="\t",
            skiprows=1,
            header=None,
            names=["id", "gender", "subset", "name"],
            dtype={"id": str},
        )
        df["id"] = df["id"].apply(lambda x: f"libri-{x}")
        df = df.loc[df["subset"] == self.subset]
        df["gender"] = df["gender"].apply(lambda x: x.lower())
        df.drop(columns=["subset", "name"], inplace=True)
        return df

    def _get_chapters_metadata(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads book and chapter metadata.

        Returns:
            tuple: (books DataFrame, chapters DataFrame)
        """
        docs_dir = self._get_docs_dir()
        file_path = docs_dir / "BOOKS.txt"
        df_books = pd.read_csv(
            file_path,
            sep="\s*\|\s*",
            header=None,
            names=["book", "book_title", "dummy"],
            engine="python",
            dtype={"book": str},
        )
        df_books.drop(columns=["dummy"], inplace=True)

        file_path = docs_dir / "CHAPTERS.txt"
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
        Loads transcript metadata from .trans.tsv files.

        Args:
            rec_dir (Path): Directory containing the audio and transcript files.

        Returns:
            pd.DataFrame: Transcription data with normalized and raw text.
        """
        logging.info("searching transcript files in %s", str(rec_dir))
        trans_files = list(rec_dir.glob("**/*.trans.tsv"))
        if not trans_files:
            # symlinks? try glob
            trans_files = [
                Path(f) for f in glob.iglob(f"{rec_dir}/**/.*trans.tsv", recursive=True)
            ]

        assert len(trans_files) > 0, "transcript files not found"
        dfs = []
        for trans_file in trans_files:

            try:
                df_i = pd.read_csv(
                    trans_file,
                    sep="\t",
                    header=None,
                    names=["id", "transcript", "transcript_normalized"],
                )
            except pd.errors.ParserError as e:
                df_i = _read_malformed_trans(trans_file)
            dfs.append(df_i)

        df_trans = pd.concat(dfs)
        df_trans["id"] = df_trans["id"].apply(lambda x: f"libritts-r-{x}")
        return df_trans

    def prepare(self) -> None:
        """
        Runs the complete LibriTTS-R data preparation pipeline:
        - Reads metadata (speakers, books, chapters, transcripts)
        - Builds RecordingSet and SegmentSet
        - Maps metadata to recordings and segments
        - Saves resulting HyperDataset
        """
        logging.info(
            "Peparing LibriTTS-R %s corpus_dir:%s -> data_dir:%s",
            self.subset,
            self.corpus_dir,
            self.output_dir,
        )

        df_spks = self._get_spks_metadata()
        df_books, df_chapters = self._get_chapters_metadata()

        rec_dir = self.corpus_dir / self.subset
        logging.info("searching audio files in %s", str(rec_dir))
        rec_files = list(rec_dir.glob("**/*.wav"))
        if not rec_files:
            # symlinks? try glob
            rec_files = [
                Path(f) for f in glob.iglob(f"{rec_dir}/**/*.wav", recursive=True)
            ]

        assert len(rec_files) > 0, "recording files not found"
        df_trans = self._get_transcripts(rec_dir)

        speakers = ["libri-" + f.parent.parent.name for f in rec_files]
        chapters = [f.parent.name for f in rec_files]
        rec_ids = ["libritts-r-" + f.with_suffix("").name for f in rec_files]
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
        df_segs["dataset"] = "libritts-r"
        df_segs["original_bandwidth"] = 12000
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
        # logging.info(
        #     "datasets containts %d segments, %d speakers", len(segments), len(speakers)
        # )
