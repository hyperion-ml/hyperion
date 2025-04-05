"""
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import glob
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from jsonargparse import ActionYesNo
from tqdm import tqdm

from ..utils import (ClassInfo, HypDataset, RecordingSet, SegmentSet, TrialKey,
                     TrialNdx)
from ..utils.misc import PathLike, urlretrieve_progress
from .data_prep import DataPrep


class LibriTTS_R(DataPrep):
    """Class for preparing LibriTTS-R database into tables,
    Attibutes:
      corpus_dir: input data directory
      subset: train/dev/eval
      output_dir: output data directory
      use_kaldi_ids: puts speaker-id in front of segment id like kaldi
      target_sample_freq: target sampling frequency to convert the audios to.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        subset: str,
        output_dir: PathLike,
        use_kaldi_ids: bool,
        target_sample_freq: int,
        num_threads: int = 10,
    ):
        super().__init__(corpus_dir, output_dir, False, target_sample_freq, num_threads)

        self.subset = subset

    @staticmethod
    def dataset_name():
        return "libritts-r"

    @staticmethod
    def add_class_args(parser):
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

    def _get_docs_dir(self):
        docs_dir = self.corpus_dir / "docs"
        if docs_dir.is_dir():
            return docs_dir
        else:
            return self.corpus_dir

    def _get_spks_metadata(self):

        docs_dir = self._get_docs_dir()
        file_path = docs_dir / "speakers.tsv"
        df = pd.read_csv(file_path, sep="\t", skiprows=1, header=0, names=["id", "gender", "subset", "name"], dtype={"id": str})
        df["id"] = df["id"].apply(lambda x: f"libri-{x}")
        df = df.loc[df["subset"] == self.subset]
        df["gender"] = df["gender"].apply(lambda x: x.lower())
        df.drop(columns=["subset", "name"], inplace=True)
        print(df, flush=True)
        return df

    def _get_chapters_metadata(self):
        docs_dir = self._get_docs_dir()
        file_path = docs_dir / "BOOKS.txt"
        df_books = pd.read_csv(
            file_path, sep="\s*\|\s*", header=0, names=["book", "book_title", "dummy"], engine="python", dtype={"book": str})
        df_books.drop(columns=["dummy"], inplace=True)

        file_path = docs_dir / "CHAPTERS.txt"
        df_chapters = pd.read_csv(
            file_path,
            sep="\s*\|\s*",
            header=0,
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
        print(df_books, "\n", df_chapters, flush=True)
        df_chapters = df_chapters.merge(df_books, how="left", on="book")
        df_chapters["speaker"] = df_chapters["speaker"].apply(lambda x: f"libri-{x}")
        df_chapters.drop(columns=["proj", "mins", "proj_title"], inplace=True)
        
        return df_books, df_chapters

    def prepare(self):
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
        df_segs["duration"] = recs.loc[df_segs["id"], "duration"].values
        df_segs = df_segs.merge(df_chapters, how="left", on="chapter")
        df_segs.rename(columns={"speaker_x": "speaker"}, inplace=True)
        df_segs.drop(columns=["speaker_y"], inplace=True)

        segments = SegmentSet(df_segs)
        segments.sort()

        logging.info("making speaker info file")
        print(df_segs, flush=True)
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

        classes = {"speaker": speakers, "book": books, "chapter": chapters}

        logging.info("making dataset")
        dataset = HypDataset(
            segments,
            classes=classes,
            recordings=recs,
        )
        logging.info("saving dataset at %s", self.output_dir)
        dataset.save(self.output_dir)
        logging.info(
            "datasets containts %d segments, %d speakers", len(segments), len(speakers)
        )
