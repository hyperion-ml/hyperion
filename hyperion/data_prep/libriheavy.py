"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from jsonargparse import ArgumentParser

from ..utils import (
    ClassInfo,
    HyperDataset,
    ParallelFileFinder,
    RecordingSet,
    SegmentSet,
)
from ..utils.misc import PathLike
from .data_prep import DataPrep


class LibriHeavyDataPrep(DataPrep):
    """
    Prepares the LibriHeavy dataset into structured tables for speaker recognition.

    This class handles:
    - Audio discovery in specified subset directory
    - Extraction of speaker and book metadata
    - Generation of RecordingSet and SegmentSet
    - Optional Kaldi-style IDs and resampling

    Attributes:
        corpus_dir (PathLike): Input directory containing LibriLight dataset structure.
        subset (str): Dataset subset (e.g., 'small', 'medium', 'large').
        output_dir (PathLike): Output directory to write prepared files.
        use_kaldi_ids (bool): Whether to prepend speaker ID to segment ID.
        target_sample_freq (Optional[int]): Optional resampling target frequency.
        num_threads (int): Number of parallel threads used for duration extraction.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        subset: str,
        output_dir: PathLike,
        librilight_corpus_dir: Optional[PathLike] = None,
        use_kaldi_ids: bool = False,
        target_sample_freq: Optional[int] = None,
        num_threads: int = 10,
    ):
        """
        Initializes the LibriHeavy data preprocessor.
        """
        super().__init__(corpus_dir, output_dir, False, target_sample_freq, num_threads)

        self.subset = subset
        if librilight_corpus_dir is None:
            librilight_corpus_dir = Path(corpus_dir) / "download" / "librilight"

        self.librilight_corpus_dir = Path(librilight_corpus_dir)

    @staticmethod
    def dataset_name() -> str:
        """Returns a string identifier for the dataset."""
        return "libriheavy"

    @staticmethod
    def add_class_args(parser: ArgumentParser) -> None:
        """
        Adds LibriHeavy-specific arguments to the CLI parser.

        Args:
            parser: The JSONArgParse ArgumentParser object.
        """
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--subset",
            choices=[
                "small",
                "medium",
                "large",
            ],
            help="""if we prepare the data for ["small", "medium", "large"]""",
            required=True,
        )
        parser.add_argument(
            "--librilight-corpus-dir",
            default=None,
            help="Path to the LibriLight corpus directory.",
        )

    @staticmethod
    def _load_transcript_entries(
        file_path: PathLike,
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """
        Read a Kaldi-style transcript file and return the utterance ids and text.

        Args:
            file_path (PathLike): Path to the transcript file.

        Returns:
            Tuple[Sequence[str], Sequence[str]]: Parallel sequences of IDs and transcript strings.
        """
        keys: list[str] = []
        values: list[str] = []
        path = Path(file_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split(maxsplit=1)
                if len(parts) != 2:
                    logging.warning(
                        "Skipping malformed transcript line %d in %s: %s",
                        line_num,
                        path,
                        stripped,
                    )
                    continue
                keys.append(parts[0])
                values.append(parts[1])
        return keys, values

    def _read_segments_and_transcripts(self) -> pd.DataFrame:
        """
        Load Kaldi segment definitions and attach both punctuated and unpunctuated transcripts.

        Returns:
            pd.DataFrame: Segment metadata with normalized IDs, recording references,
            time fields, and both punctuated and unpunctuated transcripts.
        """
        kaldi_dir = self.corpus_dir / "cases_and_punc" / "kaldi" / self.subset
        kaldi_dir_no_punc = self.corpus_dir / "upper_no_punc" / "kaldi" / self.subset

        transcript_file = kaldi_dir / "text"
        seg_id, transcript = self._load_transcript_entries(transcript_file)
        transcript_file = kaldi_dir_no_punc / "text"
        seg_id_no_punc, transcript_no_punc = self._load_transcript_entries(
            transcript_file
        )

        segments = pd.read_csv(
            kaldi_dir / "segments",
            sep=" ",
            header=None,
            names=["id", "recording", "start", "end"],
        )
        segments["duration"] = segments["end"] - segments["start"]
        segments.drop(columns=["end"], inplace=True)

        segments["transcript"] = (
            pd.Series(transcript, index=seg_id).reindex(segments["id"]).values
        )
        segments["transcript_no_punc"] = (
            pd.Series(transcript_no_punc, index=seg_id_no_punc)
            .reindex(segments["id"])
            .values
        )
        malformed_mask = (
            segments["transcript"].isna() | segments["transcript_no_punc"].isna()
        )
        if malformed_mask.any():
            dropped = segments.loc[malformed_mask, ["id", "recording"]].copy()
            logging.warning(
                "Dropping %d segments with missing transcripts:\n%s",
                malformed_mask.sum(),
                dropped,
            )
            segments = segments.loc[~malformed_mask].reset_index(drop=True)
        segments["recording"] = segments["recording"].apply(lambda x: Path(x))
        segments["recording"] = segments["recording"].apply(
            lambda x: "libriheavy-"
            + x.parent.parent.name
            + "-"
            + x.parent.name
            + "-"
            + x.name
        )
        segments["id"] = segments["id"].apply(lambda x: Path(x))
        segments["id"] = segments["id"].apply(
            lambda x: "libriheavy-"
            + x.parent.parent.name
            + "-"
            + x.parent.name
            + "-"
            + x.name
        )

        return segments

    def prepare(self) -> None:
        """
        Executes the full LibriHeavy data preparation pipeline:
        - Discovers audio files in the given subset
        - Builds RecordingSet and SegmentSet
        - Extracts metadata (speaker, book, language)
        - Saves dataset in the output directory
        """

        logging.info(
            "Preparing LibriHeavy %s corpus_dir:%s -> data_dir:%s",
            self.subset,
            self.corpus_dir,
            self.output_dir,
        )

        rec_dir = self.librilight_corpus_dir / self.subset
        logging.info("searching audio files in %s", str(rec_dir))
        file_finder = ParallelFileFinder(
            root=rec_dir,
            pattern=r".*\.flac$",
            num_threads=self.num_threads,
        )
        rec_files = file_finder()
        assert len(rec_files) > 0, "recording files not found"

        speakers = ["libri-" + f.parent.parent.name for f in rec_files]
        books = [f.parent.name for f in rec_files]
        rec_ids = [
            "libriheavy-"
            + f.parent.parent.name
            + "-"
            + f.parent.name
            + "-"
            + f.with_suffix("").name
            for f in rec_files
        ]

        file_paths = [str(r) for r in rec_files]
        logging.info("making RecordingSet")
        recs = pd.DataFrame({"id": rec_ids, "storage_path": file_paths})
        dup_mask = recs["id"].duplicated(keep=False)
        if dup_mask.any():
            logging.warning(
                "Found %d duplicated recording ids:\n%s",
                dup_mask.sum(),
                recs.loc[dup_mask],
            )
        recs = RecordingSet(recs)
        recs.sort()

        logging.info("getting recording durations")
        recs.get_durations(self.num_threads)
        if self.target_sample_freq:
            recs["target_sample_freq"] = self.target_sample_freq

        logging.info("making SegmentsSet")
        df_meta = pd.DataFrame(
            {"recording": rec_ids, "speaker": speakers, "book": books}
        )
        df_segs = self._read_segments_and_transcripts()
        df_segs = df_segs.merge(df_meta, how="left", on="recording")
        if self.use_kaldi_ids:
            df_segs["id"] = [
                f"{speaker}-{seg_id}"
                for seg_id, speaker in zip(df_segs["id"], df_segs["speaker"])
            ]
        df_segs["language"] = "eng"
        df_segs["source_type"] = "audiobook"
        df_segs["corpusid"] = "librivox"
        df_segs["dataset"] = "libriheavy"
        df_segs["original_bandwidth"] = 8000
        segments = SegmentSet(df_segs)
        segments.sort()

        logging.info("making speaker info file")
        df_spks = pd.DataFrame({"id": np.unique(df_segs["speaker"])})
        speakers = ClassInfo(df_spks)

        logging.info("making book info file")
        df_books = pd.DataFrame({"id": np.unique(df_segs["book"])})
        books = ClassInfo(df_books)

        logging.info("making language info file")
        languages = ClassInfo(pd.DataFrame({"id": ["eng"]}))
        # logging.info("making gender info file")
        # genders = ClassInfo(pd.DataFrame({"id": ["m", "f"]}))

        classes = {
            "speaker": speakers,
            "book": books,
            "language": languages,
            # "gender": genders,
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
