"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ..utils import ClassInfo, HypDataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike
from .data_prep import DataPrep


class CommonVoiceDataPrep(DataPrep):
    """
    Prepares Mozilla Common Voice datasets into structured tables.

    Supports single or multi-language preparation and allows selecting subsets such as
    'validated', 'train', 'test', etc. Builds `RecordingSet`, `SegmentSet`, and
    class info tables for speaker and language.

    Attributes:
        corpus_dir (PathLike): Root Common Voice directory containing language subfolders.
        language (str): Language code (e.g., 'en') or 'all' to process every language folder.
        subset (str): Which subset file to process (e.g., 'validated', 'train', 'test').
        output_dir (PathLike): Directory to save prepared outputs.
        use_kaldi_ids (bool): Whether to prepend speaker ID to each segment ID.
        target_sample_freq (Optional[int]): Optional target sample rate (Hz).
        num_threads (int): Number of threads for parallel audio processing.
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
    ):
        """
        Initializes the CommonVoice preparation logic.

        Args:
            corpus_dir (PathLike): Path to the dataset root.
            language (str): Language code or 'all'.
            subset (str): Subset file to process ('validated', 'train', 'test', etc.).
            output_dir (PathLike): Where to save processed data.
            use_kaldi_ids (bool): Whether to format IDs with speaker prefix.
            target_sample_freq (Optional[int]): If set, resample audio to this frequency.
            num_threads (int): Number of parallel threads for duration extraction.
        """
        super().__init__(
            corpus_dir, output_dir, use_kaldi_ids, target_sample_freq, num_threads
        )
        self.language = language.lower()
        self.subset = subset.lower()

    @staticmethod
    def dataset_name() -> str:
        """Returns the dataset name identifier."""
        return "commonvoice"

    @staticmethod
    def add_class_args(parser) -> None:
        """
        Adds CLI arguments specific to Common Voice.

        Args:
            parser: Argument parser object.
        """
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--language",
            required=True,
            help="Language code (e.g., 'en') or 'all' to prepare all available languages.",
        )
        parser.add_argument(
            "--subset",
            default="validated",
            help="Which Common Voice TSV subset to prepare (e.g., 'validated', 'test', 'train').",
        )

    def prepare(self) -> None:
        """
        Executes the preparation for one or all languages.

        Loads metadata, aligns it with audio, extracts durations, and saves a HypDataset
        for each selected language.
        """
        if self.language == "all":
            langs = [
                d.name
                for d in self.corpus_dir.iterdir()
                if (d / f"{self.subset}.tsv").is_file()
            ]
        else:
            langs = [self.language]

        for lang in langs:
            logging.info(f"Preparing Common Voice {lang} subset={self.subset}")
            self._prepare_language(lang)

    def _prepare_language(self, lang: str) -> None:
        """
        Prepares a single language subset.

        Args:
            lang (str): Language code to process.
        """
        lang_dir = self.corpus_dir / lang
        tsv_path = lang_dir / f"{self.subset}.tsv"
        clips_dir = lang_dir / "clips"

        assert tsv_path.exists(), f"Missing {self.subset}.tsv in {lang_dir}"
        assert clips_dir.is_dir(), f"Missing 'clips/' directory in {lang_dir}"

        df = pd.read_csv(tsv_path, sep="\t", dtype={"client_id": str})
        df["speaker"] = df["client_id"].apply(lambda x: f"cv-{x}")
        df["id"] = df["path"].apply(lambda x: f"cv-{Path(x).with_suffix('').name}")
        df["storage_path"] = df["path"].apply(lambda x: str((clips_dir / x).resolve()))
        df["language"] = lang

        if self.use_kaldi_ids:
            df["id"] = df.apply(lambda row: f"{row['speaker']}-{row['id']}", axis=1)

        logging.info(f"Creating RecordingSet for {lang}")
        recs = pd.DataFrame({"id": df["id"], "storage_path": df["storage_path"]})
        recs = RecordingSet(recs)
        recs.get_durations(self.num_threads)

        if self.target_sample_freq:
            recs["target_sample_freq"] = self.target_sample_freq

        df["duration"] = df["id"].map(recs.set_index("id")["duration"])

        logging.info(f"Creating SegmentsSet for {lang}")
        segments = SegmentSet(df[["id", "speaker", "sentence", "duration", "language"]])
        segments.sort()

        logging.info(f"Creating ClassInfo tables for {lang}")
        speakers = ClassInfo(pd.DataFrame({"id": np.unique(df["speaker"])}))
        languages = ClassInfo(pd.DataFrame({"id": [lang]}))

        output_path = (
            self.output_dir / lang if self.language == "all" else self.output_dir
        )
        output_path.mkdir(parents=True, exist_ok=True)

        logging.info(f"Saving dataset for {lang} to {output_path}")
        dataset = HypDataset(
            segments=segments,
            recordings=recs,
            classes={"speaker": speakers, "language": languages},
        )
        dataset.save(output_path)
        logging.info(
            "Language %s: %d segments, %d speakers",
            lang,
            len(segments),
            len(speakers),
        )
