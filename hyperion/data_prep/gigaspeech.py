import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..utils import ClassInfo, HypDataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike
from .data_prep import DataPrep


class GigaSpeechDataPrep(DataPrep):
    """
    Prepares the GigaSpeech dataset into structured tables for training or evaluation.

    This class reads the official GigaSpeech manifest (GigaSpeech.json),
    filters by subset, builds segment and recording manifests,
    and extracts speaker/language class information.

    Attributes:
        corpus_dir (PathLike): Base directory of the GigaSpeech corpus.
        subset (str): One of ["XL", "L", "M", "S", "XS", "dev", "test"] — which part of the dataset to prepare.
        output_dir (PathLike): Where to save the prepared dataset files.
        use_kaldi_ids (bool): If True, prepend speaker ID to each segment ID.
        target_sample_freq (Optional[int]): If specified, resample audio to this frequency.
        num_threads (int): Number of parallel threads for duration extraction.
    """

    TRAIN_SUBSETS = {"XL", "L", "M", "S", "XS"}
    EVAL_SUBSETS = {"DEV", "TEST"}

    def __init__(
        self,
        corpus_dir: PathLike,
        subset: str,
        output_dir: PathLike,
        use_kaldi_ids: bool = False,
        target_sample_freq: Optional[int] = None,
        num_threads: int = 10,
    ) -> None:
        """
        Initializes the GigaSpeech data preparation pipeline.

        Args:
            corpus_dir (PathLike): Directory containing the GigaSpeech corpus.
            subset (str): Dataset subset (e.g., 'M', 'XL').
            output_dir (PathLike): Destination directory for processed outputs.
            use_kaldi_ids (bool): Whether to prepend speaker ID to segment IDs.
            target_sample_freq (Optional[int]): Optional target resampling frequency.
            num_threads (int): Number of threads to use for audio duration extraction.
        """
        super().__init__(
            corpus_dir, output_dir, use_kaldi_ids, target_sample_freq, num_threads
        )
        self.subset = subset.upper()
        if self.subset not in (self.TRAIN_SUBSETS | self.EVAL_SUBSETS):
            raise ValueError(
                f"Unsupported GigaSpeech subset '{subset}'. "
                "Expected one of ['XL', 'L', 'M', 'S', 'XS', 'dev', 'test']."
            )

    @staticmethod
    def dataset_name() -> str:
        """
        Returns:
            str: Dataset name identifier.
        """
        return "gigaspeech"

    @staticmethod
    def add_class_args(parser) -> None:
        """
        Adds GigaSpeech-specific arguments to an argument parser.

        Args:
            parser: An ArgumentParser object from jsonargparse.
        """
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--subset",
            choices=["XL", "L", "M", "S", "XS", "dev", "test"],
            required=True,
            help="GigaSpeech subset to prepare (e.g., 'XL', 'M', 'S').",
        )

    def prepare(self) -> None:
        """
        Runs the complete data preparation pipeline for GigaSpeech:
        - Parses GigaSpeech.json
        - Filters audio and segments based on subset
        - Extracts recording durations
        - Builds SegmentSet, RecordingSet, and ClassInfo tables
        - Saves HypDataset to the specified output directory
        """
        logging.info(
            "Preparing GigaSpeech subset=%s corpus_dir=%s -> output_dir=%s",
            self.subset,
            self.corpus_dir,
            self.output_dir,
        )

        manifest_path = self.corpus_dir / "GigaSpeech.json"
        assert manifest_path.is_file(), f"Missing GigaSpeech manifest: {manifest_path}"

        with open(manifest_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        rec_items = []
        seg_items = []
        subset = f"{{{self.subset}}}"
        for doc in raw_data["audios"]:
            if subset not in doc["subsets"]:
                continue

            rec_id = doc["aid"]
            storage_path = self.corpus_dir / doc["path"]
            sample_freq = doc["sample_rate"]
            duration = doc["duration"]
            title = doc.get("title", pd.NA)
            source_type = doc["source"].lower()
            source_type = "afv" if source_type == "youtube" else source_type
            category = doc.get("category", "").lower()
            category = category if category not in ["", "n/a"] else pd.NA
            rec_items.append(
                {
                    "id": rec_id,
                    "storage_path": str(storage_path.resolve()),
                    "sample_freq": sample_freq,
                    "duration": duration,
                }
            )

            for seg in doc["segments"]:
                if subset not in seg["subsets"]:
                    continue

                seg_id = seg["sid"]
                speaker = seg.get("speaker", "")
                speaker = f"giga-{speaker}" if speaker not in ["", "N/A"] else pd.NA
                start_time = seg["begin_time"]
                duration = seg["end_time"] - start_time
                transcript = seg.get("text_tn", "")

                if self.use_kaldi_ids and speaker:
                    seg_id = f"{speaker}-{seg_id}"
                else:
                    seg_id = f"giga-{seg_id}"

                seg_items.append(
                    {
                        "id": seg_id,
                        "recording": rec_id,
                        "speaker": speaker,
                        "transcript": transcript,
                        "start": start_time,
                        "duration": duration,
                        "language": "eng",
                        "source_type": source_type,
                        "category": category,
                        "title": title,
                        "original_bandwidth": sample_freq / 2,
                    }
                )

        logging.info("Creating RecordingSet")
        df_recs = pd.DataFrame(rec_items)
        recs = RecordingSet(df_recs)
        recs.sort()
        if self.target_sample_freq:
            recs["target_sample_freq"] = self.target_sample_freq

        logging.info("Creating SegmentsSet")
        df_segs = pd.DataFrame(seg_items)
        segments = SegmentSet(df_segs)
        segments.sort()

        logging.info("Creating ClassInfo tables")

        def _clean_class_ids(values):
            series = pd.Series(values).dropna()
            if series.empty:
                return []
            return sorted(series.unique())

        speaker_ids = _clean_class_ids(df_segs["speaker"])
        category_ids = _clean_class_ids(df_segs["category"])

        classes = {}
        if speaker_ids:
            classes["speaker"] = ClassInfo(pd.DataFrame({"id": speaker_ids}))
        languages = ClassInfo(pd.DataFrame({"id": ["eng"]}))
        classes["language"] = languages
        source_types = ClassInfo(
            pd.DataFrame({"id": _clean_class_ids(df_segs["source_type"])})
        )
        classes["source_type"] = source_types
        if category_ids:
            classes["category"] = ClassInfo(pd.DataFrame({"id": category_ids}))

        dataset = HypDataset(segments=segments, recordings=recs, classes=classes)
        dataset.save(self.output_dir)
        dataset.describe()
