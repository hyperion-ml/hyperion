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
    filters by subset and split, builds segment and recording manifests,
    and extracts speaker/language class information.

    Attributes:
        corpus_dir (PathLike): Base directory of the GigaSpeech corpus.
        subset (str): One of ["XL", "L", "M", "S", "XS"] — which part of the dataset to prepare.
        split (str): One of ["train", "dev", "test"] — which split of the subset to prepare.
        output_dir (PathLike): Where to save the prepared dataset files.
        use_kaldi_ids (bool): If True, prepend speaker ID to each segment ID.
        target_sample_freq (Optional[int]): If specified, resample audio to this frequency.
        num_threads (int): Number of parallel threads for duration extraction.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        subset: str,
        split: str,
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
            split (str): Data split (e.g., 'train', 'dev', 'test').
            output_dir (PathLike): Destination directory for processed outputs.
            use_kaldi_ids (bool): Whether to prepend speaker ID to segment IDs.
            target_sample_freq (Optional[int]): Optional target resampling frequency.
            num_threads (int): Number of threads to use for audio duration extraction.
        """
        super().__init__(
            corpus_dir, output_dir, use_kaldi_ids, target_sample_freq, num_threads
        )
        self.subset = subset.upper()
        self.split = split

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
            choices=["XL", "L", "M", "S", "XS"],
            required=True,
            help="GigaSpeech subset to prepare (e.g., 'XL', 'M', 'S').",
        )
        parser.add_argument(
            "--split",
            choices=["train", "dev", "test"],
            required=True,
            help="Data split to prepare (e.g., 'train', 'dev', 'test').",
        )

    def prepare(self) -> None:
        """
        Runs the complete data preparation pipeline for GigaSpeech:
        - Parses GigaSpeech.json
        - Filters audio and segments based on subset and split
        - Extracts recording durations
        - Builds SegmentSet, RecordingSet, and ClassInfo tables
        - Saves HypDataset to the specified output directory
        """
        logging.info(
            "Preparing GigaSpeech subset=%s split=%s corpus_dir=%s -> output_dir=%s",
            self.subset,
            self.split,
            self.corpus_dir,
            self.output_dir,
        )

        manifest_path = self.corpus_dir / "GigaSpeech.json"
        assert manifest_path.is_file(), f"Missing GigaSpeech manifest: {manifest_path}"

        with open(manifest_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        items = []
        for doc in raw_data["audios"]:
            if doc.get("subset", "").upper() != self.subset:
                continue

            for seg in doc.get("segments", []):
                if seg.get("split") != self.split:
                    continue

                utt_id = seg["utt_id"]
                speaker = seg.get("speaker", "unknown")
                audio_path = Path(seg["audio_filepath"])
                duration = seg["duration"]
                text = seg.get("text", "")
                language = seg.get("language", "en")

                seg_id = f"giga-{utt_id}"
                if self.use_kaldi_ids:
                    seg_id = f"{speaker}-{seg_id}"

                items.append(
                    {
                        "id": seg_id,
                        "speaker": speaker,
                        "text": text,
                        "duration": duration,
                        "storage_path": str(audio_path.resolve()),
                        "language": language,
                    }
                )

        df = pd.DataFrame(items)
        df.sort_values(by="id", inplace=True)

        logging.info("Creating RecordingSet")
        recs = pd.DataFrame({"id": df["id"], "storage_path": df["storage_path"]})
        recs = RecordingSet(recs)
        recs.get_durations(self.num_threads)
        if self.target_sample_freq:
            recs["target_sample_freq"] = self.target_sample_freq

        df["duration"] = df["id"].map(recs.set_index("id")["duration"])

        logging.info("Creating SegmentsSet")
        segments = SegmentSet(df[["id", "speaker", "text", "duration", "language"]])
        segments.sort()

        logging.info("Creating ClassInfo tables")
        speakers = ClassInfo(pd.DataFrame({"id": np.unique(df["speaker"])}))
        languages = ClassInfo(pd.DataFrame({"id": np.unique(df["language"])}))

        dataset = HypDataset(
            segments=segments,
            recordings=recs,
            classes={"speaker": speakers, "language": languages},
        )
        dataset.save(self.output_dir)
        logging.info(
            "Dataset contains %d segments, %d speakers",
            len(segments),
            len(speakers),
        )
