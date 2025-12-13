"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pycountry

from ..utils import ClassInfo, HyperDataset, ParallelFileFinder, RecordingSet, SegmentSet
from ..utils.misc import PathLike
from .data_prep import DataPrep


class RAVDESSPrep(DataPrep):
    """
    Prepares the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset
    into structured metadata tables for training and evaluation.

    This includes:
    - Discovery and parsing of audio recordings.
    - Extraction of speaker, emotion, intensity, gender, and transcript information from filenames.
    - Construction of RecordingSet, SegmentSet, and ClassInfo tables.
    - Optional resampling of audio and Kaldi-style ID formatting.

    Attributes:
        corpus_dir (PathLike): Path to the root of the RAVDESS dataset.
        output_dir (PathLike): Directory where the processed output will be stored.
        use_kaldi_ids (bool): If True, segment IDs will be prefixed with speaker IDs.
        target_sample_freq (Optional[int]): If set, resamples audio to this frequency.
        num_threads (int): Number of threads used for parallel audio duration extraction.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        output_dir: PathLike,
        use_kaldi_ids: bool = False,
        target_sample_freq: Optional[int] = None,
        num_threads: int = 10,
    ) -> None:
        """
        Initializes the RAVDESS data preparation class.

        Args:
            corpus_dir (PathLike): Base directory containing the RAVDESS audio files.
            output_dir (PathLike): Directory where processed output will be saved.
            use_kaldi_ids (bool): Whether to prepend speaker ID to segment IDs.
            target_sample_freq (Optional[int]): Resample audio to this frequency if specified.
            num_threads (int): Number of threads for parallel audio duration extraction.
        """
        super().__init__(
            corpus_dir, output_dir, use_kaldi_ids, target_sample_freq, num_threads
        )

    @staticmethod
    def dataset_name() -> str:
        """
        Returns:
            str: Identifier name for the dataset ("ravdess").
        """
        return "ravdess"

    @staticmethod
    def add_class_args(parser) -> None:
        """
        Adds RAVDESS-specific arguments to a CLI parser.

        Args:
            parser (ArgumentParser): The argument parser to which RAVDESS arguments will be added.
        """
        DataPrep.add_class_args(parser)

    def prepare(self) -> None:
        """
        Executes the full RAVDESS data preparation pipeline.

        - Scans the directory for `.wav` files.
        - Extracts metadata (speaker, emotion, gender, etc.) from filenames.
        - Computes durations for all audio files.
        - Builds metadata tables: RecordingSet, SegmentSet, and ClassInfo.
        - Saves dataset files to output directory.
        """
        logging.info(
            "Preparing RAVDESS dataset corpus_dir=%s -> data_dir=%s",
            self.corpus_dir,
            self.output_dir,
        )

        rec_dir = self.corpus_dir

        logging.info("searching audio files in %s", str(rec_dir))
        file_finder = ParallelFileFinder(
            root=rec_dir,
            pattern=r".*\.wav$",
            num_threads=self.num_threads,
        )
        rec_files = file_finder()
        assert len(rec_files) > 0, "recording files not found"

        rec_ids = ["ravdess-" + f.with_suffix("").name for f in rec_files]
        file_paths = [str(r) for r in rec_files]
        logging.info("making RecordingSet")
        recs = pd.DataFrame({"id": rec_ids, "storage_path": file_paths})
        recs = RecordingSet(recs)
        recs.sort()
        recs.get_durations(self.num_threads)
        if self.target_sample_freq:
            recs["target_sample_freq"] = self.target_sample_freq

        logging.info("Creating SegmentSet")
        speakers = []
        emotions = []
        emotion_intensities = []
        transcripts = []
        genders = []
        # Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
        emotions_dict = {
            "01": "neutral",
            "02": "calm",
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "06": "fearful",
            "07": "disgust",
            "08": "surprised",
        }
        # Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
        emotion_intensities_dict = {
            "01": "normal",
            "02": "strong",
        }
        # Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
        statements_dict = {
            "01": "Kids are talking by the door",
            "02": "Dogs are sitting by the door",
        }
        for rec_id in rec_ids:
            parts = rec_id.split("-")
            if len(parts) != 8:
                logging.warning("Invalid recording ID format: %s", rec_id)
                continue

            speakers.append(f"ravdess-{parts[7]}")
            emotions.append(emotions_dict[parts[3]])
            emotion_intensities.append(emotion_intensities_dict[parts[4]])
            transcripts.append(statements_dict[parts[5]])
            genders.append("f" if int(parts[7]) % 2 == 0 else "m")

        df_segs = pd.DataFrame(
            {
                "id": rec_ids,
                "speaker": speakers,
                "gender": genders,
                "transcript": transcripts,
                "emotion": emotions,
                "emotion_intensity": emotion_intensities,
            }
        )
        df_segs["duration"] = recs.loc[df_segs["id"], "duration"].values
        df_segs["language"] = "eng"
        df_segs["corpusid"] = "ravdess"
        df_segs["dataset"] = "ravdess"
        df_segs["original_bandwidth"] = 24000
        segments = SegmentSet(df_segs)
        segments.sort()

        if self.use_kaldi_ids:
            segments["id"] = segments["speaker"] + "-" + segments["id"]
            recs["id"] = recs["speaker"] + "-" + recs["id"]

        logging.info("Creating ClassInfo tables")
        speakers = ClassInfo(pd.DataFrame({"id": np.sort(df_segs["speaker"].unique())}))
        languages = ClassInfo(pd.DataFrame({"id": ["eng"]}))
        emotions = ClassInfo(pd.DataFrame({"id": np.sort(df_segs["emotion"].unique())}))
        emotion_intensities = ClassInfo(
            pd.DataFrame({"id": np.sort(df_segs["emotion_intensity"].unique())})
        )
        genders = ClassInfo(pd.DataFrame({"id": ["m", "f"]}))

        logging.info("Saving dataset")
        dataset = HyperDataset(
            segments=segments,
            recordings=recs,
            classes={
                "speaker": speakers,
                "language": languages,
                "gender": genders,
                "emotion": emotions,
                "emotion_intensity": emotion_intensities,
            },
        )
        dataset.save(self.output_dir)
        dataset.describe()
