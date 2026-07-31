"""
Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from jsonargparse import ActionYesNo
from tqdm import tqdm

from ..utils import ClassInfo, HyperDataset, RecordingSet, SegmentSet
from ..utils.misc import PathLike
from .data_prep import DataPrep

multigender_spks = [
    "111774",
    "111781",
    "112778",
    "112783",
    "112879",
    "113153",
    "113213",
    "113603",
    "128673",
    "128770",
]


class SRECTSSupersetDataPrep(DataPrep):
    """
    Prepares the SRE-CTS Superset (LDC2021E08) dataset into structured tables.

    This includes:
    - Parsing the segment metadata from a TSV file.
    - Handling speakers with conflicting gender annotations.
    - Generating RecordingSet and SegmentSet entries.
    - Constructing ClassInfo tables for speaker, gender, language, and source type.

    Attributes:
        corpus_dir (PathLike): Directory containing audio and metadata.
        output_dir (PathLike): Output directory for the structured dataset.
        use_kaldi_ids (bool): Whether to prepend speaker ID to segment ID.
        target_sample_freq (Optional[int]): Resampling target frequency, if specified.
        num_threads (int): Number of threads to use for duration extraction.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        output_dir: PathLike,
        use_kaldi_ids: bool = False,
        target_sample_freq: Optional[int] = None,
        num_threads: int = 10,
    ):
        """
        Initializes the SRE-CTS Superset preparation pipeline.

        Args:
            corpus_dir (PathLike): Path to the SRE dataset root directory.
            output_dir (PathLike): Path to write the processed dataset.
            use_kaldi_ids (bool): Whether to prepend speaker ID to segment ID.
            target_sample_freq (Optional[int]): Resampling frequency.
            num_threads (int): Number of parallel threads for duration processing.
        """
        use_kaldi_ids = True
        super().__init__(
            corpus_dir, output_dir, use_kaldi_ids, target_sample_freq, num_threads
        )

    @staticmethod
    def dataset_name() -> str:
        """
        Returns:
            str: Dataset name identifier.
        """
        return "sre_cts_superset"

    @staticmethod
    def add_class_args(parser) -> None:
        """
        Adds SRE-CTS specific CLI arguments.

        Args:
            parser: Argument parser object.
        """
        DataPrep.add_class_args(parser)

    def _fix_multigender_spks(self, df):
        """
        Resolves speakers with conflicting gender labels by keeping the majority gender.

        Args:
            df (pd.DataFrame): Segment metadata containing 'speaker' and 'gender'.

        Returns:
            pd.DataFrame: Filtered metadata with consistent gender labels per speaker.
        """
        logging.info("Fixing multigender speakers keeping the majority gender")
        n0 = len(df)
        for spk in multigender_spks:
            male_idx = (df["speaker"] == spk) & (df["gender"] == "m")
            female_idx = (df["speaker"] == spk) & (df["gender"] == "f")
            num_male = np.sum(male_idx)
            num_female = np.sum(female_idx)
            if num_male > num_female:
                df = df[~female_idx]
            else:
                df = df[~male_idx]

        logging.info("Fixed multigender speakers, %d/%d segments remained", len(df), n0)
        return df

    def prepare(self) -> None:
        """
        Runs the full SRE-CTS Superset preparation pipeline:
        - Loads and cleans metadata
        - Extracts durations
        - Creates RecordingSet and SegmentSet
        - Builds speaker/language/gender/source ClassInfo tables
        - Saves HyperDataset to output_dir
        """
        logging.info(
            "Peparing SRE-CTS Superset corpus_dir:%s -> data_dir:%s",
            self.corpus_dir,
            self.output_dir,
        )
        logging.info("loading metadata")
        wav_dir = Path(self.corpus_dir) / "data"
        table_file = Path(self.corpus_dir) / "docs/cts_superset_segment_key.tsv"
        df_segs = pd.read_csv(table_file, sep="\t")

        logging.info("making SegmentSet")

        df_segs.drop(["speakerid"], axis=1, inplace=True)
        df_segs.rename(
            columns={"subjectid": "speaker"},
            inplace=True,
        )
        df_segs["speaker"] = df_segs["speaker"].astype("str")
        df_segs["gender"] = (
            df_segs["gender"].str.replace("female", "f").str.replace("male", "m")
        )
        df_segs["source_type"] = "cts"
        df_segs["dataset"] = self.dataset_name()
        if self.use_kaldi_ids:
            df_segs.drop(["segmentid"], axis=1, inplace=True)
            df_segs["id"] = df_segs["filename"].str.replace("/", "-")
            # put segment_id as first columnt
            cols = df_segs.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            df_segs = df_segs[cols]
        else:
            df_segs.rename(columns={"segmentid": "id"}, inplace=True)

        df_segs = self._fix_multigender_spks(df_segs)
        logging.info("sorting segments by segment_id")
        df_segs.sort_values(by="id", inplace=True)

        logging.info("making RecordingSet")
        df_recs = df_segs[["id", "filename"]].copy()
        df_segs.drop(["filename"], axis=1, inplace=True)
        df_recs["storage_path"] = df_recs["filename"].apply(
            lambda x: f"sph2pipe -f wav -p -c 1 {wav_dir / x} |"
        )
        df_recs = df_recs[["id", "storage_path"]]
        if self.target_sample_freq is not None:
            df_recs["target_sample_freq"] = self.target_sample_freq
        df_recs["sample_freq"] = 8000

        recs = RecordingSet(df_recs)
        recs.get_durations(self.num_threads)

        df_segs["duration"] = recs.loc[df_segs["id"], "duration"].values
        df_segs["original_bandwidth"] = 4000
        segments = SegmentSet(df_segs.copy())

        logging.info("making ClassInfos")
        df_spks = df_segs[["speaker", "gender"]].drop_duplicates()
        df_spks.rename(columns={"speaker": "id"}, inplace=True)
        df_spks.sort_values(by="id", inplace=True)
        speakers = ClassInfo(df_spks)

        df_langs = df_segs[["language"]].drop_duplicates()
        df_langs.rename(columns={"language": "id"}, inplace=True)
        df_langs.sort_values(by="id", inplace=True)
        languages = ClassInfo(df_langs)

        df_source = df_segs[["source_type"]].drop_duplicates()
        df_source.rename(columns={"source_type": "id"}, inplace=True)
        df_source.sort_values(by="id", inplace=True)
        sources = ClassInfo(df_source)

        genders = ClassInfo(pd.DataFrame({"id": ["m", "f"]}))

        logging.info("making dataset")
        dataset = HyperDataset(
            segments,
            {
                "speaker": speakers,
                "language": languages,
                "source_type": sources,
                "gender": genders,
            },
            recs,
        )
        logging.info("saving dataset at %s", self.output_dir)
        dataset.save(self.output_dir)
        logging.info(
            "datasets contains %d segments, %d speakers, %d languages, %f hours",
            len(segments),
            len(speakers),
            len(languages),
            segments["duration"].sum() / 3600.0,
        )
