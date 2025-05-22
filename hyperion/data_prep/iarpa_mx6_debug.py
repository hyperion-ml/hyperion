"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from jsonargparse import ActionYesNo

from ..utils import (
    ClassInfo,
    EnrollmentMap,
    HypDataset,
    RecordingSet,
    SegmentSet,
    TrialKey,
)
from ..utils.misc import PathLike
from .data_prep import DataPrep


class IARPAMixer6DebugDataPrep(DataPrep):
    """
    Prepares the IARPA Mixer 6 Debug Dataset for speaker recognition experiments.

    Supports processing of 'initialization', 'test', and 'test_intvs' partitions,
    including transcript loading, enrollment mapping, and trial generation.

    Attributes:
        corpus_dir (PathLike): Root directory of the dataset.
        partition (str): One of ['init-speaker', 'init-understand'].
        output_dir (PathLike): Directory where the prepared dataset is saved.
        use_kaldi_ids (bool): If True, use Kaldi-style IDs (<speaker>-<segment>).
        target_sample_freq (Optional[int]): Optional audio resampling frequency.
        num_threads (int): Number of threads for parallel duration computation.
    """

    def __init__(
        self,
        corpus_dir: PathLike,
        partition: str,
        output_dir: PathLike,
        use_kaldi_ids: bool = False,
        target_sample_freq: Optional[int] = None,
        num_threads: int = 10,
    ):
        super().__init__(
            corpus_dir, output_dir, use_kaldi_ids, target_sample_freq, num_threads
        )

        self.partition = partition

    @staticmethod
    def dataset_name() -> str:
        """Returns the dataset name identifier."""
        return "iarpa_mx6_debug"

    @staticmethod
    def add_class_args(parser) -> None:
        """Adds command-line interface arguments for the IARPA Mixer 6 Debug preparation."""
        DataPrep.add_class_args(parser)
        parser.add_argument(
            "--partition",
            choices=["initialization", "test", "test_intvs"],
            help="""trial side in ["initialization", "test", "test_intvs"]""",
            required=True,
        )

    def _read_docs(self) -> pd.DataFrame:
        """
        Reads and parses the debug metadata CSV, and optionally the transcript JSON.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (df_partition, df_initialization)
        """
        logging.info("loading docs")

        seg_file = self.corpus_dir / "datapack" / "phase1-1_debug.csv"
        df_segs = pd.read_csv(seg_file, sep=",", dtype={"speaker_id": str})
        # table columns:
        # category,segment_id,speaker_id,call_id,corpus,duration,channel,ntv_lg,sex,ethnic,yob,cntry_born,state_born,city_born,cntry_rsd,state_rsd,city_rsd,segno,rec_filename
        df_segs.rename(
            columns={
                "category": "partition",
                "segment_id": "id",
                "speaker_id": "speaker",
                "corpus": "corpusid",
                "ntv_lg": "native_language",
                "sex": "gender",
                "duration": "speech_duration",
            },
            inplace=True,
        )
        df_segs["native_language"] = df_segs["native_language"].str.lower()
        df_segs["channel"] = df_segs["channel"].str.lower()
        df_segs["gender"] = df_segs["gender"].str.lower()
        df_segs["language"] = "eng"
        intv_idx = df_segs["partition"] == "test_intvs"
        df_segs.loc[intv_idx, "source_type"] = "intv"
        df_segs.loc[~intv_idx, "source_type"] = "cts"

        df_enr = df_segs[df_segs["partition"] == "initialization"]
        df_segs = df_segs[df_segs["partition"] == self.partition]
        if self.partition == "test":
            transcript_file = self.corpus_dir / "iarpa-debug-transcripts.json"
            with open(transcript_file, "r") as f:
                transcripts = json.load(f)

            transcripts = {re.sub(r"^.*/", "", k): v for k, v in transcripts.items()}
            for i, row in df_segs.iterrows():
                seg_id = row["id"]
                channel = row["channel"]
                seg_ch_id = f"{seg_id[:-4]}-{channel.upper()}"
                if seg_ch_id in transcripts:
                    df_segs.at[i, "transcript"] = transcripts[seg_ch_id]
                else:
                    logging.warning("Missing transcript for segment: %s", seg_ch_id)

        return df_segs, df_enr

    def make_recording_set(self, df_segs: pd.DataFrame) -> RecordingSet:
        """
        Builds the RecordingSet table from segment metadata.

        Args:
            df_segs (pd.DataFrame): Metadata for segments.

        Returns:
            RecordingSet: Table of audio file paths and durations.
        """
        logging.info("making RecordingSet")
        wav_dir = self.corpus_dir / "datapack" / self.partition

        def channel_to_num(c):
            if c == "a":
                return 1
            elif c == "b":
                return 2
            else:
                raise ValueError(f"Invalid channel: {c}")

        if self.partition == "test_intvs":
            fs = 16000
            paths = [wav_dir / s for s in df_segs["id"]]
        else:
            fs = 8000
            paths = [
                f"sox {wav_dir / s} -c 1 -t wav - remix {channel_to_num(c)} |"
                for s, c in zip(df_segs["id"], df_segs["channel"])
            ]
        df_recs = pd.DataFrame({"id": df_segs["id"], "storage_path": paths})
        df_recs["sample_freq"] = fs
        if self.target_sample_freq is not None:
            df_recs["target_sample_freq"] = self.target_sample_freq

        recordings = RecordingSet(df_recs)
        recordings.get_durations(self.num_threads)
        return recordings

    def make_class_infos(self, df_segs: pd.DataFrame) -> dict[str, ClassInfo]:
        """
        Builds ClassInfo tables for speakers, gender, language, and source type.

        Args:
            df_segs (pd.DataFrame): Segment metadata.

        Returns:
            dict[str, ClassInfo]: Class tables keyed by name.
        """
        logging.info("making ClassInfos")
        df_segs = df_segs.reset_index(drop=True)
        df_spks = df_segs[["speaker"]].drop_duplicates().dropna()
        df_spks.rename(columns={"speaker": "id"}, inplace=True)
        df_spks.sort_values(by="id", inplace=True)
        speakers = ClassInfo(df_spks)

        languages = ClassInfo(pd.DataFrame({"id": ["eng"]}))
        sources = ClassInfo(pd.DataFrame({"id": ["cts", "intv"]}))
        return {
            "speaker": speakers,
            "language": languages,
            "source_type": sources,
        }

    def make_enrollments(self, df_segs: pd.DataFrame) -> dict[str, EnrollmentMap]:
        """
        Creates an EnrollmentMap from the initialization metadata.

        Args:
            df_segs: DataFrame containing initialization segment metadata.

        Returns:
            Dictionary with key 'enrollment' mapped to the EnrollmentMap.
        """
        logging.info("making Enrollment")
        df_enr = df_segs[["speaker", "id", "gender"]].copy()
        df_enr.rename(columns={"speaker": "modelid", "id": "segmentid"}, inplace=True)
        return {"enrollment": EnrollmentMap(df_enr)}

    def make_trials(
        self, df_segs: SegmentSet, df_enr: EnrollmentMap
    ) -> dict[str, Path]:
        """
        Constructs a TrialKey by comparing speaker IDs across enrollments and test segments.

        Args:
            df_segs: SegmentSet of test data.
            df_enr: EnrollmentMap containing enrolled speakers and their genders.

        Returns:
            Dictionary with key 'trials' mapped to the saved trial key file path.
        """

        logging.info("making Trials")
        key = TrialKey(np.unique(df_enr["id"]), np.unique(df_segs["id"]))
        for i, model_id in enumerate(key.model_set):
            for j, seg_id in enumerate(key.seg_set):
                test_spk = df_segs.at[seg_id, "speaker"]
                if model_id == test_spk:
                    key.tar[i, j] = True
                else:
                    enr_g = df_enr.at[model_id, "gender"]
                    test_g = df_segs.at[seg_id, "gender"]
                    if enr_g == test_g:
                        key.non[i, j] = True

        output_file = self.output_dir / "trials.csv"
        key.save(output_file)
        trials = {"trials": output_file}
        return trials

    def prepare(self) -> None:
        """
        Runs the full preparation pipeline for IARPA Mixer 6 Debug:
        - Reads and cleans metadata
        - Builds recording and segment tables
        - Creates class labels and optional trial/enrollment files
        - Outputs a complete HypDataset
        """
        logging.info(
            "Peparing IARPA Mixer 6 Debug %s corpus_dir: %s -> data_dir: %s",
            self.partition,
            self.corpus_dir,
            self.output_dir,
        )
        df_segs, df_enr = self._read_docs()
        recs = self.make_recording_set(df_segs)
        df_segs["duration"] = recs.loc[df_segs["id"], "duration"].values
        df_segs["dataset"] = self.dataset_name()

        classes = self.make_class_infos(df_segs)

        segments = SegmentSet(df_segs)

        enrollments = self.make_enrollments(df_enr)
        trials = None
        if self.partition != "initialization":
            trials = self.make_trials(segments, enrollments["enrollment"])
            enrollments = None

        logging.info("making dataset")
        dataset = HypDataset(
            segments,
            classes,
            recordings=recs,
            enrollments=enrollments,
            trials=trials,
        )
        logging.info("saving dataset at %s", self.output_dir)
        dataset.save(self.output_dir)
        dataset.describe()
