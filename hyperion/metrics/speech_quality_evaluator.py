"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ..hyp_defs import float_cpu
from ..io import SequentialAudioReader as AR
from ..np.metrics import compute_cer, compute_wer
from ..text_norm import EnglishTextNormalizer
from ..torch.tpm.dnsmos import DNSMOS
from ..torch.tpm.hf import WhisperTranscriber
from ..torch.tpm.utmos import UTMOSV2
from ..utils import RecordingSet, SegmentSet
from ..utils.misc import PathLike

MOS_METRIC_COLUMNS = [
    "utmos",
    "p835_ovrl_raw",
    "p835_sig_raw",
    "p835_bak_raw",
    "p835_ovrl",
    "p835_sig",
    "p835_bak",
    "p808_mos",
]


class SpeechQualityEvaluator:
    """
    A class to evaluate speech quality and ASR accuracy on a given dataset of recordings and segments.

    This evaluator can:
      - Transcribe audio using Whisper and compute WER/CER against reference transcripts.
      - Predict speech quality using UTMOSv2 and calculate MOS scores.
      - Support sharding (splitting) for parallel or distributed evaluation.

    Attributes:
        segments (SegmentSet):
            The segmented speech metadata used for evaluation. If a path is provided, it will be loaded automatically.
        recordings (RecordingSet):
            The set of audio recordings. If a path is provided, it will be loaded automatically.
        whisper (dict):
            Configuration dictionary passed to the WhisperTranscriber instance.
        transcript_name (str):
            Name of the column in `segments` containing the reference transcription.
        use_whisper (bool):
            Whether to use Whisper for transcription.
        use_utmos (bool):
            Whether to use UTMOSv2 to compute MOS scores.
        use_dnsmos (bool):
            Whether to use DNSMOS to compute MOS scores.
        part_idx (int):
            Index of the current data shard (1-based). Used for parallelism.
        num_parts (int):
            Total number of shards or splits for evaluation.
    """

    def __init__(
        self,
        segments: Union[SegmentSet, PathLike],
        recordings: Union[RecordingSet, PathLike],
        whisper: Dict[str, Any],
        transcript_name: str = "transcript",
        use_whisper: bool = True,
        use_utmos: bool = True,
        use_dnsmos: bool = True,
        dnsmos: Optional[Dict[str, Any]] = None,
        part_idx: int = 1,
        num_parts: int = 1,
    ):
        """
        Initialize a SpeechQualityEvaluator for evaluating ASR and speech quality metrics.

        Args:
            segments (SegmentSet or PathLike): Segment metadata or path to it.
            recordings (RecordingSet or PathLike): Recording metadata or path to it.
            whisper (dict): Configuration for the WhisperTranscriber.
            transcript_name (str): Name of the reference transcript column in the segments file.
            use_whisper (bool): Whether to transcribe using Whisper.
            use_utmos (bool): Whether to compute MOS scores using UTMOSv2.
            use_dnsmos (bool): Whether to compute MOS scores using DNSMOS.
            dnsmos (dict, optional): Configuration dictionary passed to the DNSMOS wrapper.
            part_idx (int): Index of this data split (1-based).
            num_parts (int): Total number of splits for parallel processing.
        """
        if isinstance(segments, (str, Path)):
            segments = SegmentSet.load(segments)

        if isinstance(recordings, (str, Path)):
            recordings = RecordingSet.load(recordings)

        if num_parts > 1:
            segments = segments.split(idx=part_idx, num_parts=num_parts)

        self.segments = segments
        self.recordings = recordings
        self.whisper = whisper
        if transcript_name not in segments:
            if use_whisper:
                logging.warning(
                    "reference transcript %s not in segments", transcript_name
                )
            use_whisper = False

        self.transcript_name = transcript_name
        self.use_whisper = use_whisper
        self.use_utmos = use_utmos
        self.use_dnsmos = use_dnsmos
        self.dnsmos_args = dnsmos or {}
        self.part_idx = part_idx
        self.num_parts = num_parts

    def __call__(self, return_df: bool = True):
        """
        Run speech quality and ASR evaluation on the current dataset slice.

        For each audio segment:
          - Uses Whisper (if enabled) to generate transcriptions and compute WER/CER.
          - Uses UTMOSv2 (if enabled) to compute predicted MOS scores.

        Returns:
            Tuple[Dict[str, Any], SegmentSet]: A dictionary of aggregated statistics,
                                               and the updated segment metadata with evaluation results.
        """
        segments = self.segments
        text_normalizer = EnglishTextNormalizer()
        stats = {}
        if self.use_whisper:
            whisper = WhisperTranscriber(**self.whisper)
            whisper_hyp = []
            whisper_ref = []

        if self.use_utmos:
            utmos = UTMOSV2()

        if self.use_dnsmos:
            dnsmos = DNSMOS(**self.dnsmos_args)

        with AR(
            recordings=self.recordings,
            segments=self.segments,
        ) as reader:
            while not reader.eof():
                seg_data = reader.read(1)
                segment_id, x, fs = seg_data[0][0], seg_data[1][0], seg_data[2][0]
                logging.info("Processing segment %s", segment_id)
                if self.use_whisper:
                    whisper_result = whisper(x, fs)
                    hyp = text_normalizer(whisper_result["text"])
                    ref = segments.at[segment_id, self.transcript_name]
                    if pd.notna(ref):
                        ref = text_normalizer(ref)
                        whisper_hyp.append(hyp)
                        whisper_ref.append(ref)

                    segments.loc[segment_id, "whisper_transcript"] = hyp

                if self.use_utmos:
                    utmos.add_audios(audios=[x], audio_fs=[fs], audio_ids=[segment_id])

                if self.use_dnsmos:
                    df_dnsmos = dnsmos(
                        audios=[x], audio_fs=[fs], audio_ids=[segment_id]
                    )
                    for col in df_dnsmos.columns:
                        if col != "id":
                            segments.loc[segment_id, col] = df_dnsmos.at[
                                segment_id, col
                            ]

        if self.use_whisper:
            (
                wer,
                w_subs,
                w_ins,
                w_dels,
                w_counts,
                segment_stats,
                word_stats,
                sub_stats,
            ) = compute_wer(whisper_hyp, whisper_ref, utt_ids=segments["id"])
            segments.df["whisper_wer"] = segment_stats["wer"]
            segments.df["whisper_word_error_details"] = segment_stats[
                "word_error_details"
            ]
            (
                cer,
                c_subs,
                c_ins,
                c_dels,
                c_counts,
                segment_stats,
                char_stats,
                sub_stats,
            ) = compute_cer(whisper_hyp, whisper_ref, utt_ids=segments["id"])
            segments.df["whisper_cer"] = segment_stats["cer"]
            segments.df["whisper_char_error_details"] = segment_stats[
                "char_error_details"
            ]
            for k, v in zip(
                [
                    "whisper_wer",
                    "whisper_word_subs",
                    "whisper_word_ins",
                    "whisper_word_dels",
                    "whisper_num_words",
                ],
                [wer, w_subs, w_ins, w_dels, w_counts],
            ):
                stats[k] = v

            for k, v in zip(
                [
                    "whisper_cer",
                    "whisper_char_subs",
                    "whisper_char_ins",
                    "whisper_char_dels",
                    "whisper_num_chars",
                ],
                [cer, c_subs, c_ins, c_dels, c_counts],
            ):
                stats[k] = v

        if self.use_utmos:
            logging.info("Predicting UTMOS scores")
            segment_ids, mos_pred = utmos()
            segments.loc[segment_ids, "utmos"] = mos_pred

        stats = self.compute_stats(segments, stats)
        if return_df:
            stats = pd.DataFrame([stats])
        return stats, segments

    @staticmethod
    def compute_stats(segments: SegmentSet, stats: Dict[str, Any] = {}):
        """
        Compute summary statistics from the updated segment set.

        If UTMOS scores are present, computes mean, standard deviation,
        and accumulators for further aggregation.

        Args:
            segments (SegmentSet): Evaluated segment metadata.
            stats (dict, optional): Optional dictionary to populate.

        Returns:
            dict: Aggregated statistics for UTMOS and segment count.
        """
        stats["num_segments"] = len(segments)
        for key in MOS_METRIC_COLUMNS:
            if key in segments:
                stats[f"{key}_mean"] = segments[key].mean()
                stats[f"{key}_std"] = segments[key].std()
                stats[f"{key}_acc"] = segments[key].sum()
                stats[f"{key}2_acc"] = (segments[key] ** 2).sum()

        return stats

    @staticmethod
    def accum_stats(stats: List[Union[Dict[str, Any], pd.DataFrame]]):
        """
        Merge statistics from multiple splits (e.g., from distributed evaluation).

        Aggregates UTMOS statistics, WER, and CER counts across all splits.

        Args:
            stats (List[dict]): A list of individual stats dictionaries from different parts.

        Returns:
            dict: A single dictionary with merged and normalized statistics.
        """
        return_df = False
        if isinstance(stats[0], pd.DataFrame):
            stats = pd.concat(stats)
            return_df = True
        else:
            stats = pd.DataFrame(stats)

        accum_stats = {
            col: (
                int(stats[col].sum())
                if pd.api.types.is_integer_dtype(stats[col])
                else stats[col].sum()
            )
            for col in stats.columns
        }

        for key in MOS_METRIC_COLUMNS:
            if f"{key}_acc" in accum_stats:
                accum_stats[f"{key}_mean"] = (
                    accum_stats[f"{key}_acc"] / accum_stats["num_segments"]
                )
            accum_stats[f"{key}_std"] = (
                accum_stats[f"{key}2_acc"] / accum_stats["num_segments"]
                - accum_stats[f"{key}_mean"] ** 2
            )

        if "whisper_wer" in accum_stats:
            accum_stats["whisper_wer"] = (
                accum_stats["whisper_word_subs"]
                + accum_stats["whisper_word_ins"]
                + accum_stats["whisper_word_dels"]
            ) / accum_stats["whisper_num_words"]

        if "whisper_cer" in accum_stats:
            accum_stats["whisper_cer"] = (
                accum_stats["whisper_char_subs"]
                + accum_stats["whisper_char_ins"]
                + accum_stats["whisper_char_dels"]
            ) / accum_stats["whisper_num_chars"]

        if return_df:
            accum_stats = pd.DataFrame([accum_stats])

        return accum_stats

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):
        """
        Add command-line arguments for configuring SpeechQualityEvaluator.

        This method adds arguments needed to instantiate the class via the CLI.
        It supports optional prefixing for nested argument groups and exclusion of specific fields.

        Args:
            parser (ArgumentParser): A jsonargparse parser instance.
            prefix (str, optional): If given, wraps arguments under this namespace using ActionParser.
            skip (set, optional): A set of argument names to exclude from the parser.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "whisper" not in skip:
            WhisperTranscriber.add_class_args(parser, prefix="whisper")

        if "transcript_name" not in skip:
            parser.add_argument(
                "--transcript-name",
                type=str,
                default="transcript",
                help="Column name in the segments file containing the reference transcript.",
            )

        if "use_whisper" not in skip:
            parser.add_argument(
                "--use-whisper",
                default=True,
                action=ActionYesNo,
                help="Use Whisper to generate transcriptions.",
            )

        if "use_utmos" not in skip:
            parser.add_argument(
                "--use-utmos",
                default=True,
                action=ActionYesNo,
                help="Use UTMOSv2 to evaluate speech quality.",
            )

        if "use_dnsmos" not in skip:
            parser.add_argument(
                "--use-dnsmos",
                default=True,
                action=ActionYesNo,
                help="Use DNSMOS to evaluate speech quality.",
            )

        if "dnsmos" not in skip:
            DNSMOS.add_class_args(parser, prefix="dnsmos")

        if "part_idx" not in skip:
            parser.add_argument(
                "--part-idx",
                type=int,
                default=1,
                help="Index of the current processing split (1-based).",
            )

        if "num_parts" not in skip:
            parser.add_argument(
                "--num-parts",
                type=int,
                default=1,
                help="Total number of processing splits (used for sharding).",
            )

        if prefix is not None:
            outer_parser.add_argument(f"--{prefix}", action=ActionParser(parser=parser))
