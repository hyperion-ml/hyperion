"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ..hyp_defs import float_cpu
from ..io import RandomAccessAudioReader as RAR
from ..io import SequentialAudioReader as AR
from ..np.metrics import (
    compute_cer,
    compute_lsd,
    compute_pesq,
    compute_si_snr,
    compute_snr,
    compute_stoi,
    compute_wer,
)
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

REF_METRIC_COLUMNS = [
    "snr",
    "si_snr",
    "stoi",
    "estoi",
    "pesq",
    "lsd",
    "si_pesq",
    "si_lsd",
]

QUALITY_METRIC_COLUMNS = MOS_METRIC_COLUMNS + REF_METRIC_COLUMNS


class SpeechQualityEvaluator:
    """
    A class to evaluate speech quality and ASR accuracy on a given dataset of recordings and segments.

    This evaluator can:
      - Transcribe audio using Whisper and compute WER/CER against reference transcripts.
      - Predict speech quality using UTMOSv2 and calculate MOS scores.
      - Predict DNSMOS quality scores.
      - Compute intrusive quality metrics against reference audio (SNR, SI-SNR,
        STOI/ESTOI, PESQ, LSD and scale-invariant PESQ/LSD) when reference
        segments/recordings are provided.
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
        ref_segments (Optional[SegmentSet]):
            Optional reference segment table for intrusive metrics.
        ref_recordings (Optional[RecordingSet]):
            Optional reference recording table for intrusive metrics.
        ref_column (str):
            Column in ``segments`` that points to the reference segment id.
        use_ref_metrics (bool):
            Whether intrusive reference metrics are enabled.
        part_idx (int):
            Index of the current data shard (1-based). Used for parallelism.
        num_parts (int):
            Total number of shards or splits for evaluation.

    Example:
        ```python
        from hyperion.metrics import SpeechQualityEvaluator

        evaluator = SpeechQualityEvaluator(
            segments="data/eval_segments.csv",
            recordings="data/eval_recordings.csv",
            whisper={"model_id": "openai/whisper-large-v3"},
            transcript_name="transcript",
            use_whisper=True,
            use_utmos=True,
            use_dnsmos=False,
            ref_segments="data/ref_segments.csv",
            ref_recordings="data/ref_recordings.csv",
            ref_column="clean_id",
        )
        stats, segments_scored = evaluator(return_df=True)
        ```
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
        ref_segments: Union[SegmentSet, PathLike, None] = None,
        ref_recordings: Union[RecordingSet, PathLike, None] = None,
        ref_column: str = "id",
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
            ref_segments (SegmentSet or PathLike, optional): Reference segments used to
                compute intrusive quality metrics.
            ref_recordings (RecordingSet or PathLike, optional): Reference recordings
                used to load reference speech for intrusive quality metrics.
            ref_column (str): Column in ``segments`` containing the reference segment id.
            part_idx (int): Index of this data split (1-based).
            num_parts (int): Total number of splits for parallel processing.

        Notes:
            Intrusive metrics are only enabled when both ``ref_segments`` and
            ``ref_recordings`` are provided. The ``ref_column`` must exist in
            ``segments`` and is used to map each evaluated segment to a
            reference segment id.
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
        self.ref_column = ref_column
        self.ref_segments = None
        self.ref_recordings = None
        self.use_ref_metrics = False
        self._warned_unsupported_pesq_fs = set()
        self._disabled_ref_metric_groups = set()
        self._warned_missing_ref_metric_deps = set()

        if isinstance(ref_segments, (str, Path)):
            ref_segments = SegmentSet.load(ref_segments)

        if isinstance(ref_recordings, (str, Path)):
            ref_recordings = RecordingSet.load(ref_recordings)

        if (ref_segments is None) != (ref_recordings is None):
            logging.warning(
                "intrusive reference metrics require both ref_segments and "
                "ref_recordings; disabling reference metrics"
            )
        elif ref_segments is not None and ref_recordings is not None:
            if ref_column not in segments:
                raise ValueError(
                    f"ref_column={ref_column} not found in segments columns"
                )
            self.ref_segments = ref_segments
            self.ref_recordings = ref_recordings
            self.use_ref_metrics = True

        self.part_idx = part_idx
        self.num_parts = num_parts

    def _compute_ref_metrics(
        self, seg_data: Any, ref_seg_data: Any
    ) -> Dict[str, Optional[float]]:
        """Compute intrusive quality metrics against a reference segment.

        Args:
            seg_data: Tuple ``(x, fs)`` with waveform and sampling frequency for
                the evaluated segment.
            ref_seg_data: Tuple ``(x_ref, fs_ref)`` with waveform and sampling
                frequency for the reference segment.

        Returns:
            dict: Metric dictionary with keys listed in
            ``REF_METRIC_COLUMNS``. Values are floats when available and
            ``np.nan`` when a metric cannot be computed (e.g., unsupported PESQ
            sample rate).

        Raises:
            ValueError: If the segment and reference sample rates do not match.
        """
        metrics: Dict[str, Optional[float]] = {k: np.nan for k in REF_METRIC_COLUMNS}
        x, fs = seg_data
        x_ref, ref_fs = ref_seg_data

        if fs != ref_fs:
            raise ValueError(
                f"sampling-rate mismatch between segment and reference: fs={fs} "
                f"ref_fs={ref_fs}"
            )

        x = np.asarray(x, dtype=float_cpu())
        x_ref = np.asarray(x_ref, dtype=float_cpu())

        if x.ndim > 1:
            x = np.squeeze(x)
            if x.ndim > 1:
                x = x[0]
        if x_ref.ndim > 1:
            x_ref = np.squeeze(x_ref)
            if x_ref.ndim > 1:
                x_ref = x_ref[0]

        x = np.asarray(x, dtype=float_cpu()).reshape(-1)
        x_ref = np.asarray(x_ref, dtype=float_cpu()).reshape(-1)
        num_samples = min(x.shape[0], x_ref.shape[0])
        if num_samples == 0:
            return metrics

        x = x[:num_samples]
        x_ref = x_ref[:num_samples]

        metrics["snr"] = float(compute_snr(pred=x, target=x_ref))
        metrics["si_snr"] = float(compute_si_snr(pred=x, target=x_ref))
        metrics["lsd"] = float(compute_lsd(pred=x, target=x_ref))
        metrics["si_lsd"] = float(
            compute_lsd(pred=x, target=x_ref, scale_invariant=True)
        )

        if "stoi" not in self._disabled_ref_metric_groups:
            try:
                metrics["stoi"] = float(compute_stoi(pred=x, target=x_ref, fs=fs))
                metrics["estoi"] = float(
                    compute_stoi(pred=x, target=x_ref, fs=fs, extended=True)
                )
            except ImportError as e:
                self._disabled_ref_metric_groups.add("stoi")
                if "stoi" not in self._warned_missing_ref_metric_deps:
                    logging.warning(
                        "disabling STOI/ESTOI metrics because dependency is "
                        "missing: %s",
                        e,
                    )
                    self._warned_missing_ref_metric_deps.add("stoi")

        if fs in (8000, 16000):
            if "pesq" not in self._disabled_ref_metric_groups:
                try:
                    metrics["pesq"] = float(compute_pesq(pred=x, target=x_ref, fs=fs))
                    metrics["si_pesq"] = float(
                        compute_pesq(
                            pred=x, target=x_ref, fs=fs, scale_invariant=True
                        )
                    )
                except ImportError as e:
                    self._disabled_ref_metric_groups.add("pesq")
                    if "pesq" not in self._warned_missing_ref_metric_deps:
                        logging.warning(
                            "disabling PESQ metrics because dependency is "
                            "missing: %s",
                            e,
                        )
                        self._warned_missing_ref_metric_deps.add("pesq")
        elif fs not in self._warned_unsupported_pesq_fs:
            logging.warning(
                "skipping PESQ metrics for fs=%s. PESQ only supports fs in {8000, 16000}",
                fs,
            )
            self._warned_unsupported_pesq_fs.add(fs)

        return metrics

    def __call__(self, return_df: bool = True):
        """
        Run speech quality and ASR evaluation on the current dataset slice.

        For each audio segment:
          - Uses Whisper (if enabled) to generate transcriptions and compute WER/CER.
          - Uses UTMOSv2 (if enabled) to compute predicted MOS scores.
          - Uses DNSMOS (if enabled) to compute MOS-related scores.
          - Uses intrusive reference metrics (if reference data is configured)
            by mapping each segment via ``ref_column``.

        Returns:
            Tuple[Dict[str, Any] or pd.DataFrame, SegmentSet]:
                Aggregated statistics and the updated segment metadata with
                per-segment evaluation results. Statistics are returned as a
                ``pd.DataFrame`` when ``return_df=True``.
        """
        segments = self.segments
        text_normalizer = EnglishTextNormalizer()
        stats = {}
        if self.use_whisper:
            whisper = WhisperTranscriber(**self.whisper)
            whisper_hyp = []
            whisper_ref = []
            whisper_utt_ids = []

        if self.use_utmos:
            utmos = UTMOSV2()

        if self.use_dnsmos:
            dnsmos = DNSMOS(**self.dnsmos_args)

        with ExitStack() as stack:
            reader = stack.enter_context(
                AR(
                    recordings=self.recordings,
                    segments=self.segments,
                )
            )
            ref_reader = None
            if self.use_ref_metrics:
                ref_reader = stack.enter_context(
                    RAR(
                        recordings=self.ref_recordings,
                        segments=self.ref_segments,
                    )
                )

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
                        whisper_utt_ids.append(segment_id)

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

                if ref_reader is not None:
                    ref_segment_id = segments.at[segment_id, self.ref_column]
                    if pd.isna(ref_segment_id):
                        continue

                    ref_segment_id = str(ref_segment_id)
                    if ref_segment_id not in self.ref_segments.index:
                        logging.warning(
                            "reference segment %s not found for segment %s",
                            ref_segment_id,
                            segment_id,
                        )
                        continue

                    try:
                        ref_data = ref_reader.read(ref_segment_id)
                        ref_x, ref_fs = ref_data[0][0], ref_data[1][0]
                    except Exception as e:
                        logging.warning(
                            "reference audio read failed for segment %s "
                            "with ref segment %s: %s",
                            segment_id,
                            ref_segment_id,
                            e,
                        )
                        continue

                    try:
                        ref_metrics = self._compute_ref_metrics(
                            seg_data=(x, fs), ref_seg_data=(ref_x, ref_fs)
                        )
                    except Exception as e:
                        logging.warning(
                            "reference metric computation failed for segment %s "
                            "with ref segment %s: %s",
                            segment_id,
                            ref_segment_id,
                            e,
                        )
                        continue

                    for metric_name, metric_value in ref_metrics.items():
                        segments.loc[segment_id, metric_name] = metric_value

        if self.use_whisper:
            segments.df["whisper_wer"] = np.nan
            segments.df["whisper_word_error_details"] = pd.NA
            segments.df["whisper_cer"] = np.nan
            segments.df["whisper_char_error_details"] = pd.NA

            if len(whisper_utt_ids) == 0:
                logging.warning(
                    "no non-null reference transcripts found in column %s; "
                    "skipping Whisper WER/CER computation",
                    self.transcript_name,
                )
            else:
                (
                    wer,
                    w_subs,
                    w_ins,
                    w_dels,
                    w_counts,
                    segment_stats,
                    word_stats,
                    sub_stats,
                ) = compute_wer(whisper_hyp, whisper_ref, utt_ids=whisper_utt_ids)
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
                ) = compute_cer(whisper_hyp, whisper_ref, utt_ids=whisper_utt_ids)
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
    def compute_stats(segments: SegmentSet, stats: Optional[Dict[str, Any]] = None):
        """
        Compute summary statistics from the updated segment set.

        For every available metric in ``QUALITY_METRIC_COLUMNS``, computes
        mean, standard deviation, and first/second-order accumulators for
        later split-wise aggregation.

        Args:
            segments (SegmentSet): Evaluated segment metadata.
            stats (dict, optional): Optional dictionary to populate/extend.

        Returns:
            dict: Aggregated metric statistics and segment count.
        """
        if stats is None:
            stats = {}

        stats["num_segments"] = len(segments)
        for key in QUALITY_METRIC_COLUMNS:
            if key in segments:
                metric_values = pd.to_numeric(segments[key], errors="coerce")
                metric_count = int(metric_values.count())
                metric_acc = float(metric_values.sum()) if metric_count > 0 else 0.0
                metric_acc2 = (
                    float((metric_values**2).sum()) if metric_count > 0 else 0.0
                )
                if metric_count > 0:
                    metric_mean = metric_acc / metric_count
                    metric_var = max(metric_acc2 / metric_count - metric_mean**2, 0.0)
                    metric_std = float(np.sqrt(metric_var))
                else:
                    metric_mean = np.nan
                    metric_std = np.nan

                stats[f"{key}_count"] = metric_count
                stats[f"{key}_mean"] = metric_mean
                stats[f"{key}_std"] = metric_std
                stats[f"{key}_acc"] = metric_acc
                stats[f"{key}2_acc"] = metric_acc2

        return stats

    @staticmethod
    def accum_stats(stats: List[Union[Dict[str, Any], pd.DataFrame]]):
        """
        Merge statistics from multiple splits (e.g., from distributed evaluation).

        Sums per-split accumulator fields, recomputes means/std for all metric
        keys in ``QUALITY_METRIC_COLUMNS`` when accumulators are present, and
        recomputes WER/CER from total operation counts.

        Args:
            stats (List[dict or pd.DataFrame]): A list of individual split
                statistics.

        Returns:
            dict or pd.DataFrame: Merged and normalized statistics in the same
            container style as the input.
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

        for key in QUALITY_METRIC_COLUMNS:
            if (
                f"{key}_count" not in accum_stats
                or f"{key}_acc" not in accum_stats
                or f"{key}2_acc" not in accum_stats
            ):
                continue

            metric_count = int(accum_stats[f"{key}_count"])
            if metric_count > 0:
                accum_stats[f"{key}_mean"] = accum_stats[f"{key}_acc"] / metric_count
                metric_var = max(
                    accum_stats[f"{key}2_acc"] / metric_count
                    - accum_stats[f"{key}_mean"] ** 2,
                    0.0,
                )
                accum_stats[f"{key}_std"] = float(np.sqrt(metric_var))
            else:
                accum_stats[f"{key}_mean"] = np.nan
                accum_stats[f"{key}_std"] = np.nan

        if "whisper_wer" in accum_stats:
            if accum_stats["whisper_num_words"] > 0:
                accum_stats["whisper_wer"] = (
                    accum_stats["whisper_word_subs"]
                    + accum_stats["whisper_word_ins"]
                    + accum_stats["whisper_word_dels"]
                ) / accum_stats["whisper_num_words"]
            else:
                accum_stats["whisper_wer"] = np.nan

        if "whisper_cer" in accum_stats:
            if accum_stats["whisper_num_chars"] > 0:
                accum_stats["whisper_cer"] = (
                    accum_stats["whisper_char_subs"]
                    + accum_stats["whisper_char_ins"]
                    + accum_stats["whisper_char_dels"]
                ) / accum_stats["whisper_num_chars"]
            else:
                accum_stats["whisper_cer"] = np.nan

        if return_df:
            accum_stats = pd.DataFrame([accum_stats])

        return accum_stats

    @staticmethod
    def add_class_args(parser, prefix=None, skip=None):
        """
        Add command-line arguments for configuring SpeechQualityEvaluator.

        This method adds arguments needed to instantiate the class via the CLI.
        It supports optional prefixing for nested argument groups and exclusion
        of specific fields, including optional intrusive reference-metric
        inputs.

        Args:
            parser (ArgumentParser): A jsonargparse parser instance.
            prefix (str, optional): If given, wraps arguments under this namespace using ActionParser.
            skip (set, optional): A set of argument names to exclude from the parser.
        """
        if skip is None:
            skip = set()

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

        if "ref_segments" not in skip:
            parser.add_argument(
                "--ref-segments",
                type=str,
                default=None,
                help="Path to reference segments table for intrusive metrics.",
            )

        if "ref_recordings" not in skip:
            parser.add_argument(
                "--ref-recordings",
                type=str,
                default=None,
                help="Path to reference recordings table for intrusive metrics.",
            )

        if "ref_column" not in skip:
            parser.add_argument(
                "--ref-column",
                type=str,
                default="id",
                help=(
                    "Column in the evaluation segments table containing the reference "
                    "segment id."
                ),
            )

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
