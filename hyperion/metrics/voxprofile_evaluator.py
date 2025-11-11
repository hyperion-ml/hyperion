"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ..hyp_defs import float_cpu
from ..io import SequentialAudioReader as AR
from ..torch.tpm.usc import (
    VoxProfileAgeSexEvaluator,
    VoxProfileBroadAccentEvaluator,
    VoxProfileCategoricalEmotionEvaluator,
    VoxProfileDimensionalEmotionEvaluator,
    VoxProfileFluencyEvaluator,
    VoxProfileNarrowAccentEvaluator,
    VoxProfileVoiceQualityEvaluator,
)
from ..utils import RecordingSet, SegmentSet
from ..utils.misc import PathLike, filter_func_args


class VoxProfileEvaluator:
    """Coordinate multiple VoxProfile attribute evaluators for a segment set.

    Attributes:
        segments: Segment metadata enriched in-place with evaluation outputs.
        recordings: Recording metadata used to load audio waveforms.
        narrow_accent/broad_accent/...: Configuration dictionaries forwarded to
            the corresponding evaluator constructors.
        use_*: Boolean toggles enabling each evaluator.
        part_idx: Index of the current data shard (1-based).
        num_parts: Total number of shards across which the dataset is split.
        sequential_model_evaluation: Whether to evaluate each model over every
            segment sequentially (currently unimplemented).
        batch_size: Number of segments evaluated together when running models
            in parallel mode.
    """

    def __init__(
        self,
        segments: Union[SegmentSet, PathLike],
        recordings: Union[RecordingSet, PathLike],
        narrow_accent: Dict[str, Any] = {},
        broad_accent: Dict[str, Any] = {},
        categorical_emotion: Dict[str, Any] = {},
        dimensional_emotion: Dict[str, Any] = {},
        fluency: Dict[str, Any] = {},
        voice_quality: Dict[str, Any] = {},
        agesex: Dict[str, Any] = {},
        use_narrow_accent: bool = True,
        use_broad_accent: bool = True,
        use_categorical_emotion: bool = True,
        use_dimensional_emotion: bool = True,
        use_fluency: bool = True,
        use_voice_quality: bool = True,
        use_agesex: bool = True,
        part_idx: int = 1,
        num_parts: int = 1,
        sequential_model_evaluation: bool = False,
        batch_size: int = 1,
    ):
        """Instantiate a :class:`VoxProfileEvaluator`.

        Args:
            segments: Segment metadata or a path to it.
            recordings: Recording metadata or a path to it.
            narrow_accent: Keyword arguments for
                :class:`VoxProfileNarrowAccentEvaluator`.
            broad_accent: Keyword arguments for
                :class:`VoxProfileBroadAccentEvaluator`.
            categorical_emotion: Keyword arguments for
                :class:`VoxProfileCategoricalEmotionEvaluator`.
            dimensional_emotion: Keyword arguments for
                :class:`VoxProfileDimensionalEmotionEvaluator`.
            fluency: Keyword arguments for :class:`VoxProfileFluencyEvaluator`.
            voice_quality: Keyword arguments for
                :class:`VoxProfileVoiceQualityEvaluator`.
            agesex: Keyword arguments for :class:`VoxProfileAgeSexEvaluator`.
            use_*: Boolean toggles enabling each evaluator.
            part_idx: Index of this data split (1-based).
            num_parts: Total number of splits for sharded evaluation.
            sequential_model_evaluation: When ``True`` evaluate each model
                sequentially (not yet implemented).
            batch_size: Number of segments processed together in parallel mode.
        """
        if isinstance(segments, (str, Path)):
            segments = SegmentSet.load(segments)

        if isinstance(recordings, (str, Path)):
            recordings = RecordingSet.load(recordings)

        if num_parts > 1:
            segments = segments.split(idx=part_idx, num_parts=num_parts)

        self.segments = segments
        self.recordings = recordings
        self.narrow_accent = narrow_accent
        self.broad_accent = broad_accent
        self.categorical_emotion = categorical_emotion
        self.dimensional_emotion = dimensional_emotion
        self.fluency = fluency
        self.voice_quality = voice_quality
        self.agesex = agesex
        self.use_narrow_accent = use_narrow_accent
        self.use_broad_accent = use_broad_accent
        self.use_categorical_emotion = use_categorical_emotion
        self.use_dimensional_emotion = use_dimensional_emotion
        self.use_fluency = use_fluency
        self.use_voice_quality = use_voice_quality
        self.use_agesex = use_agesex
        self.part_idx = part_idx
        self.num_parts = num_parts
        self.sequential_model_evaluation = sequential_model_evaluation
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.batch_size = batch_size

    def _evaluate_models_parallel(
        self, return_df: bool = True
    ) -> Tuple[Union[Dict[str, Any], pd.DataFrame], SegmentSet]:
        """Evaluate enabled models in batched parallel fashion.

        Segments are streamed from :class:`SequentialAudioReader` in groups of
        ``self.batch_size`` and each enabled evaluator is run once per batch.

        Args:
            return_df: When ``True`` wrap the statistics dictionary in a single
                row :class:`pandas.DataFrame`.

        Returns:
            Tuple containing aggregated statistics (dictionary or DataFrame) and
            the updated ``SegmentSet``.
        """
        segments = self.segments
        stats: Dict[str, Any] = {}

        evaluators = {}
        if self.use_narrow_accent:
            evaluators["narrow_accent"] = VoxProfileNarrowAccentEvaluator(
                **self.narrow_accent
            )
        if self.use_broad_accent:
            evaluators["broad_accent"] = VoxProfileBroadAccentEvaluator(
                **self.broad_accent
            )
        if self.use_categorical_emotion:
            evaluators["categorical_emotion"] = VoxProfileCategoricalEmotionEvaluator(
                **self.categorical_emotion
            )
        if self.use_dimensional_emotion:
            evaluators["dimensional_emotion"] = VoxProfileDimensionalEmotionEvaluator(
                **self.dimensional_emotion
            )
        if self.use_fluency:
            evaluators["fluency"] = VoxProfileFluencyEvaluator(**self.fluency)
        if self.use_voice_quality:
            evaluators["voice_quality"] = VoxProfileVoiceQualityEvaluator(
                **self.voice_quality
            )
        if self.use_agesex:
            evaluators["agesex"] = VoxProfileAgeSexEvaluator(**self.agesex)

        with AR(
            recordings=self.recordings,
            segments=self.segments,
        ) as reader:
            while not reader.eof():
                batch = reader.read(self.batch_size)
                segment_ids, audios, audio_fs = batch[:3]
                logging.info("Processing batch with %s segments", str(segment_ids))

                if not segment_ids:
                    break

                batch_df = pd.DataFrame(index=segment_ids)

                if "narrow_accent" in evaluators:
                    df_accent = evaluators["narrow_accent"](
                        audios=audios, audio_fs=audio_fs, audio_ids=segment_ids
                    )
                    batch_df = batch_df.join(df_accent, how="left")

                if "broad_accent" in evaluators:
                    df_accent = evaluators["broad_accent"](
                        audios=audios, audio_fs=audio_fs, audio_ids=segment_ids
                    )
                    batch_df = batch_df.join(df_accent, how="left")

                if "categorical_emotion" in evaluators:
                    df_emotion = evaluators["categorical_emotion"](
                        audios=audios, audio_fs=audio_fs, audio_ids=segment_ids
                    )
                    batch_df = batch_df.join(df_emotion, how="left")

                if "dimensional_emotion" in evaluators:
                    df_emotion = evaluators["dimensional_emotion"](
                        audios=audios, audio_fs=audio_fs, audio_ids=segment_ids
                    )
                    batch_df = batch_df.join(df_emotion, how="left")

                if "agesex" in evaluators:
                    df_agesex = evaluators["agesex"](
                        audios=audios, audio_fs=audio_fs, audio_ids=segment_ids
                    )
                    batch_df = batch_df.join(df_agesex, how="left")

                if "voice_quality" in evaluators:
                    df_voice_quality = evaluators["voice_quality"](
                        audios=audios, audio_fs=audio_fs, audio_ids=segment_ids
                    )
                    batch_df = batch_df.join(df_voice_quality, how="left")

                if "fluency" in evaluators:
                    df_fluency = evaluators["fluency"](
                        audios=audios, audio_fs=audio_fs, audio_ids=segment_ids
                    )
                    batch_df = batch_df.join(df_fluency, how="left")

                for col in batch_df.columns:
                    if col not in segments.df.columns:
                        segments.df[col] = pd.NA
                    segments.df.loc[segment_ids, col] = batch_df[col].values

        stats = self.compute_stats(segments, stats)
        if return_df:
            stats = pd.DataFrame([stats])
        return stats, segments

    def _evaluate_models_sequential(self, return_df: bool = True):
        """Evaluate each enabled model sequentially across the dataset."""
        segments = self.segments
        stats: Dict[str, Any] = {}

        evaluator_confs = [
            (
                "narrow_accent",
                self.use_narrow_accent,
                VoxProfileNarrowAccentEvaluator,
                self.narrow_accent,
            ),
            (
                "broad_accent",
                self.use_broad_accent,
                VoxProfileBroadAccentEvaluator,
                self.broad_accent,
            ),
            (
                "categorical_emotion",
                self.use_categorical_emotion,
                VoxProfileCategoricalEmotionEvaluator,
                self.categorical_emotion,
            ),
            (
                "dimensional_emotion",
                self.use_dimensional_emotion,
                VoxProfileDimensionalEmotionEvaluator,
                self.dimensional_emotion,
            ),
            ("fluency", self.use_fluency, VoxProfileFluencyEvaluator, self.fluency),
            (
                "voice_quality",
                self.use_voice_quality,
                VoxProfileVoiceQualityEvaluator,
                self.voice_quality,
            ),
            ("agesex", self.use_agesex, VoxProfileAgeSexEvaluator, self.agesex),
        ]

        for name, enabled, evaluator_cls, evaluator_kwargs in evaluator_confs:
            if not enabled:
                continue

            evaluator = evaluator_cls(**evaluator_kwargs)

            with AR(
                recordings=self.recordings,
                segments=self.segments,
            ) as reader:
                while not reader.eof():
                    batch = reader.read(self.batch_size)
                    segment_ids, audios, audio_fs = batch[:3]
                    logging.info(
                        "Processing batch with %s segments for %s evaluator",
                        str(segment_ids),
                        name,
                    )

                    if not segment_ids:
                        break

                    df = evaluator(
                        audios=audios, audio_fs=audio_fs, audio_ids=segment_ids
                    )

                    for col in df.columns:
                        if col not in segments.df.columns:
                            segments.df[col] = pd.NA
                        segments.df.loc[segment_ids, col] = df[col].values

            del evaluator

        stats = self.compute_stats(segments, stats)
        if return_df:
            stats = pd.DataFrame([stats])
        return stats, segments

    def __call__(
        self, return_df: bool = True
    ) -> Tuple[Union[Dict[str, Any], pd.DataFrame], SegmentSet]:
        """Evaluate the configured VoxProfile metrics.

        Args:
            return_df: When ``True`` wrap statistics in a single-row DataFrame,
                otherwise return a plain dictionary.

        Returns:
            Tuple of aggregated statistics and the updated ``SegmentSet``.
        """
        if self.sequential_model_evaluation:
            return self._evaluate_models_sequential(return_df=return_df)
        else:
            return self._evaluate_models_parallel(return_df=return_df)

    def compute_stats(
        self, segments: SegmentSet, stats: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute aggregate statistics from the enriched segment table.

        Currently records the number of segments and, for columns ending with
        ``"_fluency"``, the share of non-null entries equal to ``"fluent"``.

        Args:
            segments: Segment metadata containing evaluator outputs.
            stats: Optional dictionary to populate; a new one is created when
                ``None``.

        Returns:
            Dictionary with accumulated statistics.
        """
        if stats is None:
            stats = {}
        stats["num_segments"] = len(segments)
        for col in segments.df.columns:
            if col.endswith("_fluency"):
                col_values = segments.df[col].dropna()
                stats[f"{col}_fluent_prob"] = (
                    (col_values == "fluent").astype(float).mean()
                    if len(col_values) > 0
                    else np.nan
                )
        return stats

    @staticmethod
    def accum_stats(
        stats: List[Union[Dict[str, Any], pd.DataFrame]],
    ) -> Union[Dict[str, Any], pd.DataFrame]:
        """Merge statistics produced by multiple evaluator shards.

        Args:
            stats: List of statistics dictionaries or single-row DataFrames.

        Returns:
            Aggregated statistics matching the input format (dictionary or
            DataFrame).
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

        if return_df:
            accum_stats = pd.DataFrame([accum_stats])

        return accum_stats

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[set] = None,
    ) -> None:
        """Register CLI arguments for building a :class:`VoxProfileEvaluator`."""
        if skip is None:
            skip = set()
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "narrow_accent" not in skip:
            VoxProfileNarrowAccentEvaluator.add_class_args(
                parser, prefix="narrow_accent"
            )

        if "broad_accent" not in skip:
            VoxProfileBroadAccentEvaluator.add_class_args(parser, prefix="broad_accent")

        if "categorical_emotion" not in skip:
            VoxProfileCategoricalEmotionEvaluator.add_class_args(
                parser, prefix="categorical_emotion"
            )

        if "dimensional_emotion" not in skip:
            VoxProfileDimensionalEmotionEvaluator.add_class_args(
                parser, prefix="dimensional_emotion"
            )

        if "fluency" not in skip:
            VoxProfileFluencyEvaluator.add_class_args(parser, prefix="fluency")

        if "voice_quality" not in skip:
            VoxProfileVoiceQualityEvaluator.add_class_args(
                parser, prefix="voice_quality"
            )

        if "agesex" not in skip:
            VoxProfileAgeSexEvaluator.add_class_args(parser, prefix="agesex")

        if "use_narrow_accent" not in skip:
            parser.add_argument(
                "--use-narrow-accent",
                action=ActionYesNo,
                default=True,
                help="Enable narrow accent evaluation.",
            )

        if "use_broad_accent" not in skip:
            parser.add_argument(
                "--use-broad-accent",
                action=ActionYesNo,
                default=True,
                help="Enable broad accent evaluation.",
            )

        if "use_categorical_emotion" not in skip:
            parser.add_argument(
                "--use-categorical-emotion",
                action=ActionYesNo,
                default=True,
                help="Enable categorical emotion evaluation.",
            )

        if "use_dimensional_emotion" not in skip:
            parser.add_argument(
                "--use-dimensional-emotion",
                action=ActionYesNo,
                default=True,
                help="Enable dimensional emotion evaluation.",
            )

        if "use_fluency" not in skip:
            parser.add_argument(
                "--use-fluency",
                action=ActionYesNo,
                default=True,
                help="Enable fluency evaluation.",
            )

        if "use_voice_quality" not in skip:
            parser.add_argument(
                "--use-voice-quality",
                action=ActionYesNo,
                default=True,
                help="Enable voice quality evaluation.",
            )

        if "use_agesex" not in skip:
            parser.add_argument(
                "--use-agesex",
                action=ActionYesNo,
                default=True,
                help="Enable age and sex evaluation.",
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

        if "sequential_model_evaluation" not in skip:
            parser.add_argument(
                "--sequential-model-evaluation",
                action=ActionYesNo,
                default=True,
                help="If true, evaluates all segments for each model before moving to the next model.",
            )
        if "batch_size" not in skip:
            parser.add_argument(
                "--batch-size",
                type=int,
                default=1,
                help="Number of segments processed together when evaluating models.",
            )

        if prefix is not None:
            outer_parser.add_argument(f"--{prefix}", action=ActionParser(parser=parser))

    @staticmethod
    def filter_args(**kwargs) -> Dict[str, Any]:
        """Filter and return arguments relevant to the VoxProfileEvaluator."""
        return filter_func_args(VoxProfileEvaluator.__init__, **kwargs)
