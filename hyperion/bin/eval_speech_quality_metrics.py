#!/usr/bin/env python
"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ActionYesNo,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger
from hyperion.metrics import SpeechQualityEvaluator as SQE


def eval_speech_quality_metrics(
    segments_file: Union[str, Path],
    recordings_file: Union[str, Path],
    global_metrics_file: Union[str, Path],
    segments_metrics_file: Union[str, Path],
    **kwargs: Any,
) -> None:
    """
    Evaluate speech-quality metrics for a set of segments and recordings.

    Args:
        segments_file: Path to the manifest containing segment metadata.
        recordings_file: Path to the recordings manifest used by the evaluator.
        global_metrics_file: Output path for aggregated (global) metrics.
        segments_metrics_file: Output path for per-segment metrics.
        **kwargs: Additional keyword arguments forwarded to the evaluator.
    """
    logging.info(
        "Evaluating segments: %s recordings: %s", segments_file, recordings_file
    )
    evaluator = SQE(segments_file, recordings_file, **kwargs)
    stats, segments = evaluator()
    logging.info("saving segments metrics to %s", segments_metrics_file)
    segments.save(segments_metrics_file)
    logging.info("saving global metrics to %s", global_metrics_file)
    global_metrics_file = Path(global_metrics_file)
    global_metrics_file.parent.mkdir(exist_ok=True, parents=True)
    sep = "\t" if global_metrics_file.suffix == ".tsv" else ","
    stats.to_csv(
        global_metrics_file, sep=sep, index=False, float_format="{:.3f}".format
    )
    pd.options.display.float_format = "{:.3}".format
    print(stats.to_string(), flush=True)


def main() -> None:
    """CLI entry point for speech-quality metric evaluation."""
    parser = ArgumentParser(description="Evaluate speech quality metrics")
    parser.add_argument(
        "--cfg",
        action=ActionConfigFile,
        help="Load command-line options from the provided configuration file",
    )
    parser.add_argument(
        "--segments-file",
        required=True,
        help="Path to the segments manifest used for evaluation",
    )
    parser.add_argument(
        "--recordings-file",
        required=True,
        help="Path to the recordings manifest consumed by the evaluator",
    )
    SQE.add_class_args(parser)
    parser.add_argument(
        "--global-metrics-file",
        required=True,
        help="File where aggregated (global) metrics will be written",
    )
    parser.add_argument(
        "--segments-metrics-file",
        required=True,
        help="File where per-segment metrics will be written",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="Verbosity level: 0=warning, 1=info, 2=debug, 3=trace",
    )

    args = parser.parse_args()
    kwargs: Dict[str, Any] = namespace_to_dict(args)
    config_logger(kwargs["verbose"])
    del kwargs["verbose"]
    del kwargs["cfg"]
    eval_speech_quality_metrics(**kwargs)


if __name__ == "__main__":
    main()
