#!/usr/bin/env python
"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from pathlib import Path

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
    segments_file,
    recordings_file,
    global_metrics_file,
    segments_metrics_file,
    **kwargs,
):
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


def main():
    parser = ArgumentParser(description="Evaluate speaker verification metrics")
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--segments-file", required=True)
    parser.add_argument("--recordings-file", required=True)
    SQE.add_class_args(parser)
    parser.add_argument("--global-metrics-file", required=True)
    parser.add_argument("--segments-metrics-file", required=True)
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
    )

    args = parser.parse_args()
    kwargs = namespace_to_dict(args)
    config_logger(kwargs["verbose"])
    del kwargs["verbose"]
    del kwargs["cfg"]
    eval_speech_quality_metrics(**kwargs)


if __name__ == "__main__":
    main()
