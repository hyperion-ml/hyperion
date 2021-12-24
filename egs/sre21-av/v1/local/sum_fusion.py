#!/bin/env python
"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ArgumentParser, namespace_to_dict
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from hyperion.hyp_defs import config_logger


def sum_fusion(ndx_file, audio_scores, visual_scores, output_scores, verbose):
    config_logger(verbose)
    df_ndx = pd.read_csv(
        ndx_file, sep=",", usecols=["model_id", "image_id", "segment_id"]
    )
    df_audio = pd.read_csv(
        audio_scores,
        sep=" ",
        header=None,
        names=["model_id", "segment_id", "score_audio"],
    )
    df_visual = pd.read_csv(
        visual_scores,
        sep=" ",
        header=None,
        names=["image_id", "segment_id", "score_visual"],
    )
    df_av = pd.merge(df_ndx, df_audio, on=["model_id", "segment_id"], how="left")
    df_av = pd.merge(df_av, df_visual, on=["image_id", "segment_id"], how="left")
    df_av["score"] = df_av["score_audio"] + df_av["score_visual"]
    df_av.to_csv(output_scores + ".csv", sep=",", index=False)
    df_av["model_id"] = df_av[["model_id", "image_id"]].agg("@".join, axis=1)

    df_av.to_csv(
        output_scores,
        sep=" ",
        header=False,
        index=False,
        columns=["model_id", "segment_id", "score"],
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="fuses audio and visual scores")
    parser.add_argument(
        "--ndx-file",
        required=True,
    )
    parser.add_argument("--audio-scores", required=True)
    parser.add_argument("--visual-scores", required=True)
    parser.add_argument("--output-scores", required=True)

    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )
    args = parser.parse_args()
    sum_fusion(**namespace_to_dict(args))
