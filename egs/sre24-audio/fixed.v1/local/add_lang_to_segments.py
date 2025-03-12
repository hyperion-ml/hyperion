#!/usr/bin/env python
""" 
 Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba) 
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""
import logging
import os
import sys
import time
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ActionYesNo,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger
from hyperion.utils import SegmentSet


def add_lang_to_segments(segments_file, score_file):
    segments_file = Path(segments_file)
    score_file = Path(score_file)
    bk_file = segments_file.with_suffix(segments_file.suffix + ".bk")
    if not bk_file.is_file():
        shutil.copy2(segments_file, bk_file)

    sep = "\t" if score_file.suffix == ".tsv" else ","
    df_scores = pd.read_csv(score_file, sep=sep)

    lang_columns = list(df_scores.columns)
    lang_columns.remove("id")
    scores = np.asarray(df_scores[lang_columns])
    lang_pred = np.argmax(scores, axis=1)
    lang_columns = np.asarray(lang_columns)
    df_scores["language"] = lang_columns[lang_pred]
    df_scores.set_index("id", inplace=True)
    segments = SegmentSet.load(segments_file)
    segments["language"] = df_scores.loc[segments["id"], "language"]
    segments.save(segments_file)


def main():
    parser = ArgumentParser(
        description="Adds language column to segments file from lang id"
    )
    parser.add_argument("--segments-file", required=True)
    parser.add_argument("--score-file", required=True)

    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )
    args = parser.parse_args()
    config_logger(args.verbose)
    logging.debug(args)
    del args["verbose"]

    add_lang_to_segments(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
