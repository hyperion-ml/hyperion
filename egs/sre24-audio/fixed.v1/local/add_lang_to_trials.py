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
from hyperion.utils import SegmentSet, EnrollmentMap
from hyperion.data_prep.sre24 import LangTrialCond


def add_lang_to_trials(
    enroll_map_file, enroll_segments_file, test_segments_file, ndx_file
):
    ndx_file = Path(ndx_file)
    bk_file = ndx_file.with_suffix(ndx_file.suffix + ".bk")
    if not bk_file.is_file():
        shutil.copy2(ndx_file, bk_file)

    sep = "\t" if ndx_file.suffix == ".tsv" else ","
    df_trials = pd.read_csv(ndx_file, sep=sep)

    enroll_map = EnrollmentMap.load(enroll_map_file)
    enroll_segments = SegmentSet.load(enroll_segments_file)
    test_segments = SegmentSet.load(test_segments_file)

    enroll_map.add_columns(
        right_table=enroll_segments[["id", "language"]], on="segmentid", right_on="id"
    )
    df_trials = df_trials.merge(
        test_segments[["id", "language"]],
        how="inner",
        left_on="segmentid",
        right_index=True,
    )
    df_trials.rename(columns={"language": "test_language"}, inplace=True)
    df_trials.drop(columns=["id"], inplace=True)
    df_enr = (
        enroll_map.df.groupby(enroll_map.index)["language"]
        .agg(pd.Series.mode)
        .to_frame()
    )
    idx = df_enr["language"].isin(["ARA", "ENG", "FRA"])
    df_enr.loc[~idx, "language"] = "ARA"
    df_trials = df_trials.merge(
        df_enr,
        how="inner",
        left_on="modelid",
        right_on="id",
    )
    df_trials.rename(columns={"language": "enroll_language"}, inplace=True)
    df_trials["language_match"] = "N"
    df_trials.loc[
        df_trials["enroll_language"] == df_trials["test_language"], "language_match"
    ] = "Y"

    df_trials["language"] = df_trials.apply(
        lambda x: LangTrialCond.get_trial_cond(
            x.enroll_language, x.test_language
        ).value,
        axis=1,
    )
    df_trials.to_csv(ndx_file, sep=sep, index=False)


def main():
    parser = ArgumentParser(
        description="Adds language to trials files",
    )
    parser.add_argument("--enroll-map-file", required=True)
    parser.add_argument("--enroll-segments-file", required=True)
    parser.add_argument("--test-segments-file", required=True)
    parser.add_argument("--ndx-file", required=True)

    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )
    args = parser.parse_args()
    config_logger(args.verbose)
    logging.debug(args)
    del args["verbose"]

    add_lang_to_trials(**namespace_to_dict(args))


if __name__ == "__main__":
    main()
