#!/bin/env python
"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ArgumentParser, namespace_to_dict
import logging
from pathlib import Path
import pandas as pd


def merge_scores(in_score_files, out_score_file):

    dfs = []
    for f in in_score_files:
        df_f = pd.read_csv(f, sep="\t")
        dfs.append(df_f)

    df = pd.concat(dfs)
    df.sort_values(by="segmentid", inplace=True)
    df.to_csv(out_score_file, sep="\t", index=False)


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Split Segment list into training and validation"
    )
    parser.add_argument("--in-score-files", nargs="+", required=True)
    parser.add_argument("--out-score-file", required=True)
    args = parser.parse_args()
    merge_scores(**namespace_to_dict(args))
