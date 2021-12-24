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


def make_enroll_dir(df_trials, img_dir, output_path):
    enroll_dir = Path(output_path + "_enroll")
    img_dir = img_dir / "enrollment"
    logging.info("making enrollment dir %s", enroll_dir)
    enroll_dir.mkdir(parents=True, exist_ok=True)
    segments = df_trials["model_id"].sort_values().unique()
    with open(enroll_dir / "utt2spk", "w") as f1, open(
        enroll_dir / "spk2utt", "w"
    ) as f2:
        for u in segments:
            f1.write(f"{u} {u}\n")
            f2.write(f"{u} {u}\n")

    with open(enroll_dir / "vid.scp", "w") as f:
        for u in segments:
            f.write(f"{u} {img_dir}/{u}\n")


def write_simple_trialfile(df_trials, output_file):
    df_trials.to_csv(
        output_file,
        sep=" ",
        columns=["model_id", "segment_id"],
        index=False,
        header=False,
    )


def make_test_dir(df_trials, vid_dir, output_path):
    test_dir = Path(output_path + "_test")
    vid_dir = vid_dir / "test"
    logging.info("making test dir %s", test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    segments = df_trials["segment_id"].sort_values().unique()

    with open(test_dir / "utt2spk", "w") as f1, open(test_dir / "spk2utt", "w") as f2:
        for u in segments:
            f1.write(f"{u} {u}\n")
            f2.write(f"{u} {u}\n")

    with open(test_dir / "vid.scp", "w") as f:
        for u in segments:
            f.write(f"{u} {vid_dir}/{u}\n")

    df_trials.to_csv(test_dir / "trials.csv", sep=",", index=False)
    write_simple_trialfile(df_trials, test_dir / "trials")


def prepare_sre21av_eval_visual(corpus_dir, output_path, verbose):
    config_logger(verbose)
    logging.info("Preparing corpus %s -> %s", corpus_dir, output_path)
    corpus_dir = Path(corpus_dir)
    img_dir = corpus_dir / "data" / "image"
    vid_dir = corpus_dir / "data" / "video"
    key_file = corpus_dir / "docs" / "sre21_visual_eval_trials.tsv"
    df_trials = pd.read_csv(key_file, sep="\t")
    df_trials.rename(
        columns={"segmentid": "segment_id", "imageid": "model_id"},
        inplace=True,
    )

    make_enroll_dir(df_trials, img_dir, output_path)
    make_test_dir(df_trials, vid_dir, output_path)

    key_file = corpus_dir / "docs" / "sre21_audio-visual_eval_trials.tsv"
    df_trials = pd.read_csv(key_file, sep="\t")
    df_trials = df_trials.drop("modelid", axis=1).drop_duplicates()
    df_trials.rename(
        columns={"segmentid": "segment_id", "imageid": "model_id"},
        inplace=True,
    )
    test_dir = Path(output_path + "_test")
    df_trials.to_csv(test_dir / "trials_av.csv", sep=",", index=False)
    write_simple_trialfile(df_trials, test_dir / "trials_av")


if __name__ == "__main__":

    parser = ArgumentParser(description="Prepares SRE21 eval visual part")

    parser.add_argument(
        "--corpus-dir", required=True, help="Path to the original dataset"
    )
    parser.add_argument("--output-path", required=True, help="Ouput data path prefix")
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )
    args = parser.parse_args()
    prepare_sre21av_eval_visual(**namespace_to_dict(args))
