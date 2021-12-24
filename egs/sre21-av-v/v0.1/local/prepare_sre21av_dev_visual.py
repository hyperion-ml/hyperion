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


def make_enroll_dir(df_segms, img_dir, output_path):
    enroll_dir = Path(output_path + "_enroll")
    img_dir = img_dir / "enrollment"
    logging.info("making enrollment dir %s", enroll_dir)
    enroll_dir.mkdir(parents=True, exist_ok=True)
    df_segms = (
        df_segms[
            (df_segms["partition"] == "enrollment") & (df_segms["source_type"] == "na")
        ]
        .drop(["partition", "source_type", "language"], axis=1)
        .sort_values(by="segment_id")
    )
    segment_file = enroll_dir / "segments.csv"
    df_segms.to_csv(segment_file, sep=",", index=False)

    with open(enroll_dir / "utt2spk", "w") as f1, open(
        enroll_dir / "spk2utt", "w"
    ) as f2:
        for u in df_segms["segment_id"]:
            f1.write(f"{u} {u}\n")
            f2.write(f"{u} {u}\n")

    with open(enroll_dir / "vid.scp", "w") as f:
        for u in df_segms["segment_id"]:
            f.write(f"{u} {img_dir}/{u}\n")


def write_simple_trialfile(df_key, output_file):
    df_key.to_csv(
        output_file,
        sep=" ",
        columns=["model_id", "segment_id", "targettype"],
        index=False,
        header=False,
    )


def make_test_dir(df_segms, df_key, vid_dir, output_path):
    test_dir = Path(output_path + "_test")
    vid_dir = vid_dir / "test"
    logging.info("making test dir %s", test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    df_segms = (
        df_segms[(df_segms["partition"] == "test") & (df_segms["source_type"] == "na")]
        .drop(["partition", "source_type", "language"], axis=1)
        .sort_values(by="segment_id")
    )
    segment_file = test_dir / "segments.csv"
    df_segms.to_csv(segment_file, sep=",", index=False)

    with open(test_dir / "utt2spk", "w") as f1, open(test_dir / "spk2utt", "w") as f2:
        for u in df_segms["segment_id"]:
            f1.write(f"{u} {u}\n")
            f2.write(f"{u} {u}\n")

    with open(test_dir / "vid.scp", "w") as f:
        for u in df_segms["segment_id"]:
            f.write(f"{u} {vid_dir}/{u}\n")

    df_key.to_csv(test_dir / "trials.csv", sep=",", index=False)
    write_simple_trialfile(df_key, test_dir / "trials")


def prepare_sre21av_dev_visual(corpus_dir, output_path, verbose):
    config_logger(verbose)
    logging.info("Preparing corpus %s -> %s", corpus_dir, output_path)
    corpus_dir = Path(corpus_dir)
    img_dir = corpus_dir / "data" / "image"
    vid_dir = corpus_dir / "data" / "video"
    segments_file = corpus_dir / "docs" / "sre21_dev_segment_key.tsv"
    df_segms = pd.read_csv(segments_file, sep="\t")
    df_segms.rename(
        columns={"segmentid": "segment_id", "subjectid": "speaker_id"},
        inplace=True,
    )

    key_file = corpus_dir / "docs" / "sre21_visual_dev_trial_key.tsv"
    df_key = pd.read_csv(key_file, sep="\t")
    df_key.rename(
        columns={"segmentid": "segment_id", "imageid": "model_id"},
        inplace=True,
    )

    make_enroll_dir(df_segms, img_dir, output_path)
    make_test_dir(df_segms, df_key, vid_dir, output_path)


if __name__ == "__main__":

    parser = ArgumentParser(description="Prepares SRE21 dev visual part")

    parser.add_argument(
        "--corpus-dir", required=True, help="Path to the original dataset"
    )
    parser.add_argument("--output-path", required=True, help="Ouput data path prefix")
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )
    args = parser.parse_args()
    prepare_sre21av_dev_visual(**namespace_to_dict(args))
