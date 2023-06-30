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


def prepare_voxsrc22_test(corpus_dir, output_dir, verbose):
    config_logger(verbose)
    logging.info(
        "Preparing corpus %s -> %s", corpus_dir, output_dir,
    )
    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials_file = corpus_dir / "Track12_blind.txt"
    df_trials = pd.read_csv(
        trials_file, header=None, names=["enroll", "test"], sep=" ",
    )
    trials_file = output_dir / "trials"
    logging.info("creating trials file %s", trials_file)
    with open(trials_file, "w") as f:
        for _, row in df_trials.iterrows():
            f.write("%s %s\n" % (row["enroll"], row["test"]))

    enroll_file = output_dir / "utt2model"
    logging.info("creating enrollment file %s", enroll_file)
    file_ids = df_trials["enroll"].unique()
    with open(enroll_file, "w") as f:
        for file_id in file_ids:
            f.write("%s %s\n" % (file_id, file_id))

    u2s_file = output_dir / "utt2spk"
    logging.info("creating utt2spk file %s", u2s_file)
    file_ids = np.unique(np.concatenate((df_trials["enroll"], df_trials["test"])))
    with open(u2s_file, "w") as f:
        for file_id in file_ids:
            f.write("%s %s\n" % (file_id, file_id))

    s2u_file = output_dir / "spk2utt"
    logging.info("creating spk2utt file %s", s2u_file)
    with open(s2u_file, "w") as f:
        for file_id in file_ids:
            f.write("%s %s\n" % (file_id, file_id))

    wav_file = output_dir / "wav.scp"
    logging.info("creating wav.scp file %s", wav_file)
    with open(wav_file, "w") as f:
        for file_id in file_ids:
            wav_file = corpus_dir / "Track12_test_data" / file_id
            f.write("%s %s\n" % (file_id, wav_file))


if __name__ == "__main__":

    parser = ArgumentParser(description="Prepares VoxSRC22 Track1/2 test data")

    parser.add_argument("--corpus-dir", required=True, help="Path to voxsrc22 dataset")

    parser.add_argument("--output-dir", required=True, help="Ouput data path prefix")
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )
    args = parser.parse_args()
    prepare_voxsrc22_test(**namespace_to_dict(args))
