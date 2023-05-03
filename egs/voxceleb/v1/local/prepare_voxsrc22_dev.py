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


def prepare_voxsrc22_dev(vox1_corpus_dir, voxsrc22_corpus_dir, output_dir, verbose):
    config_logger(verbose)
    logging.info(
        "Preparing corpus %s + %s -> %s",
        vox1_corpus_dir,
        voxsrc22_corpus_dir,
        output_dir,
    )
    vox1_corpus_dir = Path(vox1_corpus_dir)
    voxsrc22_corpus_dir = Path(voxsrc22_corpus_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials_file = voxsrc22_corpus_dir / "voxsrc2022_dev.txt"
    df_trials = pd.read_csv(
        trials_file, header=None, names=["target", "enroll", "test"], sep=" ",
    )

    trials_file = output_dir / "trials"
    logging.info("creating trials file %s", trials_file)
    with open(trials_file, "w") as f:
        for _, row in df_trials.iterrows():
            t = "target" if row["target"] == 1 else "nontarget"
            f.write("%s %s %s\n" % (row["enroll"], row["test"], t))

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
            if "VoxSRC2022_dev" in file_id:
                wav_file = voxsrc22_corpus_dir / file_id
            else:
                wav_file = vox1_corpus_dir / "wav" / file_id

            f.write("%s %s\n" % (file_id, wav_file))


if __name__ == "__main__":

    parser = ArgumentParser(description="Prepares VoxSRC22 Track1/2 validation data")

    parser.add_argument(
        "--vox1-corpus-dir", required=True, help="Path to voxceleb1 v2 dataset"
    )
    parser.add_argument(
        "--voxsrc22-corpus-dir", required=True, help="Path to voxsrc22 dataset"
    )

    parser.add_argument("--output-dir", required=True, help="Ouput data path prefix")
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )
    args = parser.parse_args()
    prepare_voxsrc22_dev(**namespace_to_dict(args))
