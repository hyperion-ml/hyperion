#!/bin/env python
"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ArgumentParser, namespace_to_dict
import logging
from pathlib import Path
import glob
import re
import numpy as np
import pandas as pd

from hyperion.hyp_defs import config_logger


def make_kaldi(df, output_dir, target_fs):
    # make wav.scp
    logging.info("making wav.scp")
    with open(output_dir / "wav.scp", "w") as f:
        for _, row in df.iterrows():
            segment_id = row["id"]
            filename = row["filename"]
            source = row["source"]
            if source == "cts":
                wav = f"sph2pipe -f wav -p -c 1 {filename} |"
                if target_fs != 8000:
                    wav = f"{wav} sox -t wav - -t wav -r {target_fs} - |"
            else:
                wav = f"sox {filename}  -t wav -r {target_fs} - |"

            f.write(f"{segment_id} {wav}\n")

    # Kaldi data directory files
    # utt2xxx files
    logging.info("saving Kaldi utt2xxx files")
    columns = [
        "id",
        "id",
        "language",
    ]
    files = [
        "utt2spk",
        "spk2utt",
        "utt2lang",
    ]
    for c, f in zip(columns, files):
        output_file = output_dir / f
        if c in df:
            df.to_csv(output_file,
                      sep=" ",
                      columns=["id", c],
                      header=False,
                      index=False)


def prepare_babel(corpus_dir, lang_code, output_dir, target_fs, verbose):
    config_logger(verbose)
    logging.info("Preparing corpus %s -> %s", corpus_dir, output_dir)
    corpus_dir = Path(corpus_dir)
    logging.info("searching audio files")
    wavs = glob.glob(str(corpus_dir / "**/audio/*.sph"), recursive=True)
    logging.info("found %d files", len(wavs))
    wavs = [corpus_dir / w for w in wavs]
    seg_ids = [w.stem for w in wavs]
    df = pd.DataFrame({"id": seg_ids, "filename": wavs})

    # sort by segment id
    df.sort_values(by="id", inplace=True)
    df["corpus_id"] = "babel"
    df["sample_rate"] = target_fs
    df["language"] = lang_code
    df["source"] = "cts"
    logging.info("saving files")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / "segments.csv"
    logging.info("saving %s", output_file)
    df.drop(["filename"], axis=1).to_csv(output_file, sep=",", index=False)

    make_kaldi(df, output_dir, target_fs)


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Prepares Babel datasets for training in LRE")
    parser.add_argument("--corpus-dir",
                        required=True,
                        help="Path to the original dataset")
    parser.add_argument(
        "--lang-code",
        required=True,
        help="language code",
    )
    parser.add_argument("--output-dir", required=True, help="data path")
    parser.add_argument("--target-fs",
                        default=8000,
                        type=int,
                        help="Target sampling frequency")
    parser.add_argument("-v",
                        "--verbose",
                        dest="verbose",
                        default=1,
                        choices=[0, 1, 2, 3],
                        type=int)
    args = parser.parse_args()
    prepare_babel(**namespace_to_dict(args))
