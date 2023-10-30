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


def make_kaldi(df, wav_dir, output_dir, target_fs):
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
    ]
    files = [
        "utt2spk",
        "spk2utt",
    ]
    for c, f in zip(columns, files):
        output_file = output_dir / f
        if c in df:
            df.to_csv(
                output_file, sep=" ", columns=["id", c], header=False, index=False
            )


def prepare_lre22(corpus_dir, output_dir, target_fs, verbose):
    config_logger(verbose)
    subset = "eval"
    logging.info("Preparing corpus %s -> %s", corpus_dir, output_dir)
    corpus_dir = Path(corpus_dir)
    wav_dir = corpus_dir / "data" / subset
    table_info = corpus_dir / "docs" / "lre22_eval_trials.tsv"
    df = pd.read_csv(table_info, sep="\t")
    df.rename(
        columns={
            "segmentid": "id",
        },
        inplace=True,
    )

    # sort by segment id
    df.sort_values(by="id", inplace=True)

    df["filename"] = df.apply(lambda x: wav_dir / f"{x.id}.sph", axis=1)
    df["source_coding"] = "alaw"
    df["source"] = "cts"
    df["corpus_id"] = "lre22"
    df["sample_rate"] = target_fs

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / "segments.csv"
    logging.info("saving %s", output_file)
    df.drop(["filename"], axis=1).to_csv(output_file, sep=",", index=False)

    make_kaldi(df, wav_dir, output_dir, target_fs)


if __name__ == "__main__":

    parser = ArgumentParser(description="Prepares LRE22 eval data")
    parser.add_argument(
        "--corpus-dir", required=True, help="Path to the original dataset"
    )
    parser.add_argument("--output-dir", required=True, help="data path")
    parser.add_argument(
        "--target-fs", default=8000, type=int, help="Target sampling frequency"
    )
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )
    args = parser.parse_args()
    prepare_lre22(**namespace_to_dict(args))
