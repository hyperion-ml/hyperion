#!/bin/env python
"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ArgumentParser, namespace_to_dict
import logging
from pathlib import Path
import re
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
        "language",
        "duration",
    ]
    files = [
        "utt2spk",
        "spk2utt",
        "utt2lang",
        "utt2speech_dur",
    ]
    for c, f in zip(columns, files):
        output_file = output_dir / f
        if c in df:
            df.to_csv(
                output_file, sep=" ", columns=["id", c], header=False, index=False
            )


def prepare_lre17(corpus_dir, subset, source, output_dir, target_fs, verbose):
    config_logger(verbose)
    logging.info("Preparing corpus %s - %s -> %s", corpus_dir, subset, output_dir)
    corpus_dir = Path(corpus_dir)
    wav_dir = corpus_dir / "data" / subset
    if subset == "eval":
        table_info = corpus_dir / "docs" / f"lre17_eval_segment_keys.tsv"
    else:
        table_info = corpus_dir / "docs" / f"{subset}_info.tab"
    df = pd.read_csv(table_info, sep="\t")
    df.rename(
        columns={
            "language_code": "language",
            "segmentid": "id",
            "file_duration": "duration",
        },
        inplace=True,
    )

    if subset == "eval":
        df["data_source"] = df["data_source"].str.lower()
        df["sample_coding"] = df["data_source"].apply(
            lambda x: "mulaw" if x == "mls14" else "pcm"
        )
        df.loc[df["speech_duration"].isnull(), "speech_duration"] = 1000
        df["length_condition"] = df.pop("speech_duration").astype("int32")

    if subset in ["dev", "eval"]:
        # drop files of 3 and 10 secs since they are contained in the files of 30 secs
        df = df[df["length_condition"] > 10]
        if source != "all":
            df = df[df["data_source"] == source]

    # move segment column to first positon
    first_col = df.pop("id")
    df.insert(0, "id", first_col)

    # sort by segment id
    df.sort_values(by="id", inplace=True)

    if subset == "train":
        df["filename"] = df.apply(lambda x: wav_dir / x.language / x.id, axis=1)
    else:
        df["filename"] = df.apply(lambda x: wav_dir / x.id, axis=1)
    df["source"] = df["id"].apply(lambda x: "cts" if re.match(r".*\.sph", x) else "afv")
    df["corpus_id"] = "lre17"
    df["sample_rate"] = target_fs

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / "segments.csv"
    logging.info("saving %s", output_file)
    df.drop(["filename"], axis=1).to_csv(output_file, sep=",", index=False)

    make_kaldi(df, wav_dir, output_dir, target_fs)


if __name__ == "__main__":

    parser = ArgumentParser(description="Prepares LDC2022E16/17 LRE17 for training")
    parser.add_argument(
        "--corpus-dir", required=True, help="Path to the original dataset"
    )
    parser.add_argument(
        "--subset",
        required=True,
        help="train/dev/eval",
        choices=["train", "dev", "eval"],
    )
    parser.add_argument(
        "--source",
        default="all",
        help="all/mls14/vast",
        choices=["all", "mls14", "vast"],
    )

    parser.add_argument("--output-dir", required=True, help="data path")
    parser.add_argument(
        "--target-fs", default=8000, type=int, help="Target sampling frequency"
    )
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )
    args = parser.parse_args()
    prepare_lre17(**namespace_to_dict(args))
