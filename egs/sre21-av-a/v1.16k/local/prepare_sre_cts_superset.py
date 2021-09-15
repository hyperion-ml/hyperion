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

multigender_spks = [
    "111774",
    "111781",
    "112778",
    "112783",
    "112879",
    "113153",
    "113213",
    "113603",
    "128673",
    "128770",
]


def fix_multigender_spks(df):

    logging.info("Fixing multigender speakers")
    n0 = len(df)
    for spk in multigender_spks:
        male_idx = (df["speaker_id"] == spk) & (df["gender"] == "male")
        female_idx = (df["speaker_id"] == spk) & (df["gender"] == "female")
        num_male = np.sum(male_idx)
        num_female = np.sum(female_idx)
        if num_male > num_female:
            df = df[~female_idx]
        else:
            df = df[~male_idx]

    logging.info("Fixed multigender speakers, %d/%d segments remained", len(df), n0)
    return df


def prepare_sre_cts_superset(corpus_dir, output_dir, target_fs, verbose):
    config_logger(verbose)
    logging.info("Preparing corpus %s -> %s", corpus_dir, output_dir)
    wav_dir = Path(corpus_dir) / "data"
    table_file = Path(corpus_dir) / "docs/cts_superset_segment_key.tsv"
    df = pd.read_csv(table_file, sep="\t")
    df.drop(["segmentid", "speakerid"], axis=1, inplace=True)
    df.rename(
        columns={
            "subjectid": "speaker_id",
            "sessionid": "session_id",
            "corpusid": "corpus_id",
            "phoneid": "phone_id",
        },
        inplace=True,
    )
    df["speaker_id"] = df["speaker_id"].astype("str")
    df = fix_multigender_spks(df)

    df["segment_id"] = df["filename"].str.replace("/", "-")
    # put segment_id as first columnt
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    logging.info("sorting by segment_id")
    df.sort_values(by="segment_id", inplace=True)

    logging.info("saving segments.csv")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "segments.csv"
    df.drop(["filename"], axis=1).to_csv(output_file, sep=",", index=False)

    # Kaldi data directory files
    # utt2xxx files
    logging.info("saving Kaldi utt2xxx files")
    columns = [
        "speaker_id",
        "speech_duration",
        "session_id",
        "corpus_id",
        "phone_id",
        "language",
    ]
    files = [
        "utt2spk",
        "utt2speech_dur",
        "utt2session",
        "utt2corpus",
        "utt2phone",
        "utt2lang",
    ]
    for c, f in zip(columns, files):
        output_file = output_dir / f
        df.to_csv(
            output_file, sep=" ", columns=["segment_id", c], header=False, index=False
        )

    # make wav.scp
    logging.info("making wav.scp")
    with open(output_dir / "wav.scp", "w") as f:
        for _, row in df.iterrows():
            segment_id = row["segment_id"]
            filename = row["filename"]
            wav = f"sph2pipe -f wav -p -c 1 {wav_dir}/{filename} |"
            if target_fs != 8000:
                wav = f"{wav} sox -t wav - -t wav -r {target_fs} - |"
            f.write(f"{segment_id} {wav}\n")

    # speaker table
    logging.info("saving speaker files")
    spk_df = df[["speaker_id", "gender"]].drop_duplicates()
    output_file = output_dir / "speaker.csv"
    spk_df.to_csv(output_file, sep=",", index=False)
    gender = df["gender"].str.replace("female", "f").str.replace("male", "m")
    spk_df["gender"] = gender
    output_file = output_dir / "spk2gender"
    spk_df.to_csv(output_file, sep=" ", header=False, index=False)

    with open(output_dir / "spk2utt", "w") as f:
        for spk in df["speaker_id"].unique():
            segment_ids = " ".join(df[df["speaker_id"] == spk].segment_id.values)
            f.write(f"{spk} {segment_ids}\n")


if __name__ == "__main__":

    parser = ArgumentParser(description="Prepares SRE superset LDC2021E08")

    parser.add_argument(
        "--corpus-dir", required=True, help="Path to the original dataset"
    )
    parser.add_argument("--output-dir", required=True, help="Ouput data path")
    parser.add_argument(
        "--target-fs", default=8000, type=int, help="Target sampling frequency"
    )
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )
    args = parser.parse_args()
    prepare_sre_cts_superset(**namespace_to_dict(args))
