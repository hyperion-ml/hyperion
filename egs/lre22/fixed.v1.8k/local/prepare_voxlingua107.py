#!/bin/env python
"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ArgumentParser, namespace_to_dict, ActionYesNo
import logging
from pathlib import Path
import glob
import numpy as np
import pandas as pd

from hyperion.hyp_defs import config_logger


lre_map = {
    "pl": "qsl-pol",
    "ru": "qsl-rus",
    "cs": "qsl-cze",
    "uk": "qsl-ukr",
    "hr": "qsl-cro",
    "bg": "qsl-bul",
    "be": "qsl-bel",
    "sk": "qsl-slk",
    "sl": "qsl-slv",
    "bs": "qsl-bos",
    "sr": "qsl-ser",
    "zh": "zho-cmn",
    "fr": "fra-mix",
    "af": "afr-afr",
}


def map_to_lre(langs):
    return [lre_map[l] if l in lre_map else f"{l}-{l}" for l in langs]


def make_kaldi(df, output_dir, target_fs):
    # make wav.scp
    logging.info("making wav.scp")
    with open(output_dir / "wav.scp", "w") as f:
        for _, row in df.iterrows():
            segment_id = row["id"]
            filename = row["filename"]
            if target_fs != 16000:
                wav = f"sox {filename} -t wav -r {target_fs} - |"
            else:
                wav = filename

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
            df.to_csv(
                output_file, sep=" ", columns=["id", c], header=False, index=False
            )


def prepare_voxlingua107(
    corpus_dir, output_dir, remove_langs, map_langs_to_lre_codes, target_fs, verbose
):
    config_logger(verbose)
    logging.info("Preparing corpus %s -> %s", corpus_dir, output_dir)
    corpus_dir = Path(corpus_dir)
    files = glob.glob(str(corpus_dir / "*/*.wav"))
    langs = [Path(f).parent.stem for f in files]
    if map_langs_to_lre_codes:
        langs = map_to_lre(langs)
    ids = [f"{l}-{Path(f).stem}" for f, l in zip(files, langs)]
    df = pd.DataFrame({"id": ids, "language": langs, "filename": files})
    if remove_langs is not None:
        for lang in remove_langs:
            df = df[df["language"] != lang]

    df["sample_coding"] = "pcm"
    df["source"] = "afv"
    df["corpus_id"] = "voxlingua107"
    df["sample_rate"] = target_fs

    # sort by segment id
    df.sort_values(by="id", inplace=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / "segments.csv"
    logging.info("saving %s", output_file)
    df.drop(["filename"], axis=1).to_csv(output_file, sep=",", index=False)

    make_kaldi(df, output_dir, target_fs)


if __name__ == "__main__":

    parser = ArgumentParser(description="Prepares Voxlingua107 for training")
    parser.add_argument(
        "--corpus-dir", required=True, help="Path to the original dataset"
    )
    parser.add_argument("--output-dir", required=True, help="data path")
    parser.add_argument(
        "--remove-langs", default=None, nargs="+", help="languages to remove"
    )
    parser.add_argument(
        "--map-langs-to-lre-codes",
        default=False,
        action=ActionYesNo,
        help="use LRE17 language codes",
    )

    parser.add_argument(
        "--target-fs", default=16000, type=int, help="Target sampling frequency"
    )
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )
    args = parser.parse_args()
    prepare_voxlingua107(**namespace_to_dict(args))
