#!/usr/bin/env python
# prepare_adi17.py --corpus-dir /export/corpora6/ADI17 --output-dir data/adi17 --map-langs-to-lre-co


import sys
print(sys.version)

from jsonargparse import ArgumentParser, namespace_to_dict, ActionYesNo

import logging

import glob
import numpy as np
import pandas as pd
from pathlib import Path

from hyperion.hyp_defs import config_logger
adc_map = {
    "ALG": "ara-arq",
    "EGY": "ara-arz",
    "IRA": "ara-acm",
    "JOR": "ara-jor",
    "KSA": "ara-ksa",
    "KUW": "ara-kuw",
    "LEB": "ara-leb",
    "LIB": "ara-ayl",
    "MAU": "ara-mau",
    "MOR": "ara-mor",
    "OMA": "ara-oma",
    "PAL": "ara-pal",
    "QAT": "ara-qat",
    "SUD": "ara-sud",
    "SYR": "ara-syr",
    "UAE": "ara-uae",
    "YEM": "ara-yem"
}

def map_to_adc(langs):
    return [adc_map[l] for l in langs]

def make_kaldi(df, wav_dir, output_dir, target_fs):
    # make wav.scp
    logging.info("making wav.scp")
    with open(output_dir / "wav.scp", "w") as f:
        for _, row in df.iterrows():
            segment_id = row["id"]
            filename = row["filename"]
            if target_fs != 16000:
                wav = "sox {} -t wav -r {} - |".format(filename, target_fs)
            else:
                wav = filename

            f.write("{} {}\n".format(segment_id, wav))

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


def prepare_adi17(corpus_dir, output_dir, remove_langs, map_langs_to_lre_codes,target_fs, verbose):
    config_logger(verbose)
    logging.info("Preparing corpus %s -> %s", corpus_dir, output_dir)
    corpus_dir = Path(corpus_dir)
    wav_dir = corpus_dir
    train_files = glob.glob(str(corpus_dir / "train_segments/*/*.wav"),
                            recursive=True)
    train_ids = [Path(f).stem for f in train_files]
    train_langs = [Path(f).parent.stem for f in train_files]
    dev_files = glob.glob(str(corpus_dir / "dev_segments/*.wav"),
                          recursive=True)
    test_files = glob.glob(str(corpus_dir / "test_segments/*.wav"),
                           recursive=True)
    dev_test_files = dev_files + test_files
    df_labels = pd.concat([
        pd.read_csv(str(corpus_dir / "adi17_official_dev_label.txt"),
                    delim_whitespace=True),
        pd.read_csv(str(corpus_dir / "adi17_official_test_label.txt"),
                    delim_whitespace=True)
    ])
    df_labels = df_labels.set_index("id")
    dev_test_ids = [Path(f).stem for f in dev_test_files]
    dev_test_langs = df_labels.loc[dev_test_ids, "label"].values
    all_ids = train_ids + dev_test_ids
    all_files = train_files + dev_test_files
    all_langs = list(train_langs) + list(dev_test_langs)
    if map_langs_to_lre_codes:
        all_langs = map_to_adc(all_langs)

    all_ids_temp = []
    for a, b in zip(all_langs, all_ids):
        formatted_string = '{}-{}'.format(a, b)
        all_ids_temp.append(formatted_string)
    all_ids= all_ids_temp    
    #all_ids = [f'{a}-{b}' for a, b in zip(all_langs, all_ids)]
    df = pd.DataFrame({
        "id": all_ids,
        "language": all_langs,
        "filename": all_files
    })
    if remove_langs is not None:
        for lang in remove_langs:
            df = df[df["language"] != lang]

    df["sample_coding"] = "pcm"
    df["source"] = "afv"
    df["corpus_id"] = corpus_dir.stem
    df["sample_rate"] = target_fs

    # sort by segment id
    df.sort_values(by="id", inplace=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / "segments.csv"
    logging.info("saving %s", output_file)
    df.drop(["filename"], axis=1).to_csv(output_file, sep=",", index=False)

    make_kaldi(df, wav_dir, output_dir, target_fs)


if __name__ == "__main__":

    parser = ArgumentParser(description="Prepares ADI17 for training")
    parser.add_argument("--corpus-dir",
                        required=True,
                        help="Path to the original dataset")
    parser.add_argument("--output-dir", required=True, help="data path")
    parser.add_argument("--remove-langs",
                        default=None,
                        nargs="+",
                        help="languages to remove")
    parser.add_argument(
        "--map-langs-to-lre-codes",
        default=False,
        action=ActionYesNo,
        help="use LRE17 language codes",
    )

    parser.add_argument("--target-fs",
                        default= 16000,
                        type=int,
                        help="Target sampling frequency")
    parser.add_argument("-v",
                        "--verbose",
                        dest="verbose",
                        default=1,
                        choices=[0, 1, 2, 3],
                        type=int)
    args = parser.parse_args()

    prepare_adi17(**namespace_to_dict(args))