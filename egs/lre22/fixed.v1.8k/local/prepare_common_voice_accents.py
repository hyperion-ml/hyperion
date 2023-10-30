#!/usr/bin/env python
# prepare_common_voice.py --corpus-dir /export/corpora6/LRE/CommonVoice2020 --output-dir data/cv --map-langs-to-lre-codes --target-fs 8000
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

en_map = {"indian": "eng-ine"}
fr_map = {
    "france": "fra-fra",
    "canada": "fra-can",
    "algeria": "fra-ntf",
    "morocco": "fra-ntf",
    "tunisia": "fra-ntf",
}

lre_map = {
    "en": en_map,
    "fr": fr_map,
}


def make_kaldi(df, wav_dir, output_dir, target_fs):
    # make wav.scp
    logging.info("making wav.scp")
    with open(output_dir / "wav.scp", "w") as f:
        for _, row in df.iterrows():
            segment_id = row["id"]
            filename = row["filename"]
            if target_fs != 16000:
                wav = "ffmpeg -i {} -acodec pcm_s16le -ar {} -f wav - |".format(
                    filename, target_fs)
            else:
                wav = "ffmpeg -i {} -acodec pcm_s16le -f wav - |".format(
                    filename)

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


def prepare_common_voice(corpus_dir, output_dir, lang, target_fs, verbose):
    config_logger(verbose)
    logging.info("Preparing corpus %s -> %s", corpus_dir, output_dir)
    corpus_dir = Path(corpus_dir)
    wav_dir = corpus_dir
    my_map = lre_map[lang]
    df = pd.read_csv(corpus_dir / lang / "validated.tsv", sep="\t")
    mask = None
    for dialect in my_map.keys():
        mask_d = df["accent"] == dialect
        if mask is None:
            mask = mask_d
        else:
            mask = np.logical_or(mask, mask_d)

    df = df.loc[mask]
    files = df["path"]
    files = [corpus_dir / lang / "clips" / f for f in df["path"]]
    langs = [my_map[l] for l in df["accent"]]
    ids = ["{}-{}".format(l, Path(f).stem) for f, l in zip(files, langs)]
    df = pd.DataFrame({"id": ids, "language": langs, "filename": files})

    df["sample_coding"] = "pcm"
    df["source"] = "afv"
    df["corpus_id"] = "cv"
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

    parser = ArgumentParser(
        description="Prepares Common Voice Accents for training in LRE22")
    parser.add_argument("--corpus-dir",
                        required=True,
                        help="Path to the original dataset")
    parser.add_argument("--output-dir", required=True, help="data path")
    parser.add_argument("--lang",
                        default="en",
                        choices=["en", "fr"],
                        help="languages")

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
    prepare_common_voice(**namespace_to_dict(args))
