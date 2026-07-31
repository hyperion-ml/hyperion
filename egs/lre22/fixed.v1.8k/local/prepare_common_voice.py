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

lre_map = {
    "afr": "afr-afr",
    "ndb": "nbl-nbl",
    "oro": "orm-orm",
    "tso": "tso-tso",
    "ven": "ven-ven",
    "xho": "xho-xho",
    "zul": "zul-zul",
    "tig": "tir-tir",
    "sae": "eng-ens",
    "ine": "eng-ine",
    "tun": "ara-aeb",
    "alg": "ara-arq",
    "lib": "ara-ayl",
    "naf": "fra-ntf"
}


def map_to_lre(langs):
    return [
        lre_map[l] if l in lre_map else "{}-{}".format(l, l) for l in langs
    ]


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


def prepare_common_voice(corpus_dir, output_dir, keep_langs,
                         map_langs_to_lre_codes, target_fs, verbose):
    config_logger(verbose)
    logging.info("Preparing corpus %s -> %s", corpus_dir, output_dir)
    corpus_dir = Path(corpus_dir)
    wav_dir = corpus_dir
    files = glob.glob(str(corpus_dir / "**/clips/*.mp3"), recursive=True)
    langs = [(Path(f).parent.parent.stem).lower() for f in files]
    if map_langs_to_lre_codes:
        langs = map_to_lre(langs)
    ids = ["{}-{}".format(l, Path(f).stem) for f, l in zip(files, langs)]
    df = pd.DataFrame({"id": ids, "language": langs, "filename": files})
    df = df[df["language"].isin(keep_langs)]
    # if remove_langs is not None:
    #     for lang in remove_langs:
    #         df = df[df["language"] != lang]

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

    parser = ArgumentParser(description="Prepares Common Voice for training")
    parser.add_argument("--corpus-dir",
                        required=True,
                        help="Path to the original dataset")
    parser.add_argument("--output-dir", required=True, help="data path")
    parser.add_argument("--keep-langs",
                        default=["tir-tir"],
                        nargs="+",
                        help="languages to keep")

    # parser.add_argument("--remove-langs",
    #                     default=None,
    #                     nargs="+",
    #                     help="languages to remove")
    parser.add_argument(
        "--map-langs-to-lre-codes",
        default=False,
        action=ActionYesNo,
        help="use LRE17 language codes",
    )

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
